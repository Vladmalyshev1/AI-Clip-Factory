"""Transcript endpoints: read a transcript, page its segments, start a run.

Routes stay thin: they validate the state of the video, delegate to
:mod:`app.services.transcript_service` and :mod:`app.services.job_service`, and
convert rows into API schemas. The queue owns the work, so ``POST`` answers
``202`` with the transcript and the new ``job_id`` instead of blocking.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi import status as http_status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.asr import transcription_options
from app.api.deps import DbSessionDep, SettingsDep
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.status import TranscriptStatus, VideoStatus
from app.models.transcript import Transcript
from app.models.video import Video
from app.schemas.common import Page
from app.schemas.transcript import (
    TranscribeRequest,
    TranscriptProgressRead,
    TranscriptRead,
    TranscriptSegmentRead,
    segment_to_read,
    transcript_to_read,
)
from app.services import job_service, transcript_service, video_service

router = APIRouter(prefix="/videos", tags=["transcripts"])

LimitQuery = Annotated[int, Query(ge=1, le=500)]
OffsetQuery = Annotated[int, Query(ge=0)]
TimeQuery = Annotated[int | None, Query(ge=0)]


def _progress_read(session: Session, transcript: Transcript) -> TranscriptProgressRead:
    """Build the chunk progress snapshot of ``transcript``."""
    snapshot = transcript_service.progress(session, transcript)
    return TranscriptProgressRead(
        chunks_total=snapshot.chunks_total,
        chunks_completed=snapshot.chunks_completed,
        chunks_running=snapshot.chunks_running,
        chunks_failed=snapshot.chunks_failed,
        audio_ms_total=snapshot.audio_ms_total,
        audio_ms_processed=snapshot.audio_ms_processed,
        progress=snapshot.progress,
        current_chunk_index=snapshot.current_chunk_index,
    )


def _read(
    session: Session, transcript: Transcript, *, job_id: str | None = None
) -> TranscriptRead:
    """Serialise a transcript with its progress and segment count."""
    return transcript_to_read(
        transcript,
        progress=_progress_read(session, transcript),
        segment_count=transcript_service.count_segments(session, transcript.id),
        job_id=job_id,
    )


@router.get("/{video_id}/transcript", response_model=TranscriptRead)
def get_transcript(video_id: str, session: DbSessionDep) -> TranscriptRead:
    """Return the transcript of a video (404 until it was started once)."""
    transcript = transcript_service.get_transcript(session, video_id)
    job = transcript_service.latest_asr_job(session, video_id)
    return _read(session, transcript, job_id=job.id if job is not None else None)


@router.get("/{video_id}/transcript/segments", response_model=Page[TranscriptSegmentRead])
def list_transcript_segments(
    video_id: str,
    session: DbSessionDep,
    limit: LimitQuery = transcript_service.DEFAULT_SEGMENT_LIMIT,
    offset: OffsetQuery = 0,
    start_ms: TimeQuery = None,
    end_ms: TimeQuery = None,
) -> Page[TranscriptSegmentRead]:
    """Page recognised segments ordered by start time.

    ``start_ms``/``end_ms`` select every segment that *overlaps* the window, so a
    client can ask for the speech of a candidate clip without knowing exact
    boundaries.
    """
    transcript = transcript_service.get_transcript(session, video_id)
    segments, total = transcript_service.list_segments(
        session,
        transcript.id,
        offset=offset,
        limit=limit,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return Page[TranscriptSegmentRead](
        items=[segment_to_read(segment) for segment in segments],
        total=total,
        limit=limit,
        offset=offset,
    )


def _ensure_transcribable(video: Video) -> None:
    """Reject videos that cannot be transcribed, with the canonical errors."""
    if video.status != VideoStatus.READY.value:
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Video '{video.id}' is '{video.status}'; only a ready video can be transcribed",
            status_code=409,
            details={"video_id": video.id, "status": video.status},
        )
    if not video.has_audio:
        raise AppError(
            ErrorCode.NO_AUDIO_STREAM,
            f"Video '{video.id}' has no audio stream to transcribe",
            status_code=422,
            details={"video_id": video.id},
        )
    if not video.duration_ms:
        raise AppError(
            ErrorCode.PROBE_FAILED,
            f"Video '{video.id}' has no known duration; probe it before transcribing",
            status_code=422,
            details={"video_id": video.id},
        )


@router.post(
    "/{video_id}/transcribe",
    response_model=TranscriptRead,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def transcribe_video(
    video_id: str,
    session: DbSessionDep,
    settings: SettingsDep,
    payload: TranscribeRequest | None = None,
) -> TranscriptRead:
    """Queue a transcription job for a video.

    * a completed transcript is left alone unless ``force`` is set (then it is
      purged and re-run from scratch);
    * an interrupted transcript (``pending``/``failed``/``canceled``/``running``
      without a live job) is resumed chunk by chunk;
    * ``force`` purges any existing transcript (completed or interrupted) and
      re-runs it from scratch;
    * a video whose transcription job is still queued or running is refused, so
      one video can never be transcribed twice at the same time.
    """
    force = bool(payload and payload.force)
    # The "is there an active job / which transcript exists" sequence is a
    # check-then-act sequence: take SQLite's write lock first so two concurrent
    # ``POST``s cannot both pass the checks before either commits.
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    video = video_service.get_video(session, video_id)
    _ensure_transcribable(video)

    active = transcript_service.find_active_asr_job(session, video_id)
    if active is not None:
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Video '{video_id}' already has an active transcription job",
            status_code=409,
            details={"video_id": video_id, "job_id": active.id, "status": active.status},
        )

    transcript = transcript_service.find_transcript(session, video_id)
    if transcript is not None:
        if transcript.status == TranscriptStatus.COMPLETED.value and not force:
            raise AppError(
                ErrorCode.TRANSCRIPT_CONFLICT,
                f"Video '{video_id}' is already transcribed; pass force=true to run it again",
                status_code=409,
                details={"video_id": video_id, "transcript_id": transcript.id},
            )
        if force:
            # ``force`` means "start over": purge chunks and segments of an
            # interrupted transcript too, not only of a completed one.
            transcript_service.reset_transcript(session, transcript.id, purge=True)

    if transcript is None:
        transcript = _create_transcript(session, settings, video)

    job = job_service.enqueue(
        session,
        type=transcript_service.ASR_JOB_TYPE,
        video_id=video_id,
        params={"language": settings.asr_language, "model": settings.asr_model},
    )
    try:
        session.commit()
    except IntegrityError:
        # The checks above are not atomic: a concurrent request can win the
        # one-transcript-per-video unique index between the SELECT and the
        # INSERT. Answer with the documented conflict instead of a 500.
        session.rollback()
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Another transcription of video '{video_id}' was queued concurrently",
            status_code=409,
            details={"video_id": video_id},
        ) from None
    return _read(session, transcript, job_id=job.id)


def _create_transcript(session: Session, settings: Settings, video: Video) -> Transcript:
    """Create the pending transcript row of a video that never had one."""
    return transcript_service.get_or_create_transcript(
        session,
        video.id,
        language=settings.asr_language,
        model_name=settings.asr_model,
        duration_ms=int(video.duration_ms or 0),
        chunk_seconds=settings.asr_chunk_seconds,
        options=transcription_options(settings),
    )
