"""Transcript persistence: chunked, resumable speech-to-text results.

A two to five hour recording is transcribed chunk by chunk. Every helper below
performs exactly one short unit of work and flushes instead of committing, so
the caller (worker or API route) owns the transaction boundary. Chunk results
are replaced atomically, which makes re-running a chunk idempotent: a crash
between two chunks never duplicates segments and never loses finished work.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode
from app.core.status import (
    UNFINISHED_CHUNK_STATUSES,
    JobStatus,
    TranscriptChunkStatus,
    TranscriptStatus,
    can_transition_chunk,
    can_transition_transcript,
    parse_chunk_status,
    parse_transcript_status,
)
from app.models.job import Job
from app.models.transcript import DEFAULT_LANGUAGE, Transcript, TranscriptChunk, TranscriptSegment
from app.models.video import Video
from app.services.job_service import clamp_progress, dump_json

logger = logging.getLogger(__name__)

#: ``Job.type`` of a job that transcribes a video.
ASR_JOB_TYPE: Final = "transcribe"
#: Longest persisted error message (keeps the UI and the DB small).
MAX_ERROR_CHARS: Final = 2000
#: Default number of segments returned by :func:`list_segments`.
DEFAULT_SEGMENT_LIMIT: Final = 100
#: Slack allowed when checking that the chunks of a transcript cover its audio.
COVERAGE_TOLERANCE_MS: Final = 1000
#: Job states that mean "this ASR job is still owned by someone".
ACTIVE_JOB_STATUSES: Final[tuple[JobStatus, ...]] = (
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.CANCEL_REQUESTED,
)


class ChunkPlanLike(Protocol):
    """Structural type of one planned audio window (owned by the media layer).

    The windows are read-only for the service, so the protocol declares
    properties: a frozen dataclass (``audio_chunks.ChunkPlan``) satisfies it.
    """

    @property
    def index(self) -> int: ...

    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...


class SegmentLike(Protocol):
    """Structural type of one recognised segment (``app.ai.asr.AsrSegment``)."""

    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def avg_logprob(self) -> float | None: ...

    @property
    def no_speech_prob(self) -> float | None: ...


@dataclass(frozen=True)
class TranscriptProgress:
    """Aggregated progress of a transcript over its chunk rows."""

    chunks_total: int
    chunks_completed: int
    chunks_running: int
    chunks_failed: int
    audio_ms_total: int
    audio_ms_processed: int
    progress: float
    current_chunk_index: int | None


def _now(now_ms: int | None) -> int:
    """Return ``now_ms`` or the current epoch milliseconds."""
    return now_ms if now_ms is not None else utc_now_ms()


def _truncate(message: str) -> str:
    """Clamp an error message to :data:`MAX_ERROR_CHARS` characters."""
    return message[:MAX_ERROR_CHARS]


def _ensure_transcript_transition(
    current: TranscriptStatus | str, target: TranscriptStatus | str
) -> None:
    """Raise ``INVALID_STATE_TRANSITION`` (409) when the move is not allowed."""
    if not can_transition_transcript(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid transcript transition from '{parse_transcript_status(current)}' "
            f"to '{parse_transcript_status(target)}'",
            status_code=409,
        )


def _ensure_chunk_transition(
    current: TranscriptChunkStatus | str, target: TranscriptChunkStatus | str
) -> None:
    """Raise ``INVALID_STATE_TRANSITION`` (409) when the chunk move is not allowed."""
    if not can_transition_chunk(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid transcript chunk transition from '{parse_chunk_status(current)}' "
            f"to '{parse_chunk_status(target)}'",
            status_code=409,
        )


def find_transcript(session: Session, video_id: str) -> Transcript | None:
    """Return the transcript of a video, or ``None`` when it has none yet."""
    return session.scalars(select(Transcript).where(Transcript.video_id == video_id)).first()


def get_transcript(session: Session, video_id: str) -> Transcript:
    """Load the transcript of a video or raise ``TRANSCRIPT_NOT_FOUND`` (404)."""
    transcript = find_transcript(session, video_id)
    if transcript is None:
        raise AppError(
            ErrorCode.TRANSCRIPT_NOT_FOUND,
            f"Video '{video_id}' has no transcript",
            status_code=404,
            details={"video_id": video_id},
        )
    return transcript


def get_transcript_by_id(session: Session, transcript_id: str) -> Transcript:
    """Load a transcript by id or raise ``TRANSCRIPT_NOT_FOUND`` (404)."""
    transcript = session.get(Transcript, transcript_id)
    if transcript is None:
        raise AppError(
            ErrorCode.TRANSCRIPT_NOT_FOUND,
            f"Transcript '{transcript_id}' not found",
            status_code=404,
            details={"transcript_id": transcript_id},
        )
    return transcript


def _get_chunk(session: Session, transcript_id: str, chunk_index: int) -> TranscriptChunk:
    """Load one chunk row or raise ``NOT_FOUND`` (404)."""
    chunk = session.get(TranscriptChunk, (transcript_id, chunk_index))
    if chunk is None:
        raise AppError(
            ErrorCode.NOT_FOUND,
            f"Transcript '{transcript_id}' has no chunk {chunk_index}",
            status_code=404,
            details={"transcript_id": transcript_id, "chunk_index": chunk_index},
        )
    return chunk


def get_or_create_transcript(
    session: Session,
    video_id: str,
    *,
    language: str,
    model_name: str,
    duration_ms: int,
    chunk_seconds: int,
    options: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> Transcript:
    """Return the single transcript row of a video, creating it if needed.

    The probe metadata and the ASR configuration are refreshed on an existing
    row, but two fields are deliberately *not* refreshed: the status (a
    ``completed`` transcript stays completed until an explicit
    :func:`reset_transcript`) and ``chunk_seconds``. The chunk length is part of
    the transcript's identity — chunk rows are keyed by ``chunk_index`` — so it
    may only change while the transcript has no chunk rows at all, which is what
    :func:`adopt_chunk_grid` enforces.
    """
    if session.get(Video, video_id) is None:
        raise AppError(
            ErrorCode.VIDEO_NOT_FOUND,
            f"Video '{video_id}' not found",
            status_code=404,
            details={"video_id": video_id},
        )

    now = _now(now_ms)
    transcript = session.scalars(
        select(Transcript).where(Transcript.video_id == video_id)
    ).first()
    if transcript is not None:
        transcript.language = language or DEFAULT_LANGUAGE
        transcript.model_name = model_name
        transcript.duration_ms = int(duration_ms)
        if options is not None:
            transcript.options_json = dump_json(options)
        transcript.updated_at = now
        return transcript

    transcript = Transcript(
        video_id=video_id,
        language=language or DEFAULT_LANGUAGE,
        model_name=model_name,
        status=TranscriptStatus.PENDING.value,
        duration_ms=int(duration_ms),
        chunk_seconds=int(chunk_seconds),
        options_json=dump_json(options) if options is not None else None,
        created_at=now,
        updated_at=now,
    )
    session.add(transcript)
    session.flush()
    return transcript


def adopt_chunk_grid(
    session: Session, transcript: Transcript, requested_seconds: int
) -> int:
    """Return the chunk length a run has to use, recording it on the transcript.

    Chunk rows are identified by ``chunk_index`` only, so the window behind an
    index is part of the transcript's identity: re-tiling a partially finished
    transcript would either duplicate speech (smaller chunks inside a finished
    one) or silently lose it (bigger chunks swallowing several unfinished ones).
    Therefore the recorded grid wins as soon as any chunk row exists, and only a
    transcript without rows (fresh, or purged by ``force=true``) adopts the
    currently configured length.
    """
    requested = max(1, int(requested_seconds))
    stored = int(transcript.chunk_seconds or 0)
    has_rows = bool(
        session.scalar(
            select(func.count())
            .select_from(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript.id)
        )
    )
    if has_rows and stored > 0:
        if stored != requested:
            logger.warning(
                "Transcript %s keeps its recorded chunk grid of %ss "
                "(ACF_ASR_CHUNK_SECONDS is %ss); re-run with force=true to re-tile it",
                transcript.id,
                stored,
                requested,
            )
        return stored
    transcript.chunk_seconds = requested
    return requested


def ensure_chunks(
    session: Session,
    transcript: Transcript,
    plans: Iterable[ChunkPlanLike],
    *,
    now_ms: int | None = None,
) -> int:
    """Create the missing chunk rows of a plan and drop stale trailing ones.

    ``completed`` rows (and their segments) are never modified, so a resumed
    run keeps every finished window. Pending or failed rows that the current
    plan no longer contains are deleted together with their segments.

    ``chunk_index`` alone identifies a row, so a completed row whose window no
    longer matches the plan is a conflict: keeping it would silently duplicate or
    skip speech. The caller must re-tile with ``force=true`` instead.

    Returns:
        The number of chunk rows that exist for the transcript afterwards.
    """
    now = _now(now_ms)
    plan_by_index = {int(plan.index): plan for plan in plans}
    existing = {
        chunk.chunk_index: chunk
        for chunk in session.scalars(
            select(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id)
        ).all()
    }

    changed = False
    for index, plan in plan_by_index.items():
        start_ms = int(plan.start_ms)
        end_ms = int(plan.end_ms)
        duration_ms = max(0, end_ms - start_ms)
        chunk = existing.get(index)
        if chunk is None:
            chunk = TranscriptChunk(
                transcript_id=transcript.id,
                chunk_index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                status=TranscriptChunkStatus.PENDING.value,
                attempts=0,
                segment_count=0,
                created_at=now,
                updated_at=now,
            )
            session.add(chunk)
            changed = True
            continue
        if chunk.status == TranscriptChunkStatus.COMPLETED.value:
            if (
                chunk.start_ms != start_ms
                or chunk.end_ms != end_ms
                or chunk.duration_ms != duration_ms
            ):
                raise AppError(
                    ErrorCode.TRANSCRIPT_CONFLICT,
                    f"Chunk {index} of transcript '{transcript.id}' was transcribed as "
                    f"[{chunk.start_ms}, {chunk.end_ms}) ms but the plan needs "
                    f"[{start_ms}, {end_ms}) ms; re-run with force=true to re-tile it",
                    status_code=409,
                    details={
                        "transcript_id": transcript.id,
                        "chunk_index": index,
                        "stored": {
                            "start_ms": chunk.start_ms,
                            "end_ms": chunk.end_ms,
                        },
                        "planned": {"start_ms": start_ms, "end_ms": end_ms},
                    },
                )
            continue
        if (
            chunk.start_ms != start_ms
            or chunk.end_ms != end_ms
            or chunk.duration_ms != duration_ms
        ):
            chunk.start_ms = start_ms
            chunk.end_ms = end_ms
            chunk.duration_ms = duration_ms
            chunk.updated_at = now
            changed = True

    removable = [
        index
        for index, chunk in existing.items()
        if index not in plan_by_index
        and chunk.status
        in (TranscriptChunkStatus.PENDING.value, TranscriptChunkStatus.FAILED.value)
    ]
    if removable:
        # ``synchronize_session="fetch"`` keeps the identity map honest: the
        # deleted rows are dropped from the session, so a later ``session.get``
        # can never resurrect a chunk that no longer exists.
        session.execute(
            delete(TranscriptSegment)
            .where(
                TranscriptSegment.transcript_id == transcript.id,
                TranscriptSegment.chunk_index.in_(removable),
            )
            .execution_options(synchronize_session="fetch")
        )
        session.execute(
            delete(TranscriptChunk)
            .where(
                TranscriptChunk.transcript_id == transcript.id,
                TranscriptChunk.chunk_index.in_(removable),
            )
            .execution_options(synchronize_session="fetch")
        )
        changed = True

    if changed:
        transcript.updated_at = now
    session.flush()
    total = session.scalar(
        select(func.count())
        .select_from(TranscriptChunk)
        .where(TranscriptChunk.transcript_id == transcript.id)
    )
    return int(total or 0)


def claimable_chunks(session: Session, transcript_id: str) -> list[TranscriptChunk]:
    """Return the unfinished chunks of a transcript in ``chunk_index`` order.

    A chunk left in ``running`` by a crashed worker is included, so a resumed
    run retries it instead of skipping it.
    """
    statuses = tuple(status.value for status in UNFINISHED_CHUNK_STATUSES)
    return list(
        session.scalars(
            select(TranscriptChunk)
            .where(
                TranscriptChunk.transcript_id == transcript_id,
                TranscriptChunk.status.in_(statuses),
            )
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )


def mark_chunk_running(
    session: Session,
    transcript_id: str,
    chunk_index: int,
    *,
    now_ms: int | None = None,
) -> TranscriptChunk:
    """Mark a chunk as claimed and count the attempt (409 when illegal)."""
    now = _now(now_ms)
    chunk = _get_chunk(session, transcript_id, chunk_index)
    _ensure_chunk_transition(chunk.status, TranscriptChunkStatus.RUNNING)
    chunk.status = TranscriptChunkStatus.RUNNING.value
    chunk.attempts += 1
    chunk.updated_at = now
    transcript = session.get(Transcript, transcript_id)
    if transcript is not None:
        transcript.updated_at = now
    return chunk


def save_chunk_result(
    session: Session,
    transcript_id: str,
    chunk_index: int,
    segments: Sequence[SegmentLike],
    *,
    processed_at: int | None = None,
    now_ms: int | None = None,
) -> int:
    """Replace every stored segment of a chunk with ``segments`` (atomic).

    Existing rows of ``(transcript_id, chunk_index)`` are deleted first, so
    re-running a chunk never duplicates segments. Empty or whitespace-only
    texts are skipped, which makes a silent window a completed chunk with zero
    segments.

    Returns:
        The number of stored segment rows.
    """
    now = _now(now_ms)
    chunk = _get_chunk(session, transcript_id, chunk_index)
    _ensure_chunk_transition(chunk.status, TranscriptChunkStatus.COMPLETED)
    # The old rows are dropped first and flushed, so SQLite may safely reuse the
    # integer ids when the replacement segments are inserted in this transaction.
    session.execute(
        delete(TranscriptSegment)
        .where(
            TranscriptSegment.transcript_id == transcript_id,
            TranscriptSegment.chunk_index == chunk_index,
        )
        .execution_options(synchronize_session="fetch")
    )
    session.flush()

    rows: list[TranscriptSegment] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        start_ms = max(0, int(segment.start_ms))
        end_ms = max(start_ms, int(segment.end_ms))
        rows.append(
            TranscriptSegment(
                transcript_id=transcript_id,
                chunk_index=chunk_index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                created_at=now,
            )
        )
    if rows:
        session.add_all(rows)

    chunk.status = TranscriptChunkStatus.COMPLETED.value
    chunk.segment_count = len(rows)
    chunk.processed_at = processed_at if processed_at is not None else now
    chunk.error_code = None
    chunk.error_message = None
    chunk.updated_at = now
    transcript = session.get(Transcript, transcript_id)
    if transcript is not None:
        transcript.updated_at = now
    session.flush()
    return len(rows)


def mark_chunk_failed(
    session: Session,
    transcript_id: str,
    chunk_index: int,
    *,
    error_code: str,
    error_message: str,
    now_ms: int | None = None,
) -> TranscriptChunk:
    """Mark a chunk as failed; it stays claimable for a later retry."""
    now = _now(now_ms)
    chunk = _get_chunk(session, transcript_id, chunk_index)
    _ensure_chunk_transition(chunk.status, TranscriptChunkStatus.FAILED)
    chunk.status = TranscriptChunkStatus.FAILED.value
    chunk.error_code = error_code
    chunk.error_message = _truncate(error_message)
    chunk.updated_at = now
    transcript = session.get(Transcript, transcript_id)
    if transcript is not None:
        transcript.updated_at = now
    return chunk


def progress(session: Session, transcript: Transcript) -> TranscriptProgress:
    """Aggregate chunk state into one UI-friendly progress snapshot."""
    rows = session.execute(
        select(
            TranscriptChunk.status,
            func.count(TranscriptChunk.chunk_index),
            func.coalesce(func.sum(TranscriptChunk.duration_ms), 0),
            func.min(TranscriptChunk.chunk_index),
        )
        .where(TranscriptChunk.transcript_id == transcript.id)
        .group_by(TranscriptChunk.status)
    ).all()

    counts: dict[str, int] = {}
    durations: dict[str, int] = {}
    lowest: dict[str, int] = {}
    for status, count, duration_sum, min_index in rows:
        counts[str(status)] = int(count or 0)
        durations[str(status)] = int(duration_sum or 0)
        if min_index is not None:
            lowest[str(status)] = int(min_index)

    completed = TranscriptChunkStatus.COMPLETED.value
    running = TranscriptChunkStatus.RUNNING.value
    failed = TranscriptChunkStatus.FAILED.value
    unfinished = {status.value for status in UNFINISHED_CHUNK_STATUSES}

    chunks_total = sum(counts.values())
    chunks_completed = counts.get(completed, 0)
    chunks_running = counts.get(running, 0)
    chunks_failed = counts.get(failed, 0)
    audio_ms_total = sum(durations.values())
    audio_ms_processed = durations.get(completed, 0)

    current_chunk_index = lowest.get(running)
    if current_chunk_index is None:
        candidates = [index for status, index in lowest.items() if status in unfinished]
        current_chunk_index = min(candidates) if candidates else None

    if audio_ms_total > 0:
        ratio = audio_ms_processed / audio_ms_total
    elif chunks_total > 0:
        ratio = chunks_completed / chunks_total
    else:
        ratio = 0.0

    return TranscriptProgress(
        chunks_total=chunks_total,
        chunks_completed=chunks_completed,
        chunks_running=chunks_running,
        chunks_failed=chunks_failed,
        audio_ms_total=audio_ms_total,
        audio_ms_processed=audio_ms_processed,
        progress=clamp_progress(ratio),
        current_chunk_index=current_chunk_index,
    )


def count_segments(session: Session, transcript_id: str) -> int:
    """Return the number of stored segments of a transcript."""
    total = session.scalar(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(TranscriptSegment.transcript_id == transcript_id)
    )
    return int(total or 0)


def list_segments(
    session: Session,
    transcript_id: str,
    *,
    offset: int = 0,
    limit: int = DEFAULT_SEGMENT_LIMIT,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> tuple[list[TranscriptSegment], int]:
    """Return a page of segments ordered by start time and the total count.

    The time filters use overlap semantics: a segment is returned when it
    overlaps ``[start_ms, end_ms)``, not only when it is fully contained.
    """
    if start_ms is not None and end_ms is not None and end_ms <= start_ms:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "end_ms must be greater than start_ms",
            status_code=422,
            details={"start_ms": start_ms, "end_ms": end_ms},
        )

    filters = [TranscriptSegment.transcript_id == transcript_id]
    if start_ms is not None:
        filters.append(TranscriptSegment.end_ms > start_ms)
    if end_ms is not None:
        filters.append(TranscriptSegment.start_ms < end_ms)

    total = session.scalar(
        select(func.count()).select_from(TranscriptSegment).where(*filters)
    )
    items = list(
        session.scalars(
            select(TranscriptSegment)
            .where(*filters)
            .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
            .limit(max(1, limit))
            .offset(max(0, offset))
        ).all()
    )
    return items, int(total or 0)


def start_transcript(
    session: Session, transcript: Transcript, *, now_ms: int | None = None
) -> Transcript:
    """Move a transcript into ``running`` and record the first start time.

    Resuming after a failure also clears ``finished_at``: the transcript is
    running again, so reporting the end of the previous run would be wrong.
    """
    now = _now(now_ms)
    current = parse_transcript_status(transcript.status)
    if current is TranscriptStatus.COMPLETED:
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Transcript '{transcript.id}' is already completed; reset it to run it again",
            status_code=409,
            details={"transcript_id": transcript.id, "status": current.value},
        )
    if current is TranscriptStatus.CANCELED:
        # A canceled transcript has to be re-armed through ``pending``.
        _ensure_transcript_transition(current, TranscriptStatus.PENDING)
        current = TranscriptStatus.PENDING
    _ensure_transcript_transition(current, TranscriptStatus.RUNNING)

    transcript.status = TranscriptStatus.RUNNING.value
    transcript.started_at = transcript.started_at or now
    transcript.finished_at = None
    transcript.error_code = None
    transcript.error_message = None
    transcript.updated_at = now
    return transcript


def pause_transcript(
    session: Session, transcript_id: str, *, now_ms: int | None = None
) -> Transcript:
    """Return a transcript to ``pending`` while keeping every finished chunk.

    Used for user cancellation and for a graceful worker shutdown: the next run
    resumes from the claimable chunks instead of starting over.
    """
    now = _now(now_ms)
    transcript = get_transcript_by_id(session, transcript_id)
    current = parse_transcript_status(transcript.status)
    if current is TranscriptStatus.PENDING:
        transcript.updated_at = now
        return transcript
    if current is TranscriptStatus.COMPLETED:
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Transcript '{transcript_id}' is completed; reset it instead of pausing it",
            status_code=409,
            details={"transcript_id": transcript_id, "status": current.value},
        )
    _ensure_transcript_transition(current, TranscriptStatus.PENDING)

    transcript.status = TranscriptStatus.PENDING.value
    transcript.updated_at = now
    return transcript


def fail_transcript(
    session: Session,
    transcript_id: str,
    *,
    error_code: str,
    error_message: str,
    now_ms: int | None = None,
) -> Transcript:
    """Mark a transcript as failed, keeping the (truncated) reason for the UI."""
    now = _now(now_ms)
    transcript = get_transcript_by_id(session, transcript_id)
    current = parse_transcript_status(transcript.status)
    if current is TranscriptStatus.COMPLETED:
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Transcript '{transcript_id}' is already completed",
            status_code=409,
            details={"transcript_id": transcript_id, "status": transcript.status},
        )
    _ensure_transcript_transition(current, TranscriptStatus.FAILED)
    transcript.status = TranscriptStatus.FAILED.value
    transcript.error_code = error_code
    transcript.error_message = _truncate(error_message)
    transcript.finished_at = now
    transcript.updated_at = now
    return transcript


def _coverage_problem(
    rows: Sequence[TranscriptChunk], expected_ms: int | None
) -> dict[str, Any] | None:
    """Describe the first hole in the chunk coverage, or return ``None``.

    Chunk rows are keyed by ``chunk_index``, so a resumed run can end up with
    rows that no longer tile the audio (for example after the chunk length was
    changed). Completing such a transcript would report success while speech was
    never transcribed, which is why this invariant is checked explicitly.
    """
    if not rows:
        return {"reason": "no_chunks"}
    first = rows[0]
    if first.start_ms != 0:
        return {
            "reason": "start",
            "chunk_index": first.chunk_index,
            "start_ms": first.start_ms,
        }
    previous = first
    for row in rows[1:]:
        if row.chunk_index != previous.chunk_index + 1 or row.start_ms != previous.end_ms:
            return {
                "reason": "gap",
                "after_chunk_index": previous.chunk_index,
                "gap_start_ms": previous.end_ms,
                "gap_end_ms": row.start_ms,
            }
        previous = row
    if expected_ms is not None and previous.end_ms < expected_ms - COVERAGE_TOLERANCE_MS:
        return {
            "reason": "tail",
            "chunk_index": previous.chunk_index,
            "covered_until_ms": previous.end_ms,
            "expected_ms": int(expected_ms),
        }
    return None


def complete_transcript(
    session: Session,
    transcript_id: str,
    *,
    expected_ms: int | None = None,
    now_ms: int | None = None,
) -> Transcript:
    """Mark a transcript completed, but only when every chunk is completed.

    ``expected_ms`` is the duration the chunk grid was planned from (the length
    of the extracted audio). When it is given the completed chunks must also
    cover its tail, so a transcript that stopped early can never be reported as
    a success.
    """
    now = _now(now_ms)
    transcript = get_transcript_by_id(session, transcript_id)
    if parse_transcript_status(transcript.status) is TranscriptStatus.COMPLETED:
        transcript.updated_at = now
        return transcript

    rows = list(
        session.scalars(
            select(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript_id)
            .order_by(TranscriptChunk.chunk_index)
        ).all()
    )
    unfinished = [
        row.chunk_index
        for row in rows
        if row.status != TranscriptChunkStatus.COMPLETED.value
    ]
    if unfinished:
        raise AppError(
            ErrorCode.ASR_FAILED,
            f"Transcript is incomplete: {len(unfinished)} of {len(rows)} chunks processed",
            status_code=500,
            details={"unfinished_chunk_indexes": [int(index) for index in unfinished]},
        )

    problem = _coverage_problem(rows, expected_ms)
    if problem is not None:
        # A chunk grid that does not cover the audio is a conflict with the
        # recorded plan, not a decoding failure; the client can re-run with
        # ``force=true`` to re-tile and re-transcribe.
        raise AppError(
            ErrorCode.TRANSCRIPT_CONFLICT,
            f"Transcript chunks do not cover the audio ({problem['reason']}); "
            "re-run with force=true",
            status_code=409,
            details={"transcript_id": transcript_id, "coverage": problem},
        )

    _ensure_transcript_transition(transcript.status, TranscriptStatus.COMPLETED)
    transcript.status = TranscriptStatus.COMPLETED.value
    transcript.finished_at = now
    transcript.error_code = None
    transcript.error_message = None
    transcript.updated_at = now
    return transcript


def reset_transcript(
    session: Session, transcript_id: str, *, purge: bool, now_ms: int | None = None
) -> Transcript:
    """Put a transcript back to ``pending``; ``purge`` also drops all results."""
    now = _now(now_ms)
    transcript = get_transcript_by_id(session, transcript_id)
    if purge:
        session.execute(
            delete(TranscriptSegment)
            .where(TranscriptSegment.transcript_id == transcript_id)
            .execution_options(synchronize_session="fetch")
        )
        session.execute(
            delete(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript_id)
            .execution_options(synchronize_session="fetch")
        )
        transcript.started_at = None
        transcript.finished_at = None
    else:
        transcript.finished_at = None

    transcript.status = TranscriptStatus.PENDING.value
    transcript.error_code = None
    transcript.error_message = None
    transcript.updated_at = now
    session.flush()
    return transcript


def find_active_asr_job(session: Session, video_id: str) -> Job | None:
    """Return the newest live transcription job of a video, if any."""
    return session.scalars(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.type == ASR_JOB_TYPE,
            Job.status.in_(tuple(status.value for status in ACTIVE_JOB_STATUSES)),
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    ).first()


def latest_asr_job(session: Session, video_id: str) -> Job | None:
    """Return the newest transcription job of a video in any state."""
    return session.scalars(
        select(Job)
        .where(Job.video_id == video_id, Job.type == ASR_JOB_TYPE)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    ).first()


__all__ = [
    "ACTIVE_JOB_STATUSES",
    "ASR_JOB_TYPE",
    "COVERAGE_TOLERANCE_MS",
    "DEFAULT_SEGMENT_LIMIT",
    "MAX_ERROR_CHARS",
    "ChunkPlanLike",
    "SegmentLike",
    "TranscriptProgress",
    "adopt_chunk_grid",
    "claimable_chunks",
    "complete_transcript",
    "count_segments",
    "ensure_chunks",
    "fail_transcript",
    "find_active_asr_job",
    "find_transcript",
    "get_or_create_transcript",
    "get_transcript",
    "get_transcript_by_id",
    "latest_asr_job",
    "list_segments",
    "mark_chunk_failed",
    "mark_chunk_running",
    "pause_transcript",
    "progress",
    "reset_transcript",
    "save_chunk_result",
    "start_transcript",
]
