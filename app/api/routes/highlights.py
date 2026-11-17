"""Highlight endpoints: read an analysis, page candidates, start a run.

Routes stay thin: they validate the state of the video and its transcript,
delegate to :mod:`app.services.highlight_service` and
:mod:`app.services.job_service`, and convert rows into API schemas. The queue
owns the work, so ``POST`` answers ``202`` with the analysis and the new
``job_id`` instead of blocking.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi import status as http_status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.highlights import HighlightConfig
from app.api.deps import DbSessionDep, SettingsDep
from app.core.errors import AppError, ErrorCode
from app.core.status import HighlightStatus, TranscriptStatus
from app.models.highlight import Highlight
from app.schemas.common import Page
from app.schemas.highlight import (
    AnalyzeHighlightsRequest,
    HighlightCandidateRead,
    HighlightRead,
    candidate_to_read,
    highlight_to_read,
)
from app.services import highlight_service, job_service, transcript_service, video_service

router = APIRouter(prefix="/videos", tags=["highlights"])

LimitQuery = Annotated[int, Query(ge=1, le=500)]
OffsetQuery = Annotated[int, Query(ge=0)]


def _read(
    session: Session, analysis: Highlight, *, job_id: str | None = None
) -> HighlightRead:
    """Serialise an analysis with its candidate count and best score."""
    stats = highlight_service.candidate_stats(session, analysis.id)
    return highlight_to_read(
        analysis,
        candidate_count=stats.total,
        best_score=stats.best_score,
        job_id=job_id,
    )


@router.get("/{video_id}/highlights", response_model=HighlightRead)
def get_highlight_analysis(video_id: str, session: DbSessionDep) -> HighlightRead:
    """Return the highlight analysis of a video (404 until it was started once)."""
    analysis = highlight_service.get_analysis(session, video_id)
    job = highlight_service.latest_highlight_job(session, video_id)
    return _read(session, analysis, job_id=job.id if job is not None else None)


@router.get(
    "/{video_id}/highlights/candidates",
    response_model=Page[HighlightCandidateRead],
)
def list_highlight_candidates(
    video_id: str,
    session: DbSessionDep,
    limit: LimitQuery = highlight_service.DEFAULT_CANDIDATE_LIMIT,
    offset: OffsetQuery = 0,
) -> Page[HighlightCandidateRead]:
    """Page the highlight candidates of a video, best score first."""
    analysis = highlight_service.get_analysis(session, video_id)
    items, total = highlight_service.list_candidates(
        session, analysis.id, offset=offset, limit=limit
    )
    return Page[HighlightCandidateRead](
        items=[candidate_to_read(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{video_id}/highlights/candidates/{candidate_id}",
    response_model=HighlightCandidateRead,
)
def get_highlight_candidate(
    video_id: str, candidate_id: str, session: DbSessionDep
) -> HighlightCandidateRead:
    """Return one highlight candidate of a video."""
    return candidate_to_read(
        highlight_service.get_candidate(session, video_id, candidate_id)
    )


@router.post(
    "/{video_id}/detect-highlights",
    response_model=HighlightRead,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def detect_highlights(
    video_id: str,
    session: DbSessionDep,
    settings: SettingsDep,
    payload: AnalyzeHighlightsRequest | None = None,
) -> HighlightRead:
    """Queue a highlight detection job for a video.

    * the video must have a *completed* transcript;
    * a completed analysis is left alone unless ``force`` is set (then the
      candidates are purged and the analysis re-runs);
    * a video whose detection job is still queued or running is refused, so one
      video can never be analysed twice at the same time.
    """
    force = bool(payload and payload.force)
    # The "is there an active job / which analysis exists" sequence is a
    # check-then-act sequence: take SQLite's write lock first so two concurrent
    # ``POST``s cannot both pass the checks before either commits.
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    video_service.get_video(session, video_id)

    transcript = transcript_service.find_transcript(session, video_id)
    if transcript is None:
        raise AppError(
            ErrorCode.TRANSCRIPT_NOT_FOUND,
            f"Video '{video_id}' has no transcript to analyse",
            status_code=404,
            details={"video_id": video_id},
        )
    if transcript.status != TranscriptStatus.COMPLETED.value:
        raise AppError(
            ErrorCode.TRANSCRIPT_INCOMPLETE,
            f"Video '{video_id}' has no completed transcript "
            f"(status: {transcript.status})",
            status_code=409,
            details={"video_id": video_id, "transcript_status": transcript.status},
        )

    active = highlight_service.find_active_highlight_job(session, video_id)
    if active is not None:
        raise AppError(
            ErrorCode.HIGHLIGHT_CONFLICT,
            f"Video '{video_id}' already has an active highlight job",
            status_code=409,
            details={"video_id": video_id, "job_id": active.id, "status": active.status},
        )

    analysis = highlight_service.find_analysis(session, video_id)
    if analysis is not None:
        if analysis.status == HighlightStatus.COMPLETED.value and not force:
            raise AppError(
                ErrorCode.HIGHLIGHT_CONFLICT,
                f"Video '{video_id}' is already analysed; pass force=true to "
                "run it again",
                status_code=409,
                details={"video_id": video_id, "highlight_id": analysis.id},
            )
        if force:
            # ``force`` means "start over": purge the candidates of an
            # interrupted analysis too, not only of a completed one.
            highlight_service.reset_analysis(session, analysis.id, purge=True)

    if analysis is None:
        analysis = highlight_service.get_or_create_analysis(
            session, video_id, options=asdict(HighlightConfig.from_settings(settings))
        )

    job = job_service.enqueue(
        session,
        type=highlight_service.HIGHLIGHT_JOB_TYPE,
        video_id=video_id,
        params={"force": force},
    )
    try:
        session.commit()
    except IntegrityError:
        # The checks above are not atomic: a concurrent request can win the
        # one-analysis-per-video unique index between the SELECT and the
        # INSERT. Answer with the documented conflict instead of a 500.
        session.rollback()
        raise AppError(
            ErrorCode.HIGHLIGHT_CONFLICT,
            f"Another highlight analysis of video '{video_id}' was queued "
            "concurrently",
            status_code=409,
            details={"video_id": video_id},
        ) from None
    return _read(session, analysis, job_id=job.id)
