"""Highlight analysis persistence: one analysis per video and its candidates.

Mirrors :mod:`app.services.transcript_service`: every helper performs one short
unit of work and flushes instead of committing, so the caller (worker stage or
API route) owns the transaction boundary. Candidates are replaced atomically,
which makes re-running an analysis idempotent: no run can ever duplicate or
leave a half-written candidate list behind.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode
from app.core.status import (
    HighlightStatus,
    can_transition_highlight,
    parse_highlight_status,
)
from app.models.highlight import Highlight, HighlightCandidate
from app.models.job import Job
from app.models.video import Video
from app.services.job_service import dump_json
from app.services.transcript_service import ACTIVE_JOB_STATUSES, MAX_ERROR_CHARS

logger = logging.getLogger(__name__)

#: ``Job.type`` of a job that detects highlights in a video.
HIGHLIGHT_JOB_TYPE: Final = "highlight"
#: Default number of candidates returned by :func:`list_candidates`.
DEFAULT_CANDIDATE_LIMIT: Final = 50


class CandidateLike(Protocol):
    """Structural type of one detected moment (``app.ai.highlights``)."""

    @property
    def start_ms(self) -> int: ...

    @property
    def end_ms(self) -> int: ...

    @property
    def score(self) -> float: ...

    @property
    def reason(self) -> str: ...

    @property
    def transcript_text(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HighlightCandidateStats:
    """Aggregated candidate counters of one analysis."""

    total: int
    best_score: float | None


def _now(now_ms: int | None) -> int:
    """Return ``now_ms`` or the current epoch milliseconds."""
    return now_ms if now_ms is not None else utc_now_ms()


def _truncate(message: str) -> str:
    """Clamp an error message to :data:`MAX_ERROR_CHARS` characters."""
    return message[:MAX_ERROR_CHARS]


def _ensure_transition(
    current: HighlightStatus | str, target: HighlightStatus | str
) -> None:
    """Raise ``INVALID_STATE_TRANSITION`` (409) when the move is not allowed."""
    if not can_transition_highlight(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid highlight transition from '{parse_highlight_status(current)}' "
            f"to '{parse_highlight_status(target)}'",
            status_code=409,
        )


def find_analysis(session: Session, video_id: str) -> Highlight | None:
    """Return the highlight analysis of a video, or ``None`` when it has none."""
    return session.scalars(select(Highlight).where(Highlight.video_id == video_id)).first()


def get_analysis(session: Session, video_id: str) -> Highlight:
    """Load the analysis of a video or raise ``HIGHLIGHT_NOT_FOUND`` (404)."""
    analysis = find_analysis(session, video_id)
    if analysis is None:
        raise AppError(
            ErrorCode.HIGHLIGHT_NOT_FOUND,
            f"Video '{video_id}' has no highlight analysis",
            status_code=404,
            details={"video_id": video_id},
        )
    return analysis


def get_analysis_by_id(session: Session, highlight_id: str) -> Highlight:
    """Load an analysis by id or raise ``HIGHLIGHT_NOT_FOUND`` (404)."""
    analysis = session.get(Highlight, highlight_id)
    if analysis is None:
        raise AppError(
            ErrorCode.HIGHLIGHT_NOT_FOUND,
            f"Highlight analysis '{highlight_id}' not found",
            status_code=404,
            details={"highlight_id": highlight_id},
        )
    return analysis


def get_or_create_analysis(
    session: Session,
    video_id: str,
    *,
    options: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> Highlight:
    """Return the single analysis row of a video, creating it if needed.

    The configuration snapshot is refreshed on an existing row, but the status
    is never touched: re-running an analysis is an explicit API action.
    """
    if session.get(Video, video_id) is None:
        raise AppError(
            ErrorCode.VIDEO_NOT_FOUND,
            f"Video '{video_id}' not found",
            status_code=404,
            details={"video_id": video_id},
        )

    now = _now(now_ms)
    analysis = session.scalars(
        select(Highlight).where(Highlight.video_id == video_id)
    ).first()
    if analysis is not None:
        if options is not None:
            analysis.options_json = dump_json(options)
        analysis.updated_at = now
        return analysis

    analysis = Highlight(
        video_id=video_id,
        status=HighlightStatus.PENDING.value,
        options_json=dump_json(options) if options is not None else None,
        created_at=now,
        updated_at=now,
    )
    session.add(analysis)
    session.flush()
    return analysis


def start_analysis(
    session: Session, analysis: Highlight, *, now_ms: int | None = None
) -> Highlight:
    """Move an analysis into ``running`` and record the first start time.

    Resuming after a failure clears ``finished_at``: the analysis is running
    again, so reporting the end of the previous run would be wrong.
    """
    now = _now(now_ms)
    current = parse_highlight_status(analysis.status)
    if current is HighlightStatus.COMPLETED:
        raise AppError(
            ErrorCode.HIGHLIGHT_CONFLICT,
            f"Highlight analysis '{analysis.id}' is already completed; "
            "reset it to run it again",
            status_code=409,
            details={"highlight_id": analysis.id, "status": current.value},
        )
    if current is HighlightStatus.CANCELED:
        # A canceled analysis has to be re-armed through ``pending``.
        _ensure_transition(current, HighlightStatus.PENDING)
        current = HighlightStatus.PENDING
    _ensure_transition(current, HighlightStatus.RUNNING)

    analysis.status = HighlightStatus.RUNNING.value
    analysis.started_at = analysis.started_at or now
    analysis.finished_at = None
    analysis.error_code = None
    analysis.error_message = None
    analysis.updated_at = now
    return analysis


def pause_analysis(
    session: Session, highlight_id: str, *, now_ms: int | None = None
) -> Highlight:
    """Return an analysis to ``pending`` while keeping its candidates.

    Used for cancellation: the next run recomputes the candidates anyway, so a
    paused analysis simply becomes claimable again.
    """
    now = _now(now_ms)
    analysis = get_analysis_by_id(session, highlight_id)
    current = parse_highlight_status(analysis.status)
    if current is HighlightStatus.PENDING:
        analysis.updated_at = now
        return analysis
    if current is HighlightStatus.COMPLETED:
        raise AppError(
            ErrorCode.HIGHLIGHT_CONFLICT,
            f"Highlight analysis '{highlight_id}' is completed; reset it instead "
            "of pausing it",
            status_code=409,
            details={"highlight_id": highlight_id, "status": current.value},
        )
    _ensure_transition(current, HighlightStatus.PENDING)

    analysis.status = HighlightStatus.PENDING.value
    analysis.updated_at = now
    return analysis


def fail_analysis(
    session: Session,
    highlight_id: str,
    *,
    error_code: str,
    error_message: str,
    now_ms: int | None = None,
) -> Highlight:
    """Mark an analysis as failed, keeping the (truncated) reason for the UI."""
    now = _now(now_ms)
    analysis = get_analysis_by_id(session, highlight_id)
    current = parse_highlight_status(analysis.status)
    if current is HighlightStatus.COMPLETED:
        raise AppError(
            ErrorCode.HIGHLIGHT_CONFLICT,
            f"Highlight analysis '{highlight_id}' is already completed",
            status_code=409,
            details={"highlight_id": highlight_id, "status": analysis.status},
        )
    _ensure_transition(current, HighlightStatus.FAILED)
    analysis.status = HighlightStatus.FAILED.value
    analysis.error_code = error_code
    analysis.error_message = _truncate(error_message)
    analysis.finished_at = now
    analysis.updated_at = now
    return analysis


def complete_analysis(
    session: Session, highlight_id: str, *, now_ms: int | None = None
) -> Highlight:
    """Mark an analysis completed (idempotent for an already completed one)."""
    now = _now(now_ms)
    analysis = get_analysis_by_id(session, highlight_id)
    if parse_highlight_status(analysis.status) is HighlightStatus.COMPLETED:
        analysis.updated_at = now
        return analysis
    _ensure_transition(analysis.status, HighlightStatus.COMPLETED)
    analysis.status = HighlightStatus.COMPLETED.value
    analysis.finished_at = now
    analysis.error_code = None
    analysis.error_message = None
    analysis.updated_at = now
    return analysis


def reset_analysis(
    session: Session, highlight_id: str, *, purge: bool, now_ms: int | None = None
) -> Highlight:
    """Put an analysis back to ``pending``; ``purge`` also drops its candidates."""
    now = _now(now_ms)
    analysis = get_analysis_by_id(session, highlight_id)
    if purge:
        session.execute(
            delete(HighlightCandidate)
            .where(HighlightCandidate.highlight_id == highlight_id)
            .execution_options(synchronize_session="fetch")
        )
    analysis.finished_at = None
    analysis.status = HighlightStatus.PENDING.value
    analysis.error_code = None
    analysis.error_message = None
    analysis.updated_at = now
    session.flush()
    return analysis


def replace_candidates(
    session: Session,
    highlight_id: str,
    video_id: str,
    candidates: Sequence[CandidateLike],
    *,
    now_ms: int | None = None,
) -> int:
    """Replace every stored candidate of an analysis with ``candidates``.

    Existing rows are deleted first and flushed, so re-running an analysis never
    duplicates candidates and SQLite may reuse the deleted space safely.

    Returns:
        The number of stored candidate rows.
    """
    now = _now(now_ms)
    session.execute(
        delete(HighlightCandidate)
        .where(HighlightCandidate.highlight_id == highlight_id)
        .execution_options(synchronize_session="fetch")
    )
    session.flush()

    rows: list[HighlightCandidate] = []
    for candidate in candidates:
        start_ms = max(0, int(candidate.start_ms))
        end_ms = max(start_ms, int(candidate.end_ms))
        metadata = candidate.metadata or {}
        rows.append(
            HighlightCandidate(
                highlight_id=highlight_id,
                video_id=video_id,
                start_ms=start_ms,
                end_ms=end_ms,
                score=max(0.0, min(1.0, float(candidate.score))),
                reason=candidate.reason or "",
                transcript_text=candidate.transcript_text or "",
                metadata_json=dump_json(metadata) if metadata else None,
                created_at=now,
            )
        )
    if rows:
        session.add_all(rows)

    analysis = session.get(Highlight, highlight_id)
    if analysis is not None:
        analysis.updated_at = now
    session.flush()
    return len(rows)


def list_candidates(
    session: Session,
    highlight_id: str,
    *,
    offset: int = 0,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> tuple[list[HighlightCandidate], int]:
    """Return a page of candidates ordered by score (best first)."""
    total = session.scalar(
        select(func.count())
        .select_from(HighlightCandidate)
        .where(HighlightCandidate.highlight_id == highlight_id)
    )
    items = list(
        session.scalars(
            select(HighlightCandidate)
            .where(HighlightCandidate.highlight_id == highlight_id)
            .order_by(HighlightCandidate.score.desc(), HighlightCandidate.start_ms)
            .limit(max(1, limit))
            .offset(max(0, offset))
        ).all()
    )
    return items, int(total or 0)


def get_candidate(session: Session, video_id: str, candidate_id: str) -> HighlightCandidate:
    """Load one candidate of a video or raise ``HIGHLIGHT_NOT_FOUND`` (404)."""
    candidate = session.scalars(
        select(HighlightCandidate).where(
            HighlightCandidate.id == candidate_id,
            HighlightCandidate.video_id == video_id,
        )
    ).first()
    if candidate is None:
        raise AppError(
            ErrorCode.HIGHLIGHT_NOT_FOUND,
            f"Video '{video_id}' has no highlight candidate '{candidate_id}'",
            status_code=404,
            details={"video_id": video_id, "candidate_id": candidate_id},
        )
    return candidate


def candidate_stats(session: Session, highlight_id: str) -> HighlightCandidateStats:
    """Return the candidate count and best score of an analysis."""
    total, best = session.execute(
        select(
            func.count(HighlightCandidate.id),
            func.max(HighlightCandidate.score),
        ).where(HighlightCandidate.highlight_id == highlight_id)
    ).one()
    return HighlightCandidateStats(
        total=int(total or 0),
        best_score=float(best) if best is not None else None,
    )


def find_active_highlight_job(session: Session, video_id: str) -> Job | None:
    """Return the newest live highlight job of a video, if any."""
    return session.scalars(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.type == HIGHLIGHT_JOB_TYPE,
            Job.status.in_(tuple(status.value for status in ACTIVE_JOB_STATUSES)),
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    ).first()


def latest_highlight_job(session: Session, video_id: str) -> Job | None:
    """Return the newest highlight job of a video in any state."""
    return session.scalars(
        select(Job)
        .where(Job.video_id == video_id, Job.type == HIGHLIGHT_JOB_TYPE)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(1)
    ).first()


__all__ = [
    "ACTIVE_JOB_STATUSES",
    "DEFAULT_CANDIDATE_LIMIT",
    "HIGHLIGHT_JOB_TYPE",
    "CandidateLike",
    "HighlightCandidateStats",
    "candidate_stats",
    "complete_analysis",
    "fail_analysis",
    "find_active_highlight_job",
    "find_analysis",
    "get_analysis",
    "get_analysis_by_id",
    "get_candidate",
    "get_or_create_analysis",
    "latest_highlight_job",
    "list_candidates",
    "pause_analysis",
    "replace_candidates",
    "reset_analysis",
    "start_analysis",
]
