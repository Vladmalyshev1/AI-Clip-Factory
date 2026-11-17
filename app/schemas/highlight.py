"""Request/response schemas for highlight analyses and candidates (M4).

The API exposes the analysis row plus two derived aggregates: the number of
stored candidates and the best score, so a client can render "top moments"
metadata without paging the full candidate list.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.highlight import Highlight, HighlightCandidate
from app.schemas.transcript import load_options


class HighlightErrorRead(BaseModel):
    """Persisted detection error of a highlight analysis."""

    code: str
    message: str


class HighlightRead(BaseModel):
    """Full highlight analysis representation returned by the API."""

    id: str
    video_id: str
    status: str
    options: dict[str, Any] | None = None
    error: HighlightErrorRead | None = None
    candidate_count: int
    best_score: float | None = None
    job_id: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    created_at_ms: int
    updated_at_ms: int


class HighlightCandidateRead(BaseModel):
    """One scored interesting moment with absolute media timestamps."""

    id: str
    video_id: str
    highlight_id: str
    start_ms: int
    end_ms: int
    duration_ms: int
    score: float
    reason: str
    transcript: str
    metadata: dict[str, Any] | None = None


class AnalyzeHighlightsRequest(BaseModel):
    """Body of ``POST /videos/{id}/detect-highlights``.

    ``force`` throws away a finished analysis and runs it again; without it a
    completed analysis is left alone (409).
    """

    force: bool = Field(default=False)


def candidate_to_read(candidate: HighlightCandidate) -> HighlightCandidateRead:
    """Convert a candidate row into its API representation."""
    return HighlightCandidateRead(
        id=candidate.id,
        video_id=candidate.video_id,
        highlight_id=candidate.highlight_id,
        start_ms=candidate.start_ms,
        end_ms=candidate.end_ms,
        duration_ms=candidate.end_ms - candidate.start_ms,
        score=candidate.score,
        reason=candidate.reason,
        transcript=candidate.transcript_text,
        metadata=load_options(candidate.metadata_json),
    )


def highlight_to_read(
    highlight: Highlight,
    *,
    candidate_count: int,
    best_score: float | None = None,
    job_id: str | None = None,
) -> HighlightRead:
    """Convert an analysis row into its API representation."""
    error = None
    if highlight.error_code or highlight.error_message:
        error = HighlightErrorRead(
            code=highlight.error_code or "INTERNAL_ERROR",
            message=highlight.error_message or "",
        )
    return HighlightRead(
        id=highlight.id,
        video_id=highlight.video_id,
        status=highlight.status,
        options=load_options(highlight.options_json),
        error=error,
        candidate_count=candidate_count,
        best_score=best_score,
        job_id=job_id,
        started_at_ms=highlight.started_at,
        finished_at_ms=highlight.finished_at,
        created_at_ms=highlight.created_at,
        updated_at_ms=highlight.updated_at,
    )


__all__ = [
    "AnalyzeHighlightsRequest",
    "HighlightCandidateRead",
    "HighlightErrorRead",
    "HighlightRead",
    "candidate_to_read",
    "highlight_to_read",
]
