"""ORM models for highlight analyses and their detected candidates (M4).

A highlight analysis is the rule-based detection result of one video: it reads
the completed transcript produced by M3, and every detected interesting moment
is stored as one candidate row with absolute media timestamps and a score in
``[0, 1]``. All timestamps are integer epoch milliseconds.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import HighlightStatus
from app.models.base import Base
from app.models.job import new_id


class Highlight(Base):
    """Highlight detection result of one video (at most one row per video)."""

    __tablename__ = "highlights"
    __table_args__ = (Index("uq_highlights_video_id", "video_id", unique=True),)

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=HighlightStatus.PENDING.value
    )
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class HighlightCandidate(Base):
    """One scored interesting moment of a video."""

    __tablename__ = "highlight_candidates"
    __table_args__ = (
        Index("ix_highlight_candidates_highlight_score", "highlight_id", "score"),
        Index("ix_highlight_candidates_video_id", "video_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    highlight_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("highlights.id", ondelete="CASCADE"),
        nullable=False,
    )
    video_id: Mapped[str] = mapped_column(Text, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = ["Highlight", "HighlightCandidate"]
