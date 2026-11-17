"""ORM model for generated clips (M6).

One clip row describes exactly one rendered file and the candidate it was cut
from. The file itself is never stored in the database: ``output_path`` points at
``clips_dir`` and every media timestamp is an integer count of milliseconds.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import ClipKind, ClipStatus
from app.models.base import Base
from app.models.job import new_id


class Clip(Base):
    """One rendered clip of an ingested recording."""

    __tablename__ = "clips"
    __table_args__ = (
        Index("ix_clips_video_id", "video_id"),
        Index("ix_clips_status_created", "status", "created_at"),
        Index("ix_clips_parent_clip_id", "parent_clip_id"),
        Index("ix_clips_kind_created", "kind", "created_at"),
        # A candidate may produce at most one clip row; re-generating resets the
        # existing row instead of duplicating it.
        Index("uq_clips_candidate_id", "highlight_candidate_id", unique=True),
        CheckConstraint("start_ms >= 0", name="ck_clips_start_non_negative"),
        CheckConstraint("end_ms > start_ms", name="ck_clips_bounds_ordered"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Set when a highlight re-run deletes and recreates its candidates.
    highlight_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("highlights.id", ondelete="SET NULL"),
        nullable=True,
    )
    highlight_candidate_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("highlight_candidates.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default=ClipStatus.PENDING.value)
    #: ``auto`` is a generated clip; ``export`` is a re-render cut from one.
    kind: Mapped[str] = mapped_column(Text, nullable=False, default=ClipKind.AUTO.value)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: The clip this export was cut from (``None`` for generated clips).
    parent_clip_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("clips.id", ondelete="CASCADE"),
        nullable=True,
    )
    #: Editor draft of one clip (bounds, format, subtitles), JSON.
    edit_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    aspect_ratio: Mapped[str | None] = mapped_column(Text, nullable=True)

    encoding_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    crop_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Last job that rendered (or will render) this clip.
    job_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = ["Clip"]
