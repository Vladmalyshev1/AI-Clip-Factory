"""ORM model for immutable source recordings (``videos`` table).

A video row is the ingest result of one recording: the file itself is never
modified after it is stored in ``media/originals`` (or referenced in place), and
every media timestamp is an integer count of milliseconds.
"""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import SourceKind, VideoStatus
from app.models.base import Base
from app.models.job import new_id


class Video(Base):
    """One ingested source recording and its probed metadata."""

    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_status_created", "status", "created_at"),
        Index("ix_videos_sha256", "sha256"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_kind: Mapped[str] = mapped_column(
        Text, nullable=False, default=SourceKind.UPLOAD.value
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(Text, nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_codec: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_audio: Mapped[int | None] = mapped_column(Integer, nullable=True)
    container: Mapped[str | None] = mapped_column(Text, nullable=True)
    bitrate: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=VideoStatus.PENDING.value
    )
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = ["Video"]
