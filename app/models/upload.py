"""ORM model for resumable chunked upload sessions (``uploads`` table).

``received_bytes`` is the only source of truth for the resume offset: the
preallocated ``.part`` file is sparse, so its size says nothing about progress.
``video_id`` is deliberately not a foreign key, because the video row is created
only when the upload completes.
"""

from __future__ import annotations

from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import UploadStatus
from app.models.base import Base
from app.models.job import new_id


class Upload(Base):
    """One resumable upload session and its partial file."""

    __tablename__ = "uploads"
    __table_args__ = (Index("ix_uploads_status_expires", "status", "expires_at"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    received_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=UploadStatus.UPLOADING.value
    )
    temp_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = ["Upload"]
