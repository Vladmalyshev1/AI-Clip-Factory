"""ORM models for speech-to-text transcripts, their chunks and their segments.

A transcript is the resumable speech-recognition result of one source
recording: the audio is cut into fixed-length chunks, every chunk tracks its own
status and attempt counter, and the recognised segments are stored with
absolute media timestamps. All timestamps are integer epoch milliseconds.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import TranscriptChunkStatus, TranscriptStatus
from app.models.base import Base
from app.models.job import new_id

#: Default recognition language of a transcript row.
DEFAULT_LANGUAGE = "en"


class Transcript(Base):
    """Speech-to-text result of one video (at most one row per video)."""

    __tablename__ = "transcripts"
    __table_args__ = (Index("uq_transcripts_video_id", "video_id", unique=True),)

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    language: Mapped[str] = mapped_column(Text, nullable=False, default=DEFAULT_LANGUAGE)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=TranscriptStatus.PENDING.value
    )
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class TranscriptChunk(Base):
    """One resumable audio window of a transcript (append-only status log)."""

    __tablename__ = "transcript_chunks"

    transcript_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("transcripts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=TranscriptChunkStatus.PENDING.value
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class TranscriptSegment(Base):
    """One recognised speech segment with absolute media timestamps."""

    __tablename__ = "transcript_segments"
    __table_args__ = (
        Index("ix_transcript_segments_transcript_start", "transcript_id", "start_ms"),
        Index("ix_transcript_segments_transcript_chunk", "transcript_id", "chunk_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcript_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("transcripts.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_speech_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = [
    "DEFAULT_LANGUAGE",
    "Transcript",
    "TranscriptChunk",
    "TranscriptSegment",
]
