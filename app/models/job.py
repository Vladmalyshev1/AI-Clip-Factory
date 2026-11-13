"""ORM models for jobs, their stages and their event log.

All timestamps are integer epoch milliseconds (UTC). Media timestamps will use
the same convention once media models arrive.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.status import JobStatus, StageStatus
from app.models.base import Base


def new_id() -> str:
    """Return a fresh UUID4 string used as primary key."""
    return str(uuid.uuid4())


class Job(Base):
    """One unit of pipeline work."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_priority_created", "status", text("priority DESC"), "created_at"),
        Index("ix_jobs_video_id", "video_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(Text, nullable=False, default="pipeline")
    video_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=JobStatus.QUEUED.value
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress: Mapped[float] = mapped_column(nullable=False, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    heartbeat_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class JobStage(Base):
    """Per-stage progress and checkpoint of a job."""

    __tablename__ = "job_stages"
    __table_args__ = (UniqueConstraint("job_id", "name", name="uq_job_stages_job_id_name"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default=StageStatus.PENDING.value
    )
    progress: Mapped[float] = mapped_column(nullable=False, default=0.0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)


class JobEvent(Base):
    """Append-only, cheap-to-poll event log for a job."""

    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_id_id", "job_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    ts: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    code: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
