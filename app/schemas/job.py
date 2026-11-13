"""Request/response schemas for jobs and their stages/events."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.models.job import Job, JobEvent, JobStage


class JobCreate(BaseModel):
    """Body of ``POST /jobs``."""

    type: str = "pipeline"
    video_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int | None = Field(default=None, ge=1, le=10)


class JobRetryRequest(BaseModel):
    """Body of ``POST /jobs/{id}/retry``."""

    from_stage: str | None = None


class JobErrorRead(BaseModel):
    """Persisted job/stage error."""

    code: str
    message: str


class JobStageRead(BaseModel):
    """One stage of a job."""

    id: str
    name: str
    ordinal: int
    status: str
    progress: float
    attempt: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    error: JobErrorRead | None = None


class JobEventRead(BaseModel):
    """One event log row."""

    id: int
    job_id: str
    ts_ms: int
    level: str
    stage: str | None = None
    code: str | None = None
    message: str
    payload: dict[str, Any] | None = None


class JobEventList(BaseModel):
    """Envelope for ``GET /jobs/{id}/events``."""

    items: list[JobEventRead] = Field(default_factory=list)


class JobRead(BaseModel):
    """Full job representation returned by the API."""
    id: str
    type: str
    video_id: str | None = None
    status: str
    progress: float
    current_stage: str | None = None
    attempt: int
    max_attempts: int
    params: dict[str, Any] = Field(default_factory=dict)
    error: JobErrorRead | None = None
    created_at_ms: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    updated_at_ms: int
    stages: list[JobStageRead] = Field(default_factory=list)


def load_json_object(value: str | None) -> dict[str, Any]:
    """Parse a JSON object column, tolerating NULL and malformed values."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stage_to_read(stage: JobStage) -> JobStageRead:
    """Convert a stage row into its API representation."""
    error = None
    if stage.error_code or stage.error_message:
        error = JobErrorRead(
            code=stage.error_code or "STAGE_FAILED",
            message=stage.error_message or "",
        )
    return JobStageRead(
        id=stage.id,
        name=stage.name,
        ordinal=stage.ordinal,
        status=stage.status,
        progress=stage.progress,
        attempt=stage.attempt,
        started_at_ms=stage.started_at,
        finished_at_ms=stage.finished_at,
        error=error,
    )


def event_to_read(event: JobEvent) -> JobEventRead:
    """Convert an event row into its API representation."""
    payload = load_json_object(event.payload_json) or None
    return JobEventRead(
        id=event.id,
        job_id=event.job_id,
        ts_ms=event.ts,
        level=event.level,
        stage=event.stage,
        code=event.code,
        message=event.message,
        payload=payload,
    )


def job_to_read(job: Job, stages: Sequence[JobStage] = ()) -> JobRead:
    """Convert a job row (plus stage rows) into its API representation."""
    error = None
    if job.error_code or job.error_message:
        error = JobErrorRead(
            code=job.error_code or "INTERNAL_ERROR",
            message=job.error_message or "",
        )
    return JobRead(
        id=job.id,
        type=job.type,
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        params=load_json_object(job.params_json),
        error=error,
        created_at_ms=job.created_at,
        started_at_ms=job.started_at,
        finished_at_ms=job.finished_at,
        updated_at_ms=job.updated_at,
        stages=[stage_to_read(stage) for stage in stages],
    )
