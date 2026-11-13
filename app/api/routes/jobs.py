"""Job endpoints: create, list, inspect, cancel, retry, events."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi import status as http_status

from app.api.deps import DbSessionDep, SettingsDep
from app.core.status import JobStatus
from app.schemas.common import Page
from app.schemas.job import (
    JobCreate,
    JobEventList,
    JobRead,
    JobRetryRequest,
    event_to_read,
    job_to_read,
)
from app.services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])

LimitQuery = Annotated[int, Query(ge=1, le=200)]
OffsetQuery = Annotated[int, Query(ge=0)]


@router.post("", response_model=JobRead, status_code=http_status.HTTP_202_ACCEPTED)
def create_job(
    payload: JobCreate, session: DbSessionDep, settings: SettingsDep
) -> JobRead:
    """Enqueue a new job; it stays ``queued`` until a worker claims it."""
    max_attempts = (
        payload.max_attempts if payload.max_attempts is not None else settings.job_max_attempts
    )
    job = job_service.enqueue(
        session,
        type=payload.type,
        params=payload.params,
        video_id=payload.video_id,
        max_attempts=max_attempts,
    )
    session.commit()
    return job_to_read(job)


@router.get("", response_model=Page[JobRead])
def list_jobs(
    session: DbSessionDep,
    video_id: str | None = None,
    status: JobStatus | None = None,
    limit: LimitQuery = 50,
    offset: OffsetQuery = 0,
) -> Page[JobRead]:
    """List jobs newest first with optional filters."""
    jobs, total = job_service.list_jobs(
        session,
        video_id=video_id,
        status=status.value if status is not None else None,
        limit=limit,
        offset=offset,
    )
    items = [job_to_read(job, job_service.list_job_stages(session, job.id)) for job in jobs]
    return Page[JobRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: str, session: DbSessionDep) -> JobRead:
    """Return one job including its stages ordered by ordinal."""
    job = job_service.get_job(session, job_id)
    stages = job_service.list_job_stages(session, job_id)
    return job_to_read(job, stages)


@router.post(
    "/{job_id}/cancel", response_model=JobRead, status_code=http_status.HTTP_202_ACCEPTED
)
def cancel_job(job_id: str, session: DbSessionDep) -> JobRead:
    """Cancel a queued job immediately or request cancellation of a running one."""
    job = job_service.request_cancel(session, job_id)
    stages = job_service.list_job_stages(session, job_id)
    session.commit()
    return job_to_read(job, stages)


@router.post(
    "/{job_id}/retry", response_model=JobRead, status_code=http_status.HTTP_202_ACCEPTED
)
def retry_job(job_id: str, payload: JobRetryRequest, session: DbSessionDep) -> JobRead:
    """Requeue a failed or canceled job, optionally from a given stage."""
    job = job_service.retry(session, job_id, from_stage=payload.from_stage)
    stages = job_service.list_job_stages(session, job_id)
    session.commit()
    return job_to_read(job, stages)


@router.get("/{job_id}/events", response_model=JobEventList)
def list_events(
    job_id: str,
    session: DbSessionDep,
    after_id: Annotated[int, Query(ge=0)] = 0,
    limit: LimitQuery = 200,
) -> JobEventList:
    """Return job events with ``id > after_id`` (cheap polling for the UI)."""
    events = job_service.list_events(session, job_id, after_id=after_id, limit=limit)
    return JobEventList(items=[event_to_read(event) for event in events])
