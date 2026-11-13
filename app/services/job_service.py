"""Job lifecycle service: enqueue, claim, heartbeat, recovery, cancel, retry.

Every mutation runs inside a short transaction. A session is never kept open
while a pipeline stage executes, which keeps SQLite writes non-blocking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, func, or_, select
from sqlalchemy.orm import Session

from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode, ensure_job_transition, ensure_stage_transition
from app.core.status import JobStatus, StageStatus
from app.models.job import Job, JobEvent, JobStage

logger = logging.getLogger(__name__)

DEFAULT_JOB_TYPE = "pipeline"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_EVENT_LIMIT = 200


@dataclass(frozen=True)
class ClaimedJob:
    """Immutable snapshot of a claimed job handed to the orchestrator."""

    job_id: str
    worker_id: str
    type: str
    video_id: str | None
    attempt: int
    max_attempts: int
    params: dict[str, Any]


def dump_json(value: object) -> str:
    """Serialise a JSON column value deterministically."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def load_json_dict(value: str | None) -> dict[str, Any]:
    """Parse a JSON object column, tolerating NULL and malformed values."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed JSON column value")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clamp_progress(value: float) -> float:
    """Clamp a progress value into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


def _add_event(
    session: Session,
    job_id: str,
    *,
    level: str,
    message: str,
    code: str | None = None,
    stage: str | None = None,
    now_ms: int | None = None,
    payload: dict[str, Any] | None = None,
) -> JobEvent:
    event = JobEvent(
        job_id=job_id,
        ts=now_ms if now_ms is not None else utc_now_ms(),
        level=level,
        stage=stage,
        code=code,
        message=message,
        payload_json=dump_json(payload) if payload is not None else None,
    )
    session.add(event)
    return event


def get_job(session: Session, job_id: str) -> Job:
    """Load a job or raise ``JOB_NOT_FOUND``."""
    job = session.get(Job, job_id)
    if job is None:
        raise AppError(ErrorCode.JOB_NOT_FOUND, f"Job '{job_id}' not found", status_code=404)
    return job


def get_job_stage(session: Session, job_id: str, stage_name: str) -> JobStage:
    """Load one stage of a job or raise ``STAGE_NOT_FOUND``."""
    stage = session.scalars(
        select(JobStage).where(JobStage.job_id == job_id, JobStage.name == stage_name)
    ).first()
    if stage is None:
        raise AppError(
            ErrorCode.STAGE_NOT_FOUND,
            f"Job '{job_id}' has no stage '{stage_name}'",
            status_code=409,
        )
    return stage


def list_job_stages(session: Session, job_id: str) -> list[JobStage]:
    """Return the stages of a job ordered by ordinal."""
    return list(
        session.scalars(
            select(JobStage).where(JobStage.job_id == job_id).order_by(JobStage.ordinal)
        ).all()
    )


def _registry_stage_names() -> list[str]:
    from app.pipeline.stages import REGISTRY

    return [stage.name for stage in REGISTRY]


def _ensure_stages(session: Session, job: Job) -> list[JobStage]:
    """Create missing stage rows for the registry stages (idempotent)."""
    existing = {stage.name: stage for stage in list_job_stages(session, job.id)}
    stages: list[JobStage] = []
    for ordinal, name in enumerate(_registry_stage_names()):
        stage = existing.get(name)
        if stage is None:
            stage = JobStage(
                job_id=job.id,
                name=name,
                ordinal=ordinal,
                status=StageStatus.PENDING.value,
                progress=0.0,
                attempt=0,
            )
            session.add(stage)
        elif stage.ordinal != ordinal:
            stage.ordinal = ordinal
        stages.append(stage)
    session.flush()
    return sorted(stages, key=lambda item: item.ordinal)


def enqueue(
    session: Session,
    *,
    type: str = DEFAULT_JOB_TYPE,
    params: dict[str, Any] | None = None,
    video_id: str | None = None,
    max_attempts: int | None = None,
) -> Job:
    """Create a queued job; stages are created when a worker claims it."""
    now = utc_now_ms()
    job = Job(
        type=type,
        video_id=video_id,
        status=JobStatus.QUEUED.value,
        priority=0,
        progress=0.0,
        attempt=0,
        max_attempts=max_attempts if max_attempts is not None else DEFAULT_MAX_ATTEMPTS,
        params_json=dump_json(params or {}),
        cancel_requested=0,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.flush()
    _add_event(
        session,
        job.id,
        level="info",
        code="job_enqueued",
        message="Job enqueued",
        now_ms=now,
        payload={"type": type, "video_id": video_id},
    )
    return job


def claim_next(engine: Engine, worker_id: str) -> ClaimedJob | None:
    """Atomically claim the next queued job.

    ``BEGIN IMMEDIATE`` takes SQLite's write lock before the select, so two
    workers can never observe the same candidate row.
    """
    now = utc_now_ms()
    with Session(engine, expire_on_commit=False) as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        job_id = session.scalars(
            select(Job.id)
            .where(
                Job.status == JobStatus.QUEUED.value,
                Job.attempt < Job.max_attempts,
            )
            .order_by(Job.priority.desc(), Job.created_at.asc())
            .limit(1)
        ).first()
        if job_id is None:
            session.rollback()
            return None

        job = session.get(Job, job_id)
        if job is None:  # pragma: no cover - deleted between select and get
            session.rollback()
            return None

        job.status = JobStatus.RUNNING.value
        job.worker_id = worker_id
        job.attempt += 1
        job.heartbeat_at = now
        job.started_at = job.started_at or now
        job.finished_at = None
        job.updated_at = now
        job.cancel_requested = 0

        _add_event(
            session,
            job.id,
            level="info",
            code="job_claimed",
            message=f"Job claimed by {worker_id} (attempt {job.attempt}/{job.max_attempts})",
            now_ms=now,
            payload={"worker_id": worker_id, "attempt": job.attempt},
        )

        stages = _ensure_stages(session, job)
        startable = next(
            (
                stage
                for stage in stages
                if stage.status
                in (StageStatus.PENDING.value, StageStatus.RUNNING.value)
            ),
            None,
        )
        if startable is not None and startable.status == StageStatus.PENDING.value:
            _start_stage_row(session, job, startable, now_ms=now)

        session.commit()
        return ClaimedJob(
            job_id=job.id,
            worker_id=worker_id,
            type=job.type,
            video_id=job.video_id,
            attempt=job.attempt,
            max_attempts=job.max_attempts,
            params=load_json_dict(job.params_json),
        )


def _start_stage_row(
    session: Session, job: Job, stage: JobStage, *, now_ms: int | None = None
) -> JobStage:
    now = now_ms if now_ms is not None else utc_now_ms()
    ensure_stage_transition(stage.status, StageStatus.RUNNING)
    stage.status = StageStatus.RUNNING.value
    stage.attempt += 1
    stage.started_at = now
    stage.finished_at = None
    stage.error_code = None
    stage.error_message = None
    job.current_stage = stage.name
    job.updated_at = now
    _add_event(
        session,
        job.id,
        level="info",
        code="stage_started",
        stage=stage.name,
        message=f"Stage '{stage.name}' started (attempt {stage.attempt})",
        now_ms=now,
    )
    return stage


def start_stage(
    session: Session, job_id: str, stage_name: str, *, now_ms: int | None = None
) -> JobStage:
    """Mark a pending stage as running and bump its attempt."""
    job = get_job(session, job_id)
    stage = get_job_stage(session, job_id, stage_name)
    return _start_stage_row(session, job, stage, now_ms=now_ms)


def mark_stage_succeeded(
    session: Session,
    job_id: str,
    stage_name: str,
    *,
    checkpoint: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> JobStage:
    """Persist a stage result and flip the stage to ``succeeded``."""
    now = now_ms if now_ms is not None else utc_now_ms()
    stage = get_job_stage(session, job_id, stage_name)
    stage.status = StageStatus.SUCCEEDED.value
    stage.progress = 1.0
    stage.finished_at = now
    stage.error_code = None
    stage.error_message = None
    if checkpoint is not None:
        stage.checkpoint_json = dump_json(checkpoint)
    if output is not None:
        stage.output_json = dump_json(output)
    job = get_job(session, job_id)
    job.updated_at = now
    _add_event(
        session,
        job_id,
        level="info",
        code="stage_succeeded",
        stage=stage_name,
        message=f"Stage '{stage_name}' succeeded",
        now_ms=now,
    )
    return stage


def mark_stage_failed(
    session: Session,
    job_id: str,
    stage_name: str,
    *,
    error_code: str,
    error_message: str,
    now_ms: int | None = None,
) -> JobStage:
    """Persist a stage failure."""
    now = now_ms if now_ms is not None else utc_now_ms()
    stage = get_job_stage(session, job_id, stage_name)
    stage.status = StageStatus.FAILED.value
    stage.finished_at = now
    stage.error_code = error_code
    stage.error_message = error_message
    job = get_job(session, job_id)
    job.updated_at = now
    _add_event(
        session,
        job_id,
        level="error",
        code="stage_failed",
        stage=stage_name,
        message=f"Stage '{stage_name}' failed: {error_message}",
        now_ms=now,
        payload={"error_code": error_code},
    )
    return stage


def mark_stage_canceled(
    session: Session, job_id: str, stage_name: str, *, now_ms: int | None = None
) -> JobStage:
    """Persist a stage cancellation."""
    now = now_ms if now_ms is not None else utc_now_ms()
    stage = get_job_stage(session, job_id, stage_name)
    if stage.status in (StageStatus.RUNNING.value, StageStatus.PENDING.value):
        stage.status = StageStatus.CANCELED.value
        stage.finished_at = now
        job = get_job(session, job_id)
        job.updated_at = now
        _add_event(
            session,
            job_id,
            level="info",
            code="stage_canceled",
            stage=stage_name,
            message=f"Stage '{stage_name}' canceled",
            now_ms=now,
        )
    return stage


def set_progress(
    session: Session,
    job_id: str,
    progress: float,
    *,
    current_stage: str | None = None,
    now_ms: int | None = None,
) -> Job:
    """Persist clamped job progress (and optionally the current stage)."""
    now = now_ms if now_ms is not None else utc_now_ms()
    job = get_job(session, job_id)
    job.progress = clamp_progress(progress)
    if current_stage is not None:
        job.current_stage = current_stage
    job.updated_at = now
    return job


def heartbeat(session: Session, job_id: str, worker_id: str, *, now_ms: int | None = None) -> bool:
    """Refresh ``heartbeat_at`` only while this worker still owns a running job."""
    now = now_ms if now_ms is not None else utc_now_ms()
    job = session.get(Job, job_id)
    if job is None or job.status != JobStatus.RUNNING.value or job.worker_id != worker_id:
        return False
    job.heartbeat_at = now
    job.updated_at = now
    session.commit()
    return True


def recover_stale(engine: Engine, stale_seconds: float, now_ms: int) -> list[str]:
    """Requeue (or fail) jobs whose worker stopped sending heartbeats."""
    threshold = now_ms - int(max(0.0, stale_seconds) * 1000)
    affected: list[str] = []
    with Session(engine, expire_on_commit=False) as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        stale_jobs = list(
            session.scalars(
                select(Job)
                .where(
                    Job.status.in_(
                        (JobStatus.RUNNING.value, JobStatus.CANCEL_REQUESTED.value)
                    ),
                    or_(Job.heartbeat_at.is_(None), Job.heartbeat_at < threshold),
                )
                .order_by(Job.created_at)
            ).all()
        )
        for job in stale_jobs:
            affected.append(job.id)
            running_stage = session.scalars(
                select(JobStage).where(
                    JobStage.job_id == job.id,
                    JobStage.status == StageStatus.RUNNING.value,
                )
            ).first()
            if job.status == JobStatus.CANCEL_REQUESTED.value:
                _finalize_cancel(session, job, now_ms, stage=running_stage)
            elif job.attempt >= job.max_attempts:
                _fail_worker_lost(session, job, now_ms, stage=running_stage)
            else:
                _requeue_stale(session, job, now_ms, stage=running_stage)
        session.commit()
    if affected:
        logger.warning("Recovered %d stale job(s): %s", len(affected), ", ".join(affected))
    return affected


def _requeue_stale(
    session: Session, job: Job, now_ms: int, *, stage: JobStage | None
) -> None:
    if stage is not None:
        stage.status = StageStatus.PENDING.value
        stage.started_at = None
        stage.finished_at = None
    job.status = JobStatus.QUEUED.value
    job.worker_id = None
    job.current_stage = None
    job.heartbeat_at = None
    job.updated_at = now_ms
    _add_event(
        session,
        job.id,
        level="warning",
        code="job_requeued",
        stage=stage.name if stage is not None else None,
        message=(
            f"Worker heartbeat lost; job requeued (attempt {job.attempt}/{job.max_attempts})"
        ),
        now_ms=now_ms,
    )


def _fail_worker_lost(
    session: Session, job: Job, now_ms: int, *, stage: JobStage | None
) -> None:
    message = f"Worker lost after {job.attempt} attempt(s)"
    if stage is not None:
        stage.status = StageStatus.FAILED.value
        stage.finished_at = now_ms
        stage.error_code = ErrorCode.WORKER_LOST.value
        stage.error_message = message
    job.status = JobStatus.FAILED.value
    job.error_code = ErrorCode.WORKER_LOST.value
    job.error_message = message
    job.worker_id = None
    job.finished_at = now_ms
    job.updated_at = now_ms
    _add_event(
        session,
        job.id,
        level="error",
        code="job_failed",
        stage=stage.name if stage is not None else None,
        message=message,
        now_ms=now_ms,
        payload={"error_code": ErrorCode.WORKER_LOST.value},
    )


def request_cancel(session: Session, job_id: str) -> Job:
    """Cancel a queued job immediately or request cancellation of a running one."""
    job = get_job(session, job_id)
    now = utc_now_ms()
    status = JobStatus(job.status)
    if status is JobStatus.QUEUED:
        ensure_job_transition(status, JobStatus.CANCELED)
        _finalize_cancel(session, job, now, stage=None)
        return job
    if status in (JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED):
        if status is JobStatus.RUNNING:
            ensure_job_transition(status, JobStatus.CANCEL_REQUESTED)
            job.status = JobStatus.CANCEL_REQUESTED.value
            job.updated_at = now
        job.cancel_requested = 1
        _add_event(
            session,
            job.id,
            level="warning",
            code="cancel_requested",
            message="Cancellation requested",
            now_ms=now,
        )
        return job
    raise AppError(
        ErrorCode.INVALID_STATE_TRANSITION,
        f"Job '{job_id}' is already {status.value}",
        status_code=409,
    )


def _finalize_cancel(
    session: Session, job: Job, now_ms: int, *, stage: JobStage | None
) -> None:
    """Cancel every open stage and the job itself (idempotent)."""
    if job.status == JobStatus.CANCELED.value:
        return
    for row in list_job_stages(session, job.id):
        if row.status in (StageStatus.PENDING.value, StageStatus.RUNNING.value):
            row.status = StageStatus.CANCELED.value
            row.finished_at = now_ms
            _add_event(
                session,
                job.id,
                level="info",
                code="stage_canceled",
                stage=row.name,
                message=f"Stage '{row.name}' canceled",
                now_ms=now_ms,
            )
    job.status = JobStatus.CANCELED.value
    job.cancel_requested = 1
    job.worker_id = None
    job.heartbeat_at = None
    job.finished_at = now_ms
    job.updated_at = now_ms
    _add_event(
        session,
        job.id,
        level="warning",
        code="job_canceled",
        stage=stage.name if stage is not None else job.current_stage,
        message="Job canceled",
        now_ms=now_ms,
    )


def finish_canceled(
    session: Session, job_id: str, *, stage_name: str | None = None, now_ms: int | None = None
) -> Job:
    """Finalise a cancellation observed by the worker."""
    now = now_ms if now_ms is not None else utc_now_ms()
    job = get_job(session, job_id)
    stage = get_job_stage(session, job_id, stage_name) if stage_name else None
    _finalize_cancel(session, job, now, stage=stage)
    return job


def interrupt_job(
    session: Session, job_id: str, *, stage_name: str | None = None, now_ms: int | None = None
) -> Job:
    """Return a running job to the queue after a graceful worker shutdown.

    A graceful stop (Ctrl+C, SIGTERM) must not destroy hours of work: unlike a
    user cancellation, the job goes back to ``queued`` and keeps the checkpoints
    of finished stages. The interrupted attempt is refunded, because ``attempt``
    counts crashes, not planned restarts.
    """
    now = now_ms if now_ms is not None else utc_now_ms()
    job = get_job(session, job_id)
    if job.status in (
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELED.value,
    ):
        return job
    if stage_name is not None:
        stage = get_job_stage(session, job_id, stage_name)
        if stage.status == StageStatus.RUNNING.value:
            stage.status = StageStatus.PENDING.value
            stage.finished_at = None
            stage.error_code = None
            stage.error_message = None
    job.status = JobStatus.QUEUED.value
    job.attempt = max(0, job.attempt - 1)
    job.worker_id = None
    job.heartbeat_at = None
    job.current_stage = None
    job.updated_at = now
    job.error_code = None
    job.error_message = None
    _add_event(
        session,
        job_id,
        level="warning",
        code="job_interrupted",
        stage=stage_name,
        message="Worker shutdown; job returned to the queue",
        now_ms=now,
    )
    return job


def mark_job_succeeded(session: Session, job_id: str, *, now_ms: int | None = None) -> Job:
    """Mark a job as successfully finished."""
    now = now_ms if now_ms is not None else utc_now_ms()
    job = get_job(session, job_id)
    ensure_job_transition(job.status, JobStatus.SUCCEEDED)
    job.status = JobStatus.SUCCEEDED.value
    job.progress = 1.0
    job.error_code = None
    job.error_message = None
    job.finished_at = now
    job.updated_at = now
    job.heartbeat_at = None
    _add_event(
        session,
        job.id,
        level="info",
        code="job_succeeded",
        message="Job succeeded",
        now_ms=now,
    )
    return job


def mark_job_failed(
    session: Session,
    job_id: str,
    *,
    error_code: str,
    error_message: str,
    now_ms: int | None = None,
) -> Job:
    """Mark a job as failed, keeping the error for the UI and for retries."""
    now = now_ms if now_ms is not None else utc_now_ms()
    job = get_job(session, job_id)
    ensure_job_transition(job.status, JobStatus.FAILED)
    job.status = JobStatus.FAILED.value
    job.error_code = error_code
    job.error_message = error_message
    job.finished_at = now
    job.updated_at = now
    job.heartbeat_at = None
    _add_event(
        session,
        job.id,
        level="error",
        code="job_failed",
        message=f"Job failed: {error_message}",
        now_ms=now,
        payload={"error_code": error_code},
    )
    return job


def retry(session: Session, job_id: str, from_stage: str | None = None) -> Job:
    """Reset a failed/canceled job (optionally from a stage) back to ``queued``."""
    job = get_job(session, job_id)
    now = utc_now_ms()
    status = JobStatus(job.status)
    if status not in (JobStatus.FAILED, JobStatus.CANCELED):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Cannot retry a job in state '{status.value}'",
            status_code=409,
        )
    ensure_job_transition(status, JobStatus.QUEUED)

    stages = list_job_stages(session, job_id)
    start_ordinal = 0
    if from_stage is not None:
        match = next((row for row in stages if row.name == from_stage), None)
        if match is None:
            raise AppError(
                ErrorCode.STAGE_NOT_FOUND,
                f"Job '{job_id}' has no stage '{from_stage}'",
                status_code=409,
            )
        blocked = [
            row.name
            for row in stages
            if row.ordinal < match.ordinal and row.status != StageStatus.SUCCEEDED.value
        ]
        if blocked:
            raise AppError(
                ErrorCode.INVALID_STATE_TRANSITION,
                f"Cannot retry from '{from_stage}': earlier stage(s) "
                f"{', '.join(blocked)} did not succeed",
                status_code=409,
            )
        start_ordinal = match.ordinal

    for row in stages:
        if row.ordinal >= start_ordinal:
            row.status = StageStatus.PENDING.value
            row.progress = 0.0
            row.attempt = 0
            row.checkpoint_json = None
            row.output_json = None
            row.error_code = None
            row.error_message = None
            row.started_at = None
            row.finished_at = None

    job.status = JobStatus.QUEUED.value
    job.progress = 0.0
    job.attempt = 0
    job.current_stage = None
    job.error_code = None
    job.error_message = None
    job.cancel_requested = 0
    job.worker_id = None
    job.heartbeat_at = None
    job.started_at = None
    job.finished_at = None
    job.updated_at = now
    _add_event(
        session,
        job.id,
        level="info",
        code="job_retried",
        message="Job queued for retry",
        now_ms=now,
        payload={"from_stage": from_stage},
    )
    return job


def list_jobs(
    session: Session,
    *,
    video_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Job], int]:
    """Return a page of jobs (newest first) and the total matching count."""
    filters = []
    if video_id is not None:
        filters.append(Job.video_id == video_id)
    if status is not None:
        filters.append(Job.status == status)
    count_stmt = select(func.count()).select_from(Job)
    page_stmt = select(Job).order_by(Job.created_at.desc(), Job.id)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
        page_stmt = page_stmt.where(condition)
    total = session.scalar(count_stmt) or 0
    items = list(session.scalars(page_stmt.limit(limit).offset(offset)).all())
    return items, int(total)


def list_events(
    session: Session, job_id: str, *, after_id: int = 0, limit: int = DEFAULT_EVENT_LIMIT
) -> list[JobEvent]:
    """Return events of a job with ``id > after_id``, oldest first."""
    get_job(session, job_id)
    return list(
        session.scalars(
            select(JobEvent)
            .where(JobEvent.job_id == job_id, JobEvent.id > after_id)
            .order_by(JobEvent.id)
            .limit(limit)
        ).all()
    )
