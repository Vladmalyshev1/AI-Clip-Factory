"""Resumable orchestration over the stage registry."""

from __future__ import annotations

import logging
import threading
import time

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.db import make_session_factory
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.paths import safe_name
from app.core.status import JobStatus, StageStatus
from app.models.job import Job
from app.pipeline.stages import REGISTRY
from app.pipeline.stages.base import ProgressReporter, Stage, StageContext
from app.services import job_service
from app.services.job_service import ClaimedJob

logger = logging.getLogger(__name__)

PROGRESS_PERSIST_INTERVAL_S = 0.2
CANCEL_POLL_INTERVAL_S = 0.05


class StageProgressReporter(ProgressReporter):
    """Throttled progress persistence plus cancellation polling for one stage."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        *,
        job_id: str,
        stage_name: str,
        rank: int,
        total: int,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._factory = factory
        self._job_id = job_id
        self._stage_name = stage_name
        self._rank = rank
        self._total = total
        self._stop_event = stop_event
        self._last_persist = 0.0
        self._last_cancel_check = 0.0
        self._cancelled = False

    def update(self, progress: float, message: str | None = None) -> None:
        value = max(0.0, min(1.0, float(progress)))
        now = time.monotonic()
        if value < 1.0 and now - self._last_persist < PROGRESS_PERSIST_INTERVAL_S:
            return
        self._last_persist = now
        overall = (self._rank + value) / self._total if self._total else value
        if message:
            logger.debug("Job %s stage %s: %s", self._job_id, self._stage_name, message)
        with self._factory() as session:
            job_service.set_progress(
                session, self._job_id, overall, current_stage=self._stage_name
            )
            session.commit()

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise StageCancelled(f"Job {self._job_id} was cancelled")
        if self._stop_event is not None and self._stop_event.is_set():
            self._cancelled = True
            raise StageCancelled(f"Worker shutdown requested during job {self._job_id}")
        now = time.monotonic()
        if now - self._last_cancel_check < CANCEL_POLL_INTERVAL_S:
            return
        self._last_cancel_check = now
        with self._factory() as session:
            job = session.get(Job, self._job_id)
            cancelled = job is not None and (
                bool(job.cancel_requested) or job.status == JobStatus.CANCEL_REQUESTED.value
            )
        self._cancelled = cancelled
        if cancelled:
            raise StageCancelled(f"Job {self._job_id} was cancelled")


def _fail_stage_and_job(
    factory: sessionmaker[Session],
    job_id: str,
    stage_name: str,
    *,
    error_code: str,
    error_message: str,
) -> None:
    with factory() as session:
        job_service.mark_stage_failed(
            session,
            job_id,
            stage_name,
            error_code=error_code,
            error_message=error_message,
        )
        job_service.mark_job_failed(
            session,
            job_id,
            error_code=error_code,
            error_message=error_message,
        )
        session.commit()


def run_job(
    engine: Engine,
    settings: Settings,
    job: ClaimedJob,
    worker_id: str,
    *,
    stop_event: threading.Event | None = None,
) -> JobStatus:
    """Execute all not-yet-succeeded stages of a claimed job.

    Returns the final job status. Succeeded stages are skipped, which is what
    makes a restarted worker resume instead of redoing work.
    """
    factory = make_session_factory(engine)
    stages: tuple[Stage, ...] = REGISTRY
    total = len(stages)
    if total == 0:  # pragma: no cover - registry is never empty in practice
        with factory() as session:
            job_service.mark_job_succeeded(session, job.job_id)
            session.commit()
        return JobStatus.SUCCEEDED

    for rank, stage in enumerate(stages):
        with factory() as session:
            current = job_service.get_job(session, job.job_id)
            stage_row = job_service.get_job_stage(session, job.job_id, stage.name)
            if stage_row.status == StageStatus.SUCCEEDED.value:
                continue
            if (
                current.status == JobStatus.CANCEL_REQUESTED.value
                or bool(current.cancel_requested)
            ):
                job_service.finish_canceled(session, job.job_id)
                session.commit()
                return JobStatus.CANCELED
            if stop_event is not None and stop_event.is_set():
                job_service.interrupt_job(session, job.job_id)
                session.commit()
                return JobStatus.QUEUED
            if stage_row.status == StageStatus.PENDING.value:
                job_service.start_stage(session, job.job_id, stage.name)
                session.commit()
            elif stage_row.status != StageStatus.RUNNING.value:
                error_message = (
                    f"Stage '{stage.name}' is '{stage_row.status}' and blocks the job; "
                    "retry the job to continue"
                )
                job_service.mark_job_failed(
                    session,
                    job.job_id,
                    error_code=ErrorCode.STAGE_FAILED.value,
                    error_message=error_message,
                )
                session.commit()
                logger.error("Job %s blocked: %s", job.job_id, error_message)
                return JobStatus.FAILED
            checkpoint = job_service.load_json_dict(stage_row.checkpoint_json) or None
            params = job_service.load_json_dict(current.params_json) or dict(job.params)
            video_id = current.video_id
            attempt = current.attempt

        work_dir = settings.work_dir / safe_name(job.job_id) / safe_name(stage.name)
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(
            job_id=job.job_id,
            video_id=video_id,
            attempt=attempt,
            work_dir=work_dir,
            params=params,
            checkpoint=checkpoint,
        )
        reporter: ProgressReporter = StageProgressReporter(
            factory,
            job_id=job.job_id,
            stage_name=stage.name,
            rank=rank,
            total=total,
            stop_event=stop_event,
        )
        try:
            result = stage.run(ctx, reporter)
        except StageCancelled:
            with factory() as session:
                current = job_service.get_job(session, job.job_id)
                user_cancelled = (
                    current.status == JobStatus.CANCEL_REQUESTED.value
                    or bool(current.cancel_requested)
                )
                if user_cancelled:
                    job_service.finish_canceled(session, job.job_id, stage_name=stage.name)
                    status = JobStatus.CANCELED
                else:
                    # Graceful shutdown: keep the work, put the job back in the queue.
                    job_service.interrupt_job(session, job.job_id, stage_name=stage.name)
                    status = JobStatus.QUEUED
                session.commit()
            logger.info(
                "Job %s stopped during stage '%s': %s", job.job_id, stage.name, status.value
            )
            return status
        except AppError as exc:
            _fail_stage_and_job(
                factory,
                job.job_id,
                stage.name,
                error_code=exc.code.value,
                error_message=exc.message,
            )
            return JobStatus.FAILED
        except Exception as exc:
            logger.exception("Stage '%s' of job %s failed", stage.name, job.job_id)
            _fail_stage_and_job(
                factory,
                job.job_id,
                stage.name,
                error_code=ErrorCode.STAGE_FAILED.value,
                error_message=f"Stage '{stage.name}' failed: {type(exc).__name__}: {exc}",
            )
            return JobStatus.FAILED

        with factory() as session:
            job_service.mark_stage_succeeded(
                session,
                job.job_id,
                stage.name,
                checkpoint=result.checkpoint,
                output=result.output,
            )
            job_service.set_progress(
                session, job.job_id, (rank + 1) / total, current_stage=stage.name
            )
            session.commit()
        logger.info("Job %s stage '%s' succeeded", job.job_id, stage.name)

    with factory() as session:
        current = job_service.get_job(session, job.job_id)
        if current.status == JobStatus.CANCEL_REQUESTED.value or bool(current.cancel_requested):
            job_service.finish_canceled(session, job.job_id)
            session.commit()
            return JobStatus.CANCELED
        if stop_event is not None and stop_event.is_set():
            job_service.interrupt_job(session, job.job_id)
            session.commit()
            return JobStatus.QUEUED
        job_service.mark_job_succeeded(session, job.job_id)
        session.commit()
    logger.info("Job %s finished successfully (worker %s)", job.job_id, worker_id)
    return JobStatus.SUCCEEDED
