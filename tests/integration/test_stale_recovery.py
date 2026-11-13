"""Stale running jobs are retried once or failed with WORKER_LOST."""

from __future__ import annotations

import pytest
from app.core.clock import utc_now_ms
from app.core.status import JobStatus, StageStatus
from app.models.job import Job
from app.services import job_service
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

STALE_SECONDS = 60.0


def _age_heartbeat(session: Session, job_id: str, *, age_ms: int) -> None:
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(heartbeat_at=utc_now_ms() - age_ms, updated_at=utc_now_ms())
    )
    session.commit()


@pytest.mark.unit
def test_stale_running_job_is_requeued(session_factory: sessionmaker[Session], engine) -> None:
    with session_factory() as session:
        job = job_service.enqueue(session, params={"noop_duration_s": 0.0})
        job_id = job.id
        session.commit()

    claimed = job_service.claim_next(engine, "worker-a")
    assert claimed is not None

    with session_factory() as session:
        _age_heartbeat(session, job_id, age_ms=10 * 60 * 1000)

    recovered = job_service.recover_stale(engine, STALE_SECONDS, utc_now_ms())
    assert recovered == [job_id]

    with session_factory() as session:
        job = job_service.get_job(session, job_id)
        stages = job_service.list_job_stages(session, job_id)
        assert job.status == JobStatus.QUEUED.value
        assert job.worker_id is None
        assert job.current_stage is None
        assert job.heartbeat_at is None
        assert stages[0].name == "noop"
        assert stages[0].status == StageStatus.PENDING.value
        assert stages[0].attempt == 1
        events = job_service.list_events(session, job_id)
        assert any(event.code == "job_requeued" for event in events)


@pytest.mark.unit
def test_fresh_heartbeat_is_not_recovered(
    session_factory: sessionmaker[Session], engine
) -> None:
    with session_factory() as session:
        job = job_service.enqueue(session, params={"noop_duration_s": 0.0})
        job_id = job.id
        session.commit()

    assert job_service.claim_next(engine, "worker-a") is not None
    recovered = job_service.recover_stale(engine, STALE_SECONDS, utc_now_ms())
    assert recovered == []

    with session_factory() as session:
        job = job_service.get_job(session, job_id)
        assert job.status == JobStatus.RUNNING.value
        assert job.worker_id == "worker-a"


@pytest.mark.unit
def test_exhausted_job_becomes_failed_with_worker_lost(
    session_factory: sessionmaker[Session], engine
) -> None:
    with session_factory() as session:
        job = job_service.enqueue(session, params={"noop_duration_s": 0.0}, max_attempts=1)
        job_id = job.id
        session.commit()

    claimed = job_service.claim_next(engine, "worker-a")
    assert claimed is not None
    assert claimed.attempt == 1

    with session_factory() as session:
        _age_heartbeat(session, job_id, age_ms=10 * 60 * 1000)

    recovered = job_service.recover_stale(engine, STALE_SECONDS, utc_now_ms())
    assert recovered == [job_id]

    with session_factory() as session:
        job = job_service.get_job(session, job_id)
        stages = job_service.list_job_stages(session, job_id)
        assert job.status == JobStatus.FAILED.value
        assert job.error_code == "WORKER_LOST"
        assert job.finished_at is not None
        assert stages[0].status == StageStatus.FAILED.value
        events = job_service.list_events(session, job_id)
        assert any(
            event.code == "job_failed" and event.level == "error" for event in events
        )
