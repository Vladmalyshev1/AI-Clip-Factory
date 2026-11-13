"""Cancellation semantics, including a running worker reacting within 2 seconds."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from app.core.config import Settings
from app.core.status import JobStatus, StageStatus
from app.pipeline.orchestrator import run_job
from app.services import job_service
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


@pytest.mark.api
def test_cancel_queued_job_is_immediate(
    session_factory: sessionmaker[Session], engine: Engine
) -> None:
    with session_factory() as session:
        job = job_service.enqueue(session, params={"noop_duration_s": 0.0})
        job_id = job.id
        session.commit()

    with session_factory() as session:
        job = job_service.request_cancel(session, job_id)
        session.commit()
        assert job.status == JobStatus.CANCELED.value

    assert job_service.claim_next(engine, "worker-a") is None


@pytest.mark.windows_only
def test_graceful_shutdown_requeues_the_running_job(
    session_factory: sessionmaker[Session], engine: Engine, settings: Settings
) -> None:
    """Ctrl+C/SIGTERM must preserve work: the job goes back to the queue."""
    with session_factory() as session:
        job = job_service.enqueue(session, params={"noop_duration_s": 30.0})
        job_id = job.id
        session.commit()

    claimed = job_service.claim_next(engine, "worker-a")
    assert claimed is not None

    stop_event = threading.Event()
    timer = threading.Timer(0.2, stop_event.set)
    started = time.monotonic()
    timer.start()
    try:
        status = run_job(engine, settings, claimed, "worker-a", stop_event=stop_event)
    finally:
        timer.cancel()

    assert status is JobStatus.QUEUED
    assert time.monotonic() - started < 2.0

    with session_factory() as session:
        job = job_service.get_job(session, job_id)
        stages = job_service.list_job_stages(session, job_id)
        assert job.status == JobStatus.QUEUED.value
        assert job.worker_id is None
        assert job.cancel_requested == 0
        assert job.attempt == 0, "a graceful stop must not consume an attempt"
        assert stages[0].status == StageStatus.PENDING.value
        codes = [event.code for event in job_service.list_events(session, job_id)]
        assert "job_interrupted" in codes
        assert "job_canceled" not in codes

    # The requeued job is claimable again instead of being lost.
    again = job_service.claim_next(engine, "worker-b")
    assert again is not None
    assert again.job_id == job_id

    # A user cancellation still wins over a shutdown request.
    with session_factory() as session:
        job = job_service.request_cancel(session, job_id)
        session.commit()
        assert job.status == JobStatus.CANCEL_REQUESTED.value
    assert run_job(engine, settings, again, "worker-b", stop_event=stop_event) is JobStatus.CANCELED
    with session_factory() as session:
        assert job_service.get_job(session, job_id).status == JobStatus.CANCELED.value


@pytest.mark.windows_only
def test_running_worker_stops_within_two_seconds(
    client: TestClient,
    session_factory: sessionmaker[Session],
    spawn_worker: Callable[..., Any],
    wait_until: Callable[..., Any],
) -> None:
    created = client.post(
        "/api/v1/jobs",
        json={"params": {"noop_duration_s": 30.0, "noop2_duration_s": 30.0}},
    ).json()
    job_id = created["id"]

    worker = spawn_worker("--once")

    def is_running() -> bool:
        with session_factory() as session:
            return job_service.get_job(session, job_id).status == JobStatus.RUNNING.value

    assert wait_until(is_running, timeout=20), "worker did not start the job"

    started = time.monotonic()
    response = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert response.status_code == 202
    assert response.json()["status"] == JobStatus.CANCEL_REQUESTED.value

    def cancel_finished() -> bool:
        with session_factory() as session:
            job = job_service.get_job(session, job_id)
            stages = {stage.name: stage.status for stage in job_service.list_job_stages(session, job_id)}
            return job.status == JobStatus.CANCELED.value and stages["noop"] == StageStatus.CANCELED.value

    assert wait_until(cancel_finished, timeout=2.0, interval=0.02), "job was not canceled in time"
    assert worker.wait(timeout=5) == 0
    elapsed = time.monotonic() - started
    assert elapsed <= 2.0, f"worker took {elapsed:.2f}s to stop"

    with session_factory() as session:
        job = job_service.get_job(session, job_id)
        stages = {stage.name: stage.status for stage in job_service.list_job_stages(session, job_id)}
        assert job.finished_at is not None
        assert stages == {
            "noop": StageStatus.CANCELED.value,
            "noop2": StageStatus.CANCELED.value,
        }
        codes = [event.code for event in job_service.list_events(session, job_id)]
        assert "cancel_requested" in codes
        assert "job_canceled" in codes
