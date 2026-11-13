"""Worker failure, retry and resume-after-crash behaviour."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from app.core.status import JobStatus, StageStatus
from app.services import job_service
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

STALE_ENV = {
    "ACF_JOB_STALE_SECONDS": "0.3",
    "ACF_WORKER_POLL_SECONDS": "0.1",
}


def _stage_start_count(session_factory: sessionmaker[Session], job_id: str, stage: str) -> int:
    with session_factory() as session:
        events = job_service.list_events(session, job_id, limit=500)
    return sum(1 for event in events if event.stage == stage and event.code == "stage_started")


@pytest.mark.windows_only
def test_worker_failure_retry_and_resume_after_kill(
    client: TestClient,
    session_factory: sessionmaker[Session],
    run_worker: Callable[..., Any],
    spawn_worker: Callable[..., Any],
    wait_until: Callable[..., Any],
) -> None:
    # 1) A noop stage configured to fail must fail the job and store the error.
    failed_job = client.post(
        "/api/v1/jobs", json={"params": {"fail_at": "noop", "noop_duration_s": 0.0}}
    ).json()
    result = run_worker("--once", timeout=60, env_extra=STALE_ENV)
    assert result.returncode == 0, result.stderr

    failed = client.get(f"/api/v1/jobs/{failed_job['id']}").json()
    assert failed["status"] == JobStatus.FAILED.value
    assert failed["error"] is not None
    assert failed["error"]["code"] == "STAGE_FAILED"
    assert "noop" in failed["error"]["message"]

    # 2) Retry resets the job; cancel it again so it cannot interfere below.
    retried = client.post(f"/api/v1/jobs/{failed_job['id']}/retry", json={})
    assert retried.status_code == 202
    assert retried.json()["status"] == JobStatus.QUEUED.value
    assert client.post(f"/api/v1/jobs/{failed_job['id']}/cancel").status_code == 202

    # 3) Kill a worker while it is inside the second stage.
    resume_job = client.post(
        "/api/v1/jobs",
        json={"params": {"noop_duration_s": 0.05, "noop2_duration_s": 2.0}},
    ).json()
    job_id = resume_job["id"]

    worker = spawn_worker(env_extra=STALE_ENV)

    def in_second_stage() -> bool:
        with session_factory() as session:
            job = job_service.get_job(session, job_id)
            return (
                job.status == JobStatus.RUNNING.value
                and job.current_stage == "noop2"
                and _stage_start_count(session_factory, job_id, "noop") == 1
            )

    assert wait_until(in_second_stage, timeout=25), "worker never reached the second stage"
    worker.terminate()
    assert worker.wait(timeout=10) is not None

    with session_factory() as session:
        crashed = job_service.get_job(session, job_id)
        assert crashed.status == JobStatus.RUNNING.value
        assert crashed.current_stage == "noop2"

    # 4) A fresh worker recovers the stale job and resumes from the second stage.
    time.sleep(0.6)
    resumed = run_worker("--once", timeout=90, env_extra=STALE_ENV)
    assert resumed.returncode == 0, resumed.stderr

    finished = client.get(f"/api/v1/jobs/{job_id}").json()
    assert finished["status"] == JobStatus.SUCCEEDED.value
    assert finished["progress"] == 1.0
    assert finished["error"] is None
    assert {stage["status"] for stage in finished["stages"]} == {StageStatus.SUCCEEDED.value}

    # The already-succeeded first stage must not run twice; the interrupted one did.
    assert _stage_start_count(session_factory, job_id, "noop") == 1
    assert _stage_start_count(session_factory, job_id, "noop2") >= 2
