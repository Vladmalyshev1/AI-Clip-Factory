"""Job API: create, list, get, cancel, retry and event pagination."""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.core.status import JobStatus
from app.pipeline.orchestrator import run_job
from app.services import job_service
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


def _claim_and_run(
    engine: Engine, settings: Settings, job_id: str, *, worker_id: str = "test-worker"
) -> JobStatus:
    claimed = job_service.claim_next(engine, worker_id)
    assert claimed is not None
    assert claimed.job_id == job_id
    return run_job(engine, settings, claimed, worker_id)


@pytest.mark.api
def test_create_list_and_get_job(client: TestClient) -> None:
    response = client.post(
        "/api/v1/jobs",
        json={"type": "pipeline", "video_id": "vod-1", "params": {"noop_duration_s": 0.0}},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == JobStatus.QUEUED.value
    assert created["video_id"] == "vod-1"
    assert created["params"] == {"noop_duration_s": 0.0}
    assert created["stages"] == []
    assert created["max_attempts"] == 3
    assert created["error"] is None

    listing = client.get("/api/v1/jobs", params={"video_id": "vod-1"})
    assert listing.status_code == 200
    page = listing.json()
    assert page["total"] == 1
    assert page["limit"] == 50
    assert page["offset"] == 0
    assert page["items"][0]["id"] == created["id"]

    filtered = client.get("/api/v1/jobs", params={"status": "succeeded"}).json()
    assert filtered["total"] == 0

    detail = client.get(f"/api/v1/jobs/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == created["id"]


@pytest.mark.api
def test_cancel_queued_job_then_conflict(client: TestClient) -> None:
    job = client.post("/api/v1/jobs", json={}).json()

    canceled = client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert canceled.status_code == 202
    body = canceled.json()
    assert body["status"] == JobStatus.CANCELED.value
    assert body["finished_at_ms"] is not None

    again = client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "INVALID_STATE_TRANSITION"


@pytest.mark.api
def test_retry_failed_job(
    client: TestClient,
    engine: Engine,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    created = client.post(
        "/api/v1/jobs", json={"params": {"fail_at": "noop", "noop_duration_s": 0.0}}
    ).json()
    status = _claim_and_run(engine, settings, created["id"])
    assert status is JobStatus.FAILED

    failed = client.get(f"/api/v1/jobs/{created['id']}").json()
    assert failed["status"] == JobStatus.FAILED.value
    assert failed["error"] is not None
    assert failed["error"]["code"] == "STAGE_FAILED"
    stage_statuses = {stage["name"]: stage["status"] for stage in failed["stages"]}
    assert stage_statuses == {"noop": "failed", "noop2": "pending"}

    retried = client.post(f"/api/v1/jobs/{created['id']}/retry", json={})
    assert retried.status_code == 202
    body = retried.json()
    assert body["status"] == JobStatus.QUEUED.value
    assert body["attempt"] == 0
    assert body["error"] is None
    assert {stage["status"] for stage in body["stages"]} == {"pending"}

    with session_factory() as session:
        job = job_service.get_job(session, created["id"])
        assert job.error_code is None
        assert job.worker_id is None


@pytest.mark.api
def test_retry_with_unknown_stage_conflicts(client: TestClient) -> None:
    job = client.post("/api/v1/jobs", json={}).json()
    client.post(f"/api/v1/jobs/{job['id']}/cancel")

    response = client.post(
        f"/api/v1/jobs/{job['id']}/retry", json={"from_stage": "does-not-exist"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "STAGE_NOT_FOUND"


@pytest.mark.api
def test_retry_running_job_conflicts(client: TestClient, engine: Engine) -> None:
    job = client.post("/api/v1/jobs", json={"params": {"noop_duration_s": 5.0}}).json()
    claimed = job_service.claim_next(engine, "test-worker")
    assert claimed is not None

    response = client.post(f"/api/v1/jobs/{job['id']}/retry", json={})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE_TRANSITION"


@pytest.mark.api
def test_events_pagination(
    client: TestClient, engine: Engine, settings: Settings
) -> None:
    created = client.post(
        "/api/v1/jobs", json={"params": {"fail_at": "noop", "noop_duration_s": 0.0}}
    ).json()
    assert _claim_and_run(engine, settings, created["id"]) is JobStatus.FAILED

    full = client.get(f"/api/v1/jobs/{created['id']}/events", params={"limit": 200}).json()
    events = full["items"]
    assert len(events) >= 5
    ids = [event["id"] for event in events]
    assert ids == sorted(ids)
    assert all(event["job_id"] == created["id"] for event in events)
    assert events[0]["code"] == "job_enqueued"
    assert any(event["code"] == "stage_started" and event["stage"] == "noop" for event in events)

    first_page = client.get(
        f"/api/v1/jobs/{created['id']}/events", params={"after_id": 0, "limit": 2}
    ).json()["items"]
    assert len(first_page) == 2
    assert [event["id"] for event in first_page] == ids[:2]

    cursor = first_page[-1]["id"]
    second_page = client.get(
        f"/api/v1/jobs/{created['id']}/events", params={"after_id": cursor, "limit": 2}
    ).json()["items"]
    assert [event["id"] for event in second_page] == ids[2:4]
    assert all(event["id"] > cursor for event in second_page)


@pytest.mark.api
def test_retry_from_stage_keeps_earlier_succeeded_stages(
    client: TestClient, engine: Engine, settings: Settings
) -> None:
    created = client.post(
        "/api/v1/jobs",
        json={"params": {"fail_at": "noop2", "noop_duration_s": 0.0}},
    ).json()
    assert _claim_and_run(engine, settings, created["id"]) is JobStatus.FAILED

    failed = client.get(f"/api/v1/jobs/{created['id']}").json()
    statuses = {stage["name"]: stage["status"] for stage in failed["stages"]}
    assert statuses == {"noop": "succeeded", "noop2": "failed"}

    response = client.post(
        f"/api/v1/jobs/{created['id']}/retry", json={"from_stage": "noop2"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == JobStatus.QUEUED.value
    statuses = {stage["name"]: stage["status"] for stage in body["stages"]}
    assert statuses == {"noop": "succeeded", "noop2": "pending"}


@pytest.mark.api
def test_retry_from_stage_requires_earlier_success(
    client: TestClient, engine: Engine, settings: Settings
) -> None:
    created = client.post(
        "/api/v1/jobs", json={"params": {"fail_at": "noop", "noop_duration_s": 0.0}}
    ).json()
    assert _claim_and_run(engine, settings, created["id"]) is JobStatus.FAILED

    response = client.post(
        f"/api/v1/jobs/{created['id']}/retry", json={"from_stage": "noop2"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE_TRANSITION"


@pytest.mark.api
def test_events_for_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/nope/events")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"
