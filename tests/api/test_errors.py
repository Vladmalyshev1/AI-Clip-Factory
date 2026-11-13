"""Error envelope, HTTP status mapping and request id correlation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _assert_envelope(response_json: dict, *, code: str, request_id: str) -> None:
    assert response_json["error"]["code"] == code
    assert response_json["error"]["request_id"] == request_id
    assert isinstance(response_json["error"]["message"], str)
    assert response_json["error"]["message"]
    assert "details" in response_json["error"]


@pytest.mark.api
def test_unknown_job_returns_404_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    request_id = response.headers["x-request-id"]
    _assert_envelope(body, code="JOB_NOT_FOUND", request_id=request_id)


@pytest.mark.api
def test_validation_error_returns_422_envelope(client: TestClient) -> None:
    response = client.post("/api/v1/jobs", json={"params": "not-a-dict"})
    assert response.status_code == 422
    body = response.json()
    request_id = response.headers["x-request-id"]
    _assert_envelope(body, code="VALIDATION_ERROR", request_id=request_id)
    assert isinstance(body["error"]["details"], list)


@pytest.mark.api
def test_unknown_route_returns_404_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    body = response.json()
    _assert_envelope(
        body, code="NOT_FOUND", request_id=response.headers["x-request-id"]
    )


@pytest.mark.api
def test_method_not_allowed_returns_405_envelope(client: TestClient) -> None:
    response = client.delete("/api/v1/jobs")
    assert response.status_code == 405
    body = response.json()
    _assert_envelope(
        body, code="METHOD_NOT_ALLOWED", request_id=response.headers["x-request-id"]
    )


@pytest.mark.api
def test_error_echoes_incoming_request_id(client: TestClient) -> None:
    response = client.get(
        "/api/v1/jobs/missing", headers={"X-Request-ID": "trace-42"}
    )
    assert response.status_code == 404
    assert response.headers["x-request-id"] == "trace-42"
    assert response.json()["error"]["request_id"] == "trace-42"
