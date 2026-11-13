"""Health endpoint and request id propagation."""

from __future__ import annotations

import pytest
from app import __version__
from fastapi.testclient import TestClient


@pytest.mark.api
def test_health_returns_ok_and_version(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "version": __version__}


@pytest.mark.api
def test_health_echoes_provided_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "my-request-id-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "my-request-id-123"


@pytest.mark.api
def test_health_generates_request_id_when_missing(client: TestClient) -> None:
    first = client.get("/api/v1/health")
    second = client.get("/api/v1/health")
    generated = first.headers.get("x-request-id")
    assert generated
    assert len(generated) >= 16
    assert second.headers.get("x-request-id") != generated
