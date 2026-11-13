"""System readiness report shape and degraded-mode behaviour."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest
from app.core.config import Settings
from app.main import create_app
from app.services.system_service import resolve_ffmpeg_path
from fastapi.testclient import TestClient


@pytest.mark.api
def test_system_status_ready_with_real_ffmpeg(client: TestClient, settings: Settings) -> None:
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    body = response.json()

    assert body["ready"] is True
    assert body["ffmpeg"]["found"] is True
    assert body["ffmpeg"]["libass"] is True
    assert body["ffprobe"]["found"] is True
    assert body["ffmpeg"]["path"] == str(resolve_ffmpeg_path(settings))
    assert body["ffmpeg"]["version"]

    assert set(body["worker"]) == {"alive", "last_heartbeat_ms"}
    assert body["worker"]["alive"] is False
    assert body["worker"]["last_heartbeat_ms"] is None
    assert body["disk"]["min_free_gb"] == settings.min_free_disk_gb
    assert body["disk"]["free_gb"] > 0
    assert body["data_dir"] == str(settings.data_dir)


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def unreachable_ollama_client(settings: Settings) -> Iterator[TestClient]:
    port = _closed_port()
    degraded = settings.model_copy(
        update={
            "ollama_url": f"http://127.0.0.1:{port}",
            "ollama_timeout_seconds": 0.2,
        }
    )
    with TestClient(create_app(degraded)) as test_client:
        yield test_client


@pytest.mark.api
def test_unreachable_ollama_does_not_change_readiness(
    unreachable_ollama_client: TestClient,
) -> None:
    body = unreachable_ollama_client.get("/api/v1/system/status").json()
    assert body["ollama"]["reachable"] is False
    assert body["ready"] is True
