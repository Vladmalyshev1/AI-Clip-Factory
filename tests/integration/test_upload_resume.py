"""Resumable uploads across an app restart and bounded server-side memory."""

from __future__ import annotations

import hashlib
import os
import tracemalloc
from pathlib import Path

import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient

API = "/api/v1"
CHUNK_BYTES = 1024 * 1024
# 32 MiB total in 1 MiB chunks. The threshold is intentionally far below the
# total: it only passes when the server streams each chunk to disk and never
# accumulates the whole body in memory.
TOTAL_BYTES = 32 * CHUNK_BYTES
MAX_PEAK_BYTES = 8 * CHUNK_BYTES


def _create_upload(client: TestClient, *, total_bytes: int) -> dict[str, object]:
    response = client.post(
        f"{API}/uploads",
        json={"filename": "resume.mp4", "total_bytes": total_bytes},
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def _put_chunk(client: TestClient, upload_id: str, *, offset: int, data: bytes) -> dict[str, object]:
    response = client.put(
        f"{API}/uploads/{upload_id}/chunk",
        params={"offset": offset},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _complete(client: TestClient, upload_id: str) -> dict[str, object]:
    response = client.post(f"{API}/uploads/{upload_id}/complete")
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.api
def test_upload_resumes_from_a_second_app_instance(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    payload = os.urandom(3 * 97)
    first, second, third = payload[:97], payload[97:194], payload[194:]
    digest = hashlib.sha256(payload).hexdigest()

    upload = _create_upload(client, total_bytes=len(payload))
    upload_id = str(upload["id"])
    _put_chunk(client, upload_id, offset=0, data=first)
    _put_chunk(client, upload_id, offset=len(first), data=second)

    # A brand-new app/client pair on the same settings and data directory models
    # a fresh API process: only the persisted offset may be carried over.
    fresh_app = create_app(settings)
    with TestClient(fresh_app) as resumed:
        state = resumed.get(f"{API}/uploads/{upload_id}")
        assert state.status_code == 200
        assert state.json()["received_bytes"] == len(first) + len(second)
        assert state.json()["status"] == "uploading"

        after_last = _put_chunk(resumed, upload_id, offset=len(first) + len(second), data=third)
        assert after_last["received_bytes"] == len(payload)

        video = _complete(resumed, upload_id)

    assert video["sha256"] == digest
    assert Path(str(video["source_path"])).is_file()
    assert _sha256(Path(str(video["source_path"]))) == digest

    stored = client.get(f"{API}/videos/{video['id']}")
    assert stored.status_code == 200
    assert stored.json()["sha256"] == digest


@pytest.mark.api
def test_chunked_upload_does_not_buffer_the_whole_body(settings: Settings) -> None:
    tuned = settings.model_copy(
        update={
            "upload_chunk_size": CHUNK_BYTES,
            "max_upload_bytes": TOTAL_BYTES * 2,
        }
    )
    payload = os.urandom(TOTAL_BYTES)
    digest = hashlib.sha256(payload).hexdigest()

    with TestClient(create_app(tuned)) as client:
        upload = _create_upload(client, total_bytes=TOTAL_BYTES)
        upload_id = str(upload["id"])

        # Payload allocated before tracing on purpose: only the server-side work
        # (plus one chunk slice) is measured.
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            for offset in range(0, TOTAL_BYTES, CHUNK_BYTES):
                _put_chunk(
                    client,
                    upload_id,
                    offset=offset,
                    data=payload[offset : offset + CHUNK_BYTES],
                )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < MAX_PEAK_BYTES, (
            f"peak traced memory {peak} bytes for a {TOTAL_BYTES}-byte upload "
            f"in {CHUNK_BYTES}-byte chunks; the request body is being buffered"
        )

        video = _complete(client, upload_id)
        assert video["sha256"] == digest
        source = Path(str(video["source_path"]))
        assert source.stat().st_size == TOTAL_BYTES
        assert _sha256(source) == digest
