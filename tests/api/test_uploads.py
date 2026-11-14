"""Upload API: session lifecycle, chunk streaming, completion and cancellation."""

from __future__ import annotations

import errno
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import UploadStatus
from app.media.binaries import find_ffmpeg
from app.models.job import new_id
from app.models.upload import Upload
from app.services import upload_service
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

API = "/api/v1"
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240


class _DiskUsage(NamedTuple):
    """Stand-in for the named tuple returned by ``shutil.disk_usage``."""

    total: int
    used: int
    free: int


@pytest.fixture(scope="module")
def sample_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-second H.264/AAC MP4, 320x240 at 10 fps, generated once."""
    binary = find_ffmpeg()
    if binary is None:
        pytest.skip("ffmpeg is not available in this environment")
    directory = tmp_path_factory.mktemp("upload-samples")
    destination = directory / "sample.mp4"
    result = subprocess.run(
        [
            str(binary),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "1.0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - environment problem
        raise RuntimeError(f"ffmpeg sample generation failed: {result.stderr[-1000:]}")
    return destination


def _create_upload(
    client: TestClient, *, filename: str = "clip.mp4", total_bytes: int = 1024
) -> dict[str, Any]:
    response = client.post(
        f"{API}/uploads", json={"filename": filename, "total_bytes": total_bytes}
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _put_chunk(
    client: TestClient,
    upload_id: str,
    *,
    offset: int,
    data: bytes,
    expected: int = 200,
) -> dict[str, Any]:
    response = client.put(
        f"{API}/uploads/{upload_id}/chunk",
        params={"offset": offset},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == expected, response.text
    body: dict[str, Any] = response.json()
    return body


def _upload_all(client: TestClient, upload_id: str, data: bytes) -> dict[str, Any]:
    return _put_chunk(client, upload_id, offset=0, data=data)


def _insert_stale_upload(
    session_factory: sessionmaker[Session], *, temp_path: Path, expires_at: int
) -> str:
    upload_id = new_id()
    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Upload(
                id=upload_id,
                filename="stale.mp4",
                total_bytes=64,
                received_bytes=8,
                status=UploadStatus.UPLOADING.value,
                temp_path=str(temp_path),
                sha256=None,
                video_id=None,
                created_at=now - 10_000,
                updated_at=now - 10_000,
                expires_at=expires_at,
            )
        )
        session.commit()
    return upload_id


def _upload_status(
    session_factory: sessionmaker[Session], upload_id: str
) -> UploadStatus:
    with session_factory() as session:
        upload = session.get(Upload, upload_id)
        assert upload is not None
        return UploadStatus(upload.status)


@pytest.mark.api
def test_create_upload_returns_resume_state(client: TestClient, settings: Settings) -> None:
    body = _create_upload(client, filename="My VOD.mp4", total_bytes=4096)

    assert body["filename"] == "My VOD.mp4"
    assert body["total_bytes"] == 4096
    assert body["received_bytes"] == 0
    assert body["status"] == UploadStatus.UPLOADING.value
    assert body["chunk_size"] == settings.upload_chunk_size
    assert body["sha256"] is None
    assert body["video_id"] is None
    assert body["expires_at_ms"] > body["created_at_ms"]


@pytest.mark.api
def test_create_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        f"{API}/uploads", json={"filename": "notes.txt", "total_bytes": 10}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


@pytest.mark.api
def test_create_upload_rejects_oversize_total(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)

    response = client.post(
        f"{API}/uploads", json={"filename": "clip.mp4", "total_bytes": 2048}
    )
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["max_upload_bytes"] == 1024
    assert error["details"]["total_bytes"] == 2048


@pytest.mark.api
def test_create_upload_disk_preflight_returns_507(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: _DiskUsage(0, 0, 0))

    response = client.post(
        f"{API}/uploads", json={"filename": "clip.mp4", "total_bytes": 1024}
    )
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "INSUFFICIENT_DISK"


@pytest.mark.api
def test_chunk_write_enospc_returns_507(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FullDisk:
        def write(self, _data: object) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    settings.ensure_dirs()
    scratch = settings.incoming_dir / "fake.chunk"
    monkeypatch.setattr(
        upload_service,
        "_chunk_temp_file",
        lambda _settings, _upload_id: (_FullDisk(), scratch),
    )

    upload = _create_upload(client, total_bytes=64)
    response = client.put(
        f"{API}/uploads/{upload['id']}/chunk",
        params={"offset": 0},
        content=b"x" * 32,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 507, response.text
    error = response.json()["error"]
    assert error["code"] == "INSUFFICIENT_DISK"
    assert error["details"]["reason"] == "disk_full"

    fetched = client.get(f"{API}/uploads/{upload['id']}").json()
    assert fetched["received_bytes"] == 0
    assert not scratch.exists()


@pytest.mark.api
def test_create_upload_expires_stale_sessions(
    client: TestClient,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    settings.ensure_dirs()
    stale_path = settings.incoming_dir / "stale-session.part"
    stale_path.write_bytes(b"partial data")
    stale_id = _insert_stale_upload(
        session_factory, temp_path=stale_path, expires_at=utc_now_ms() - 1000
    )
    assert _upload_status(session_factory, stale_id) is UploadStatus.UPLOADING

    _create_upload(client, filename="fresh.mp4", total_bytes=1024)

    assert _upload_status(session_factory, stale_id) is UploadStatus.EXPIRED
    assert not stale_path.exists()


@pytest.mark.api
def test_two_chunk_upload_advances_offset(client: TestClient) -> None:
    first = b"a" * 128
    second = b"b" * 64
    upload = _create_upload(client, total_bytes=len(first) + len(second))

    after_first = _put_chunk(client, upload["id"], offset=0, data=first)
    assert after_first["received_bytes"] == len(first)

    after_second = _put_chunk(client, upload["id"], offset=len(first), data=second)
    assert after_second["received_bytes"] == len(first) + len(second)

    fetched = client.get(f"{API}/uploads/{upload['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["received_bytes"] == len(first) + len(second)


@pytest.mark.api
def test_wrong_offset_conflicts(client: TestClient) -> None:
    upload = _create_upload(client, total_bytes=256)

    response = client.put(
        f"{API}/uploads/{upload['id']}/chunk",
        params={"offset": 5},
        content=b"0123456789",
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "UPLOAD_CONFLICT"
    assert error["details"]["expected_offset"] == 0
    assert error["details"]["received_bytes"] == 0


@pytest.mark.api
def test_oversize_chunk_is_rejected_and_offset_unchanged(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upload_chunk_size", 64)
    upload = _create_upload(client, total_bytes=1024)

    response = client.put(
        f"{API}/uploads/{upload['id']}/chunk",
        params={"offset": 0},
        content=b"x" * 128,
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    fetched = client.get(f"{API}/uploads/{upload['id']}").json()
    assert fetched["received_bytes"] == 0


@pytest.mark.api
def test_chunk_after_cancel_returns_410(client: TestClient) -> None:
    upload = _create_upload(client, total_bytes=128)
    canceled = client.delete(f"{API}/uploads/{upload['id']}")
    assert canceled.status_code == 204

    response = client.put(
        f"{API}/uploads/{upload['id']}/chunk", params={"offset": 0}, content=b"x" * 10
    )
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "UPLOAD_EXPIRED"


@pytest.mark.api
def test_chunk_after_complete_returns_409(client: TestClient) -> None:
    data = b"definitely not a real video"
    upload = _create_upload(client, total_bytes=len(data))
    _upload_all(client, upload["id"], data)

    completed = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert completed.status_code == 201, completed.text

    response = client.put(
        f"{API}/uploads/{upload['id']}/chunk",
        params={"offset": len(data)},
        content=b"more",
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "UPLOAD_CONFLICT"


@pytest.mark.api
def test_cancel_removes_temp_file(client: TestClient, settings: Settings) -> None:
    upload = _create_upload(client, total_bytes=128)
    temp_path = settings.incoming_dir / f"{upload['id']}.part"
    assert temp_path.is_file()

    response = client.delete(f"{API}/uploads/{upload['id']}")
    assert response.status_code == 204
    assert response.content == b""
    assert not temp_path.exists()

    fetched = client.get(f"{API}/uploads/{upload['id']}").json()
    assert fetched["status"] == UploadStatus.CANCELED.value

    chunk = client.put(
        f"{API}/uploads/{upload['id']}/chunk", params={"offset": 0}, content=b"x"
    )
    assert chunk.status_code == 410


@pytest.mark.api
def test_complete_incomplete_returns_409(client: TestClient) -> None:
    upload = _create_upload(client, total_bytes=64)
    _put_chunk(client, upload["id"], offset=0, data=b"a" * 32)

    response = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "UPLOAD_INCOMPLETE"
    assert error["details"]["received_bytes"] == 32
    assert error["details"]["total_bytes"] == 64


@pytest.mark.media
@pytest.mark.api
def test_complete_moves_bytes_and_probes_media(
    client: TestClient,
    settings: Settings,
    session_factory: sessionmaker[Session],
    sample_mp4: Path,
) -> None:
    data = sample_mp4.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    upload = _create_upload(client, filename="sample.mp4", total_bytes=len(data))
    temp_path = settings.incoming_dir / f"{upload['id']}.part"

    chunk = _upload_all(client, upload["id"], data)
    assert chunk["received_bytes"] == len(data)

    response = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert response.status_code == 201, response.text
    video = response.json()

    assert video["status"] == "ready"
    assert video["source_kind"] == "upload"
    assert video["width"] == SAMPLE_WIDTH
    assert video["height"] == SAMPLE_HEIGHT
    assert video["duration_ms"] is not None
    assert 900 <= video["duration_ms"] <= 1100
    assert video["sha256"] == digest
    assert video["size_bytes"] == len(data)
    assert video["original_filename"] == "sample.mp4"

    stored = Path(video["source_path"])
    assert stored.is_file()
    assert stored.parent == settings.originals_dir
    assert stored.read_bytes() == data
    assert not temp_path.exists()

    completed = client.get(f"{API}/uploads/{upload['id']}").json()
    assert completed["status"] == UploadStatus.COMPLETED.value
    assert completed["received_bytes"] == len(data)
    assert completed["sha256"] == digest
    assert completed["video_id"] == video["id"]

    with session_factory() as session:
        upload_row = session.get(Upload, upload["id"])
        assert upload_row is not None
        assert upload_row.video_id == video["id"]


@pytest.mark.media
@pytest.mark.api
def test_complete_is_idempotent(
    client: TestClient, sample_mp4: Path
) -> None:
    data = sample_mp4.read_bytes()
    upload = _create_upload(client, filename="sample.mp4", total_bytes=len(data))
    _upload_all(client, upload["id"], data)

    first = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert first.status_code == 201, first.text

    second = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert second.status_code == 200, second.text
    assert second.json()["id"] == first.json()["id"]


@pytest.mark.media
@pytest.mark.api
def test_broken_media_yields_failed_video_but_keeps_original(client: TestClient) -> None:
    data = b"this is definitely not an mp4 file"
    upload = _create_upload(client, filename="broken.mp4", total_bytes=len(data))
    _upload_all(client, upload["id"], data)

    response = client.post(f"{API}/uploads/{upload['id']}/complete")
    assert response.status_code == 201, response.text
    video = response.json()

    assert video["status"] == "failed"
    assert video["error"] is not None
    assert video["error"]["code"]
    stored = Path(video["source_path"])
    assert stored.is_file()
    assert stored.read_bytes() == data
