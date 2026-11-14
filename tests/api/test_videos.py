"""Video API: listing, inspection, local import and thumbnails."""

from __future__ import annotations

import errno
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from app.core.config import Settings
from app.core.status import SourceKind, VideoStatus
from app.media.binaries import find_ffmpeg
from app.models.job import new_id
from app.models.video import Video
from app.services import video_service
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

API = "/api/v1"
JPEG_SOI = b"\xff\xd8\xff"


@pytest.fixture(scope="module")
def sample_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-second H.264/AAC MP4, 320x240 at 10 fps, generated once."""
    binary = find_ffmpeg()
    if binary is None:
        pytest.skip("ffmpeg is not available in this environment")
    directory = tmp_path_factory.mktemp("video-samples")
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


def _insert_video(
    session_factory: sessionmaker[Session],
    *,
    title: str,
    created_at: int,
    status: VideoStatus = VideoStatus.READY,
) -> str:
    video_id = new_id()
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title=title,
                original_filename=f"{title}.mp4",
                source_path=f"C:/media/{title}.mp4",
                source_kind=SourceKind.UPLOAD.value,
                size_bytes=1024,
                sha256=None,
                status=status.value,
                created_at=created_at,
                updated_at=created_at,
            )
        )
        session.commit()
    return video_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import(client: TestClient, *, path: Path, copy: bool) -> Any:
    return client.post(
        f"{API}/videos/import",
        json={"path": str(path), "copy": copy},
    )


@pytest.mark.api
def test_list_videos_paginates_newest_first(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    oldest = _insert_video(session_factory, title="oldest", created_at=1)
    middle = _insert_video(session_factory, title="middle", created_at=2)
    newest = _insert_video(session_factory, title="newest", created_at=3)
    failed = _insert_video(
        session_factory, title="failed", created_at=4, status=VideoStatus.FAILED
    )

    first_page = client.get(f"{API}/videos", params={"limit": 2})
    assert first_page.status_code == 200
    body = first_page.json()
    assert body["total"] == 4
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert [item["id"] for item in body["items"]] == [failed, newest]

    second_page = client.get(f"{API}/videos", params={"limit": 2, "offset": 2}).json()
    assert second_page["total"] == 4
    assert [item["id"] for item in second_page["items"]] == [middle, oldest]

    filtered = client.get(f"{API}/videos", params={"status": "failed"}).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["id"] == failed


@pytest.mark.api
def test_get_video_returns_row(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory, title="recording", created_at=10)

    response = client.get(f"{API}/videos/{video_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == video_id
    assert body["title"] == "recording"
    assert body["source_kind"] == SourceKind.UPLOAD.value
    assert body["status"] == VideoStatus.READY.value
    assert body["thumbnail_url"] == f"{API}/videos/{video_id}/thumbnail"


@pytest.mark.api
def test_unknown_video_returns_404_envelope(client: TestClient) -> None:
    response = client.get(f"{API}/videos/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "VIDEO_NOT_FOUND"
    assert body["error"]["message"]
    assert body["error"]["request_id"] == response.headers["x-request-id"]


@pytest.mark.media
@pytest.mark.api
def test_import_copy_keeps_source_and_probes_media(
    client: TestClient, settings: Settings, tmp_path: Path, sample_mp4: Path
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(sample_mp4.read_bytes())
    digest = _sha256(source)

    response = _import(client, path=source, copy=True)
    assert response.status_code == 201, response.text
    video = response.json()

    assert video["source_kind"] == SourceKind.IMPORT.value
    assert video["status"] == VideoStatus.READY.value
    assert video["sha256"] == digest
    assert video["width"] == 320
    assert video["height"] == 240

    stored = Path(video["source_path"])
    assert stored.is_file()
    assert stored.parent == settings.originals_dir
    assert stored != source
    assert _sha256(stored) == digest

    assert source.is_file()
    assert _sha256(source) == digest


@pytest.mark.media
@pytest.mark.api
def test_import_in_place_references_source_without_copy(
    client: TestClient, settings: Settings, tmp_path: Path, sample_mp4: Path
) -> None:
    source = tmp_path / "in-place.mp4"
    source.write_bytes(sample_mp4.read_bytes())
    digest = _sha256(source)
    before = set(settings.originals_dir.iterdir())

    response = _import(client, path=source, copy=False)
    assert response.status_code == 201, response.text
    video = response.json()

    assert video["source_kind"] == SourceKind.IMPORT_IN_PLACE.value
    assert video["status"] == VideoStatus.READY.value
    assert video["source_path"] == str(source.resolve())
    assert video["sha256"] is None

    assert set(settings.originals_dir.iterdir()) == before
    assert source.is_file()
    assert _sha256(source) == digest


@pytest.mark.api
def test_import_missing_path_returns_404(client: TestClient, tmp_path: Path) -> None:
    response = _import(client, path=tmp_path / "missing.mp4", copy=True)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SOURCE_NOT_FOUND"


@pytest.mark.api
def test_import_unsupported_extension_returns_415(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not a video", encoding="utf-8")

    response = _import(client, path=source, copy=True)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


@pytest.mark.api
def test_import_inside_data_dir_returns_422(
    client: TestClient, settings: Settings
) -> None:
    settings.ensure_dirs()
    inside = settings.originals_dir / "inside.mp4"
    inside.write_bytes(b"internal data")

    response = _import(client, path=inside, copy=True)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.api
def test_import_outside_allowed_roots_returns_403(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    monkeypatch.setattr(settings, "import_roots", [allowed])
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"not really a video")

    response = _import(client, path=source, copy=True)
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["allowed_roots"] == [str(allowed.resolve())]
    assert error["details"]["path"] == str(source.resolve())


@pytest.mark.api
def test_empty_import_roots_deny_server_side_imports(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "import_roots", [])
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"not really a video")

    response = _import(client, path=source, copy=True)
    assert response.status_code == 403
    assert response.json()["error"]["details"]["allowed_roots"] == []


@pytest.mark.api
def test_import_copy_enospc_returns_507(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"not really a video")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", _fail)

    response = _import(client, path=source, copy=True)
    assert response.status_code == 507
    error = response.json()["error"]
    assert error["code"] == "INSUFFICIENT_DISK"
    assert error["details"]["reason"] == "disk_full"
    # The half-written copy is removed; the source is untouched.
    assert source.is_file()
    assert list(settings.originals_dir.iterdir()) == []


@pytest.mark.media
@pytest.mark.api
def test_thumbnail_is_jpeg_and_cached(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
    sample_mp4: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "thumb-source.mp4"
    source.write_bytes(sample_mp4.read_bytes())
    imported = _import(client, path=source, copy=True)
    assert imported.status_code == 201, imported.text
    video_id = imported.json()["id"]

    first = client.get(f"{API}/videos/{video_id}/thumbnail")
    assert first.status_code == 200, first.text
    assert first.headers["content-type"] == "image/jpeg"
    assert first.content.startswith(JPEG_SOI)
    assert len(first.content) > 3

    thumbnail_file = settings.thumbs_dir / f"{video_id}.jpg"
    assert thumbnail_file.is_file()
    stat_before = thumbnail_file.stat()

    def _forbidden(_settings: Settings | None = None) -> Path:
        raise AssertionError("thumbnail should have been served from cache")

    monkeypatch.setattr(video_service, "require_ffmpeg", _forbidden)

    second = client.get(f"{API}/videos/{video_id}/thumbnail")
    assert second.status_code == 200, second.text
    assert second.content == first.content
    stat_after = thumbnail_file.stat()
    assert stat_after.st_size == stat_before.st_size
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns


@pytest.mark.api
def test_thumbnail_for_failed_video_returns_409(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(
        session_factory, title="broken", created_at=20, status=VideoStatus.FAILED
    )

    response = client.get(f"{API}/videos/{video_id}/thumbnail")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE_TRANSITION"
    assert response.json()["error"]["details"]["status"] == VideoStatus.FAILED.value
