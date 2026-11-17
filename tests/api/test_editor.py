"""API of the clip editor, exports and source streaming (M7)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.core.clock import utc_now_ms
from app.core.status import SourceKind, VideoStatus
from app.models.clip import Clip
from app.models.job import new_id
from app.models.video import Video
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

pytestmark = [pytest.mark.api, pytest.mark.db]

NOW = utc_now_ms()


@pytest.fixture
def video_id(session_factory: sessionmaker[Session]) -> str:
    """One ready video with a real (tiny) source file on disk."""
    vid = new_id()
    source = Path(os.environ["ACF_SOURCE_FILE"])
    with session_factory() as session:
        session.add(
            Video(
                id=vid,
                title="VOD",
                original_filename="vod.mp4",
                source_path=str(source),
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=source.stat().st_size,
                duration_ms=600_000,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()
    return vid


@pytest.fixture
def clip_id(session_factory: sessionmaker[Session], video_id: str) -> str:
    """One completed auto clip with a published file."""
    cid = new_id()
    with session_factory() as session:
        session.add(
            Clip(
                id=cid,
                video_id=video_id,
                kind="auto",
                status="completed",
                title="auto clip",
                start_ms=0,
                end_ms=4_000,
                duration_ms=4_000,
                output_format="vertical",
                crop_mode="pad",
                encoding_mode="accurate",
                width=1080,
                height=1920,
                output_path=os.environ["ACF_CLIP_FILE"],
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()
    return cid


@pytest.fixture(autouse=True)
def _media_files(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Two tiny distinct MP4-ish files: the VOD source and a published clip."""
    source = tmp_path_factory.mktemp("media") / "vod.mp4"
    source.write_bytes(b"\x00" * 4096)
    clip = tmp_path_factory.mktemp("clips") / "clip.mp4"
    clip.write_bytes(b"\x11" * 2048)
    os.environ["ACF_SOURCE_FILE"] = str(source)
    os.environ["ACF_CLIP_FILE"] = str(clip)
    yield


# ---------------------------------------------------------------------- editor


def test_get_editor_returns_clip_defaults(client: TestClient, clip_id: str) -> None:
    response = client.get(f"/api/v1/clips/{clip_id}/editor")

    assert response.status_code == 200
    body = response.json()
    assert body["draft"]["start_ms"] == 0
    assert body["draft"]["end_ms"] == 4_000
    assert body["draft"]["subtitles"]["enabled"] is False
    assert body["draft"]["subtitles"]["cues"] == []
    assert body["transcript_available"] is False
    assert body["suggested_cues"] == []


def test_put_editor_saves_the_draft(client: TestClient, clip_id: str) -> None:
    payload = {
        "start_ms": 500,
        "end_ms": 3_500,
        "subtitles": {
            "enabled": True,
            "style": "bold",
            "font_size": 72,
            "primary_colour": "#123ABC",
            "cues": [{"start_ms": 0, "end_ms": 1_500, "text": "hello clip"}],
        },
    }
    saved = client.put(f"/api/v1/clips/{clip_id}/editor", json=payload)

    assert saved.status_code == 200
    body = saved.json()
    assert body["start_ms"] == 500
    assert body["subtitles"]["enabled"] is True
    assert body["subtitles"]["primary_colour"] == "#123ABC"

    reread = client.get(f"/api/v1/clips/{clip_id}/editor")
    assert reread.json()["draft"]["subtitles"]["cues"][0]["text"] == "hello clip"

    # The clip row itself is untouched.
    clip = client.get(f"/api/v1/clips/{clip_id}").json()
    assert clip["end_ms"] == 4_000


def test_put_editor_rejects_bad_bounds(client: TestClient, clip_id: str) -> None:
    response = client.put(
        f"/api/v1/clips/{clip_id}/editor", json={"start_ms": 3_000, "end_ms": 1_000}
    )
    assert response.status_code == 422


def test_put_editor_rejects_bad_colour(client: TestClient, clip_id: str) -> None:
    response = client.put(
        f"/api/v1/clips/{clip_id}/editor",
        json={"subtitles": {"enabled": True, "primary_colour": "reddish", "cues": [
            {"start_ms": 0, "end_ms": 500, "text": "x"}
        ]}},
    )
    assert response.status_code == 422


def test_patch_clip_renames(client: TestClient, clip_id: str) -> None:
    response = client.patch(f"/api/v1/clips/{clip_id}", json={"title": "  Best moment  "})
    assert response.status_code == 200
    assert response.json()["title"] == "Best moment"

    cleared = client.patch(f"/api/v1/clips/{clip_id}", json={"title": "   "})
    assert cleared.json()["title"] is None


def test_clip_transcript_endpoint(client: TestClient, clip_id: str) -> None:
    response = client.get(f"/api/v1/clips/{clip_id}/transcript")
    assert response.status_code == 200
    body = response.json()
    assert body == {"transcript_available": False, "cues": []}


# --------------------------------------------------------------------- exports


def test_export_roundtrip(client: TestClient, clip_id: str, session_factory) -> None:
    created = client.post(f"/api/v1/clips/{clip_id}/exports", json={})
    assert created.status_code == 201
    body = created.json()
    assert body["total"] == 1
    export = body["items"][0]
    assert export["kind"] == "export"
    assert export["parent_clip_id"] == clip_id
    assert export["status"] == "pending"

    # Idempotent while active.
    again = client.post(f"/api/v1/clips/{clip_id}/exports", json={})
    assert again.json()["items"][0]["id"] == export["id"]

    listing = client.get(f"/api/v1/clips/{clip_id}/exports")
    assert listing.json()["total"] == 1

    single = client.get(f"/api/v1/exports/{export['id']}")
    assert single.json()["id"] == export["id"]


def test_export_with_body_uses_given_bounds(client: TestClient, clip_id: str) -> None:
    response = client.post(
        f"/api/v1/clips/{clip_id}/exports",
        json={"start_ms": 1_000, "end_ms": 3_000},
    )
    assert response.status_code == 201
    export = response.json()["items"][0]
    assert export["start_ms"] == 1_000
    assert export["end_ms"] == 3_000


def test_export_of_unknown_clip_404(client: TestClient) -> None:
    assert client.post("/api/v1/clips/missing/exports", json={}).status_code == 404


def test_export_file_requires_completion(
    client: TestClient, clip_id: str, session_factory
) -> None:
    export = client.post(f"/api/v1/clips/{clip_id}/exports", json={}).json()["items"][0]
    response = client.get(f"/api/v1/exports/{export['id']}/file")
    assert response.status_code == 404


def test_delete_export(client: TestClient, clip_id: str, session_factory) -> None:
    export = client.post(f"/api/v1/clips/{clip_id}/exports", json={}).json()["items"][0]
    export_id = export["id"]

    # Pending: must be cancelled first.
    assert client.delete(f"/api/v1/exports/{export_id}").status_code == 409

    cancelled = client.post(f"/api/v1/exports/{export_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "canceled"

    retried = client.post(f"/api/v1/exports/{export_id}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "pending"

    cancelled = client.post(f"/api/v1/exports/{export_id}/cancel")
    assert cancelled.status_code == 200

    deleted = client.delete(f"/api/v1/exports/{export_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/exports/{export_id}").status_code == 404


# ---------------------------------------------------------------------- source


def test_source_without_range_returns_200(client: TestClient, video_id: str) -> None:
    response = client.get(f"/api/v1/videos/{video_id}/source")
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"
    assert len(response.content) == 4096


def test_source_with_range_returns_206(client: TestClient, video_id: str) -> None:
    response = client.get(
        f"/api/v1/videos/{video_id}/source", headers={"Range": "bytes=100-199"}
    )
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 100-199/4096"
    assert len(response.content) == 100


def test_source_suffix_range(client: TestClient, video_id: str) -> None:
    response = client.get(f"/api/v1/videos/{video_id}/source", headers={"Range": "bytes=-100"})
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 3996-4095/4096"
    assert len(response.content) == 100


def test_source_open_ended_range(client: TestClient, video_id: str) -> None:
    response = client.get(f"/api/v1/videos/{video_id}/source", headers={"Range": "bytes=4000-"})
    assert response.status_code == 206
    assert len(response.content) == 96


def test_source_unsatisfiable_range_returns_416(client: TestClient, video_id: str) -> None:
    response = client.get(
        f"/api/v1/videos/{video_id}/source", headers={"Range": "bytes=99999-"}
    )
    assert response.status_code == 416


def test_source_of_missing_video_404(client: TestClient) -> None:
    assert client.get("/api/v1/videos/missing/source").status_code == 404


def test_source_of_missing_file_404(
    client: TestClient, video_id: str, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        video = session.get(Video, video_id)
        assert video is not None
        video.source_path = "C:/definitely/missing/vod.mp4"
        session.commit()
    assert client.get(f"/api/v1/videos/{video_id}/source").status_code == 404

