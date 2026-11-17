"""Clip API: generation, listing, lifecycle actions and file download (M6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import ClipStatus, JobStatus, SourceKind, VideoStatus
from app.main import create_app
from app.models.clip import Clip
from app.models.highlight import HighlightCandidate
from app.models.job import Job, new_id
from app.models.video import Video
from app.services import clip_service, highlight_service
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.api

API = "/api/v1"
DURATION_MS = 600_000


@dataclass(frozen=True)
class Candidate:
    """Minimal candidate accepted by ``highlight_service.replace_candidates``."""

    start_ms: int
    end_ms: int
    score: float
    reason: str = "excited"
    transcript_text: str = "clip this"
    metadata: dict[str, object] | None = None


def _insert_video(
    session_factory: sessionmaker[Session],
    *,
    status: str = VideoStatus.READY.value,
    duration_ms: int = DURATION_MS,
) -> str:
    """Insert one probed source recording and return its id."""
    video_id = new_id()
    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title="VOD",
                original_filename="vod.mp4",
                source_path="C:/media/vod.mp4",
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=1024,
                duration_ms=duration_ms,
                has_audio=1,
                status=status,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def _seed_candidates(
    session_factory: sessionmaker[Session],
    video_id: str,
    candidates: list[Candidate],
    *,
    completed: bool = True,
) -> list[str]:
    """Create the video's analysis with ``candidates``; return ids by score desc."""
    with session_factory() as session:
        analysis = highlight_service.get_or_create_analysis(session, video_id)
        highlight_service.start_analysis(session, analysis)
        highlight_service.replace_candidates(
            session, analysis.id, video_id, candidates
        )
        if completed:
            highlight_service.complete_analysis(session, analysis.id)
        session.commit()
    with session_factory() as session:
        return [
            row.id
            for row in session.scalars(
                select(HighlightCandidate)
                .where(HighlightCandidate.video_id == video_id)
                .order_by(HighlightCandidate.score.desc())
            )
        ]


def _generate(
    client: TestClient, video_id: str, payload: dict[str, object] | None = None
) -> Response:
    return client.post(f"{API}/videos/{video_id}/clips/generate", json=payload or {})


def _clip(session_factory: sessionmaker[Session], clip_id: str) -> Clip:
    with session_factory() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        return clip


def _publish(
    session_factory: sessionmaker[Session],
    settings: Settings,
    clip_id: str,
    *,
    status: str = ClipStatus.COMPLETED.value,
) -> Path:
    """Store ``output_path`` on a clip and write the file under ``clips_dir``."""
    clip = _clip(session_factory, clip_id)
    output = settings.clips_dir / clip.video_id / f"{clip.id}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"fake mp4 body")
    with session_factory() as session:
        stored = session.get(Clip, clip_id)
        assert stored is not None
        stored.status = status
        stored.output_path = str(output)
        session.commit()
    return output


def test_generate_queues_one_clip_per_candidate(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(125_000, 145_000, 0.91)]
    )[0]

    response = _generate(client, video_id, {"candidate_ids": [candidate_id]})

    assert response.status_code == 202
    body = response.json()
    assert len(body["clips"]) == 1
    clip = body["clips"][0]
    assert clip["status"] == ClipStatus.PENDING.value
    assert clip["highlight_candidate_id"] == candidate_id
    assert (clip["start_ms"], clip["end_ms"]) == (120_000, 155_000)
    assert clip["score"] == pytest.approx(0.91)
    assert clip["job_id"]

    with session_factory() as session:
        job = session.get(Job, clip["job_id"])
        assert job is not None and job.type == clip_service.CLIP_JOB_TYPE
        assert job.status == JobStatus.QUEUED.value


def test_generate_is_idempotent_for_the_same_candidate(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]

    first = _generate(client, video_id, {"candidate_ids": [candidate_id]}).json()
    second = _generate(client, video_id, {"candidate_ids": [candidate_id]}).json()

    assert first["clips"][0]["id"] == second["clips"][0]["id"]
    with session_factory() as session:
        assert len(session.scalars(select(Clip)).all()) == 1
        jobs = session.scalars(
            select(Job).where(Job.type == clip_service.CLIP_JOB_TYPE)
        ).all()
        assert len(jobs) == 1


def test_generate_leaves_a_completed_clip_untouched(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip_id = _generate(
        client, video_id, {"candidate_ids": [candidate_id]}
    ).json()["clips"][0]["id"]
    _publish(session_factory, _settings_of(client), clip_id)

    response = _generate(client, video_id, {"candidate_ids": [candidate_id]})

    assert response.status_code == 202
    clips = response.json()["clips"]
    assert clips[0]["id"] == clip_id
    assert clips[0]["status"] == ClipStatus.COMPLETED.value


def test_generate_by_min_score_selects_the_best_candidates(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.95), Candidate(100_000, 110_000, 0.2)],
    )

    response = _generate(client, video_id, {"min_score": 0.5})

    assert response.status_code == 202
    clips = response.json()["clips"]
    assert len(clips) == 1
    assert clips[0]["score"] == pytest.approx(0.95)


def test_generate_without_selection_is_rejected(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])

    response = _generate(client, video_id, {"candidate_ids": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_generate_requires_a_completed_analysis(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)], completed=False
    )[0]

    response = _generate(client, video_id, {"candidate_ids": [candidate_id]})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "HIGHLIGHT_INCOMPLETE"


def test_generate_requires_an_analysis(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)

    response = _generate(client, video_id, {"candidate_ids": ["missing"]})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HIGHLIGHT_NOT_FOUND"


def test_generate_rejects_unknown_and_unready_videos(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    missing = _generate(client, "missing", {"candidate_ids": ["x"]})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "VIDEO_NOT_FOUND"

    pending_id = _insert_video(session_factory, status=VideoStatus.PENDING.value)
    unready = _generate(client, pending_id, {"candidate_ids": ["x"]})
    assert unready.status_code == 409
    assert unready.json()["error"]["code"] == "CLIP_SOURCE_UNAVAILABLE"


def test_generate_rejects_impossible_bounds_and_bad_resolution(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    # 130 s of interest exceeds the 120 s maximum clip duration.
    long_id = _seed_candidates(
        session_factory, video_id, [Candidate(0, 130_000, 0.5)]
    )[0]

    bounds = _generate(client, video_id, {"candidate_ids": [long_id]})
    assert bounds.status_code == 422
    assert bounds.json()["error"]["code"] == "CLIP_BOUNDS_INVALID"

    resolution = _generate(client, video_id, {"resolution": "1279x1920"})
    assert resolution.status_code == 422
    assert resolution.json()["error"]["code"] == "VALIDATION_ERROR"

    square = _generate(client, video_id, {"resolution": "1080x1080"})
    assert square.status_code == 422
    assert square.json()["error"]["code"] == "CLIP_BOUNDS_INVALID"


def test_generate_rejects_mode_combinations_that_cannot_be_rendered(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Rejecting here keeps a doomed render from becoming a failed clip."""
    video_id = _insert_video(session_factory)
    _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])

    fast_vertical = _generate(
        client, video_id, {"min_score": 0.1, "encoding_mode": "fast"}
    )
    assert fast_vertical.status_code == 422
    assert fast_vertical.json()["error"]["code"] == "CLIP_BOUNDS_INVALID"

    target_without_mode = _generate(client, video_id, {"min_score": 0.1, "target_duration_ms": 20_000})
    assert target_without_mode.status_code == 422
    assert target_without_mode.json()["error"]["code"] == "VALIDATION_ERROR"

    with session_factory() as session:
        assert session.scalars(select(Clip)).all() == []


def test_generate_enforces_the_active_job_cap(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _insert_video(session_factory)
    _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    capped = settings.model_copy(update={"clip_max_concurrent_jobs": 1})

    with TestClient(create_app(capped)) as client:
        response = _generate(client, video_id, {"min_score": 0.5})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CLIP_CONFLICT"


def test_list_pages_and_filters_the_clips_of_a_video(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    _generate(client, video_id, {"candidate_ids": ids})
    clip_id = _clip_ids(session_factory, video_id)[0]
    _publish(session_factory, _settings_of(client), clip_id)

    response = client.get(f"{API}/videos/{video_id}/clips", params={"limit": 1})

    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 2
    assert page["limit"] == 1 and page["offset"] == 0
    assert len(page["items"]) == 1

    completed = client.get(
        f"{API}/videos/{video_id}/clips", params={"status": "completed"}
    ).json()
    assert completed["total"] == 1
    assert completed["items"][0]["id"] == clip_id

    unknown = client.get(f"{API}/videos/missing/clips")
    assert unknown.status_code == 404


def test_stats_count_the_clips_of_a_video(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    _generate(client, video_id, {"candidate_ids": ids})

    response = client.get(f"{API}/videos/{video_id}/clips/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "pending": 2,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "canceled": 0,
    }


def test_get_clip_reports_404_for_unknown_ids(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip_id = _generate(
        client, video_id, {"candidate_ids": [candidate_id]}
    ).json()["clips"][0]["id"]

    found = client.get(f"{API}/clips/{clip_id}")
    assert found.status_code == 200
    assert found.json()["id"] == clip_id

    missing = client.get(f"{API}/clips/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "CLIP_NOT_FOUND"


def test_retry_requeues_a_failed_clip_and_refuses_a_completed_one(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip_id = _generate(
        client, video_id, {"candidate_ids": [candidate_id]}
    ).json()["clips"][0]["id"]
    old_job_id = _clip(session_factory, clip_id).job_id

    with session_factory() as session:
        stored = session.get(Clip, clip_id)
        assert stored is not None
        stored.status = ClipStatus.FAILED.value
        stored.error_code = "RENDER_FAILED"
        job = session.get(Job, stored.job_id) if stored.job_id else None
        if job is not None:
            job.status = JobStatus.FAILED.value
        session.commit()

    response = client.post(f"{API}/clips/{clip_id}/retry")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == ClipStatus.PENDING.value
    assert body["error"] is None
    assert body["job_id"] != old_job_id

    _publish(session_factory, _settings_of(client), clip_id)
    completed = client.post(f"{API}/clips/{clip_id}/retry")
    assert completed.status_code == 409
    assert completed.json()["error"]["code"] == "CLIP_CONFLICT"


def test_retrying_the_underlying_clip_job_is_refused(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A stale clip job must never race the clip's current job for the same file."""
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip = _generate(client, video_id, {"candidate_ids": [candidate_id]}).json()["clips"][0]
    with session_factory() as session:
        job = session.get(Job, clip["job_id"])
        assert job is not None
        job.status = JobStatus.FAILED.value
        session.commit()

    response = client.post(f"{API}/jobs/{clip['job_id']}/retry", json={})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CLIP_CONFLICT"


def test_cancel_cancels_a_pending_clip_and_refuses_a_completed_one(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    clips = _generate(client, video_id, {"candidate_ids": ids}).json()["clips"]
    pending_id, other_id = clips[0]["id"], clips[1]["id"]

    response = client.post(f"{API}/clips/{pending_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == ClipStatus.CANCELED.value
    with session_factory() as session:
        job = session.get(Job, clips[0]["job_id"])
        assert job is not None and job.status == JobStatus.CANCELED.value

    _publish(session_factory, _settings_of(client), other_id)
    refused = client.post(f"{API}/clips/{other_id}/cancel")
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "CLIP_CONFLICT"


def test_delete_removes_the_row_and_the_file(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip_id = _generate(
        client, video_id, {"candidate_ids": [candidate_id]}
    ).json()["clips"][0]["id"]
    output = _publish(session_factory, _settings_of(client), clip_id)

    response = client.delete(f"{API}/clips/{clip_id}")

    assert response.status_code == 204
    assert not output.exists()
    with session_factory() as session:
        assert session.get(Clip, clip_id) is None


def test_delete_refuses_a_processing_clip(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)]
    )[0]
    clip_id = _generate(
        client, video_id, {"candidate_ids": [candidate_id]}
    ).json()["clips"][0]["id"]
    _publish(
        session_factory, _settings_of(client), clip_id, status=ClipStatus.PROCESSING.value
    )

    response = client.delete(f"{API}/clips/{clip_id}")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CLIP_CONFLICT"
    with session_factory() as session:
        assert session.get(Clip, clip_id) is not None


def test_download_serves_only_published_files_inside_the_clip_directory(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    video_id = _insert_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    clips = _generate(client, video_id, {"candidate_ids": ids}).json()["clips"]
    ready_id, unpublished_id = clips[0]["id"], clips[1]["id"]

    not_ready = client.get(f"{API}/clips/{unpublished_id}/file")
    assert not_ready.status_code == 404
    assert not_ready.json()["error"]["code"] == "CLIP_NOT_FOUND"

    _publish(session_factory, _settings_of(client), ready_id)
    download = client.get(f"{API}/clips/{ready_id}/file")
    assert download.status_code == 200
    assert download.content == b"fake mp4 body"
    assert download.headers["content-disposition"].endswith('.mp4"')

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"user data")
    with session_factory() as session:
        stored = session.get(Clip, ready_id)
        assert stored is not None
        stored.output_path = str(outside)
        session.commit()

    escaped = client.get(f"{API}/clips/{ready_id}/file")
    assert escaped.status_code == 500
    assert escaped.json()["error"]["code"] == "CLIP_FAILED"
    assert outside.exists()


def _clip_ids(session_factory: sessionmaker[Session], video_id: str) -> list[str]:
    """Return the clip ids of a video, oldest first."""
    with session_factory() as session:
        return list(
            session.scalars(
                select(Clip.id)
                .where(Clip.video_id == video_id)
                .order_by(Clip.created_at, Clip.id)
            )
        )


def _settings_of(client: TestClient) -> Settings:
    """Return the settings the test client runs with."""
    settings = client.app.state.settings
    assert isinstance(settings, Settings)
    return settings
