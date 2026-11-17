"""Highlight API: status, candidate paging and queueing a detection run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from app.ai.asr import AsrSegment
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import JobStatus, SourceKind, VideoStatus
from app.models.job import Job, new_id
from app.models.video import Video
from app.pipeline.stages.base import StageContext
from app.pipeline.stages.highlights import HighlightStage
from app.services import highlight_service, transcript_service
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.api

API = "/api/v1"
DURATION_MS = 120_000


@dataclass(frozen=True)
class Plan:
    """Minimal chunk plan accepted by ``transcript_service.ensure_chunks``."""

    index: int
    start_ms: int
    end_ms: int


def _insert_video(session_factory: sessionmaker[Session]) -> str:
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
                duration_ms=DURATION_MS,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def _seed_transcript(
    session_factory: sessionmaker[Session], video_id: str, *, completed: bool = True
) -> None:
    """Create a transcript with one completed chunk of excited segments."""
    with session_factory() as session:
        transcript = transcript_service.get_or_create_transcript(
            session,
            video_id,
            language="en",
            model_name="fake-whisper",
            duration_ms=DURATION_MS,
            chunk_seconds=300,
            options={"language": "en"},
        )
        transcript_service.ensure_chunks(
            session, transcript, [Plan(index=0, start_ms=0, end_ms=DURATION_MS)]
        )
        if not completed:
            session.commit()
            return
        transcript_service.start_transcript(session, transcript)
        transcript_service.mark_chunk_running(session, transcript.id, 0)
        transcript_service.save_chunk_result(
            session,
            transcript.id,
            0,
            [
                AsrSegment(
                    start_ms=10_000 + index * 1000,
                    end_ms=10_000 + index * 1000 + 900,
                    text="wow no way insane!",
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                )
                for index in range(20)
            ],
        )
        transcript_service.complete_transcript(
            session, transcript.id, expected_ms=DURATION_MS
        )
        session.commit()


def _run_stage(
    settings: Settings, session_factory: sessionmaker[Session], video_id: str, tmp_path: Path
) -> None:
    """Run the detection stage the way a worker would."""
    ctx = StageContext(
        job_id="job-1",
        video_id=video_id,
        attempt=1,
        work_dir=tmp_path,
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )
    HighlightStage().run(ctx, _RecordingReporter())


class _RecordingReporter:
    def update(self, progress: float, message: str | None = None) -> None:
        return None

    def check_cancelled(self) -> None:
        return None


def _finish_jobs(session_factory: sessionmaker[Session], video_id: str) -> None:
    """Mark every job of ``video_id`` terminal so no active job remains."""
    with session_factory() as session:
        rows = list(session.scalars(select(Job).where(Job.video_id == video_id)).all())
        for row in rows:
            row.status = JobStatus.SUCCEEDED.value
        session.commit()


def test_detect_requires_an_existing_video(client: TestClient) -> None:
    response = client.post(f"{API}/videos/missing/detect-highlights")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_NOT_FOUND"


def test_detect_requires_a_transcript(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    response = client.post(f"{API}/videos/{video_id}/detect-highlights")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRANSCRIPT_NOT_FOUND"


def test_detect_requires_a_completed_transcript(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id, completed=False)
    response = client.post(f"{API}/videos/{video_id}/detect-highlights")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "TRANSCRIPT_INCOMPLETE"


def test_detect_is_accepted_and_reports_the_job(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)

    response = client.post(f"{API}/videos/{video_id}/detect-highlights")
    assert response.status_code == 202
    body = response.json()
    assert body["video_id"] == video_id
    assert body["status"] == "pending"
    assert body["job_id"]
    assert body["candidate_count"] == 0

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert analysis.status == "pending"
        job = highlight_service.latest_highlight_job(session, video_id)
    assert job is not None and job.type == "highlight"


def test_detect_refuses_an_active_job(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)
    assert client.post(f"{API}/videos/{video_id}/detect-highlights").status_code == 202

    response = client.post(f"{API}/videos/{video_id}/detect-highlights")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "HIGHLIGHT_CONFLICT"


def test_status_is_404_before_the_first_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)
    response = client.get(f"{API}/videos/{video_id}/highlights")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HIGHLIGHT_NOT_FOUND"


def test_a_completed_analysis_needs_force_to_rerun(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)
    assert client.post(f"{API}/videos/{video_id}/detect-highlights").status_code == 202
    _run_stage(settings, session_factory, video_id, tmp_path)
    _finish_jobs(session_factory, video_id)

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert highlight_service.candidate_stats(session, analysis.id).total >= 1

    refused = client.post(f"{API}/videos/{video_id}/detect-highlights")
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "HIGHLIGHT_CONFLICT"

    forced = client.post(
        f"{API}/videos/{video_id}/detect-highlights", json={"force": True}
    )
    assert forced.status_code == 202
    assert forced.json()["status"] == "pending"

    # ``force`` purges the candidates of the finished run before re-queueing.
    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        stats = highlight_service.candidate_stats(session, analysis.id)
    assert stats.total == 0


def test_status_lists_candidates_after_a_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)
    assert client.post(f"{API}/videos/{video_id}/detect-highlights").status_code == 202
    _run_stage(settings, session_factory, video_id, tmp_path)

    status = client.get(f"{API}/videos/{video_id}/highlights")
    assert status.status_code == 200
    analysis = status.json()
    assert analysis["status"] == "completed"
    assert analysis["candidate_count"] >= 1
    assert analysis["best_score"] is not None

    page = client.get(f"{API}/videos/{video_id}/highlights/candidates")
    assert page.status_code == 200
    body = page.json()
    assert body["total"] == analysis["candidate_count"]
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    first = body["items"][0]
    assert 0.0 <= first["score"] <= 1.0
    assert first["transcript"]
    assert "wow" in first["transcript"]

    single = client.get(
        f"{API}/videos/{video_id}/highlights/candidates/{first['id']}"
    )
    assert single.status_code == 200
    assert single.json()["id"] == first["id"]

    missing = client.get(f"{API}/videos/{video_id}/highlights/candidates/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "HIGHLIGHT_NOT_FOUND"


def test_candidate_pages_are_limited(
    client: TestClient,
    session_factory: sessionmaker[Session],
    settings: Settings,
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id)
    assert client.post(f"{API}/videos/{video_id}/detect-highlights").status_code == 202
    _run_stage(settings, session_factory, video_id, tmp_path)

    page = client.get(
        f"{API}/videos/{video_id}/highlights/candidates",
        params={"limit": 1, "offset": 0},
    )
    assert page.status_code == 200
    body = page.json()
    assert len(body["items"]) == 1
    assert body["total"] >= 1
