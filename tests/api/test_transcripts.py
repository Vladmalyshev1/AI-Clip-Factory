"""Transcript API: read a transcript, page segments, queue a transcription."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from app.ai.asr import AsrSegment
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import (
    JobStatus,
    SourceKind,
    TranscriptStatus,
    VideoStatus,
)
from app.models.job import Job, new_id
from app.models.transcript import TranscriptChunk
from app.models.video import Video
from app.services import transcript_service
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

API = "/api/v1"
CHUNK_SECONDS = 300
DURATION_MS = 900_000


@dataclass(frozen=True)
class Plan:
    """Minimal chunk plan accepted by ``transcript_service.ensure_chunks``."""

    index: int
    start_ms: int
    end_ms: int


def _plans() -> list[Plan]:
    """Return the three 300 s windows of :data:`DURATION_MS`."""
    return [
        Plan(index=index, start_ms=index * 300_000, end_ms=(index + 1) * 300_000)
        for index in range(3)
    ]


def _insert_video(
    session_factory: sessionmaker[Session],
    *,
    video_id: str | None = None,
    status: VideoStatus = VideoStatus.READY,
    duration_ms: int | None = DURATION_MS,
    has_audio: int = 1,
) -> str:
    """Insert one probed source recording and return its id."""
    identifier = video_id or new_id()
    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Video(
                id=identifier,
                title="Source recording",
                original_filename="vod.mp4",
                source_path="C:/media/vod.mp4",
                source_kind=SourceKind.UPLOAD.value,
                size_bytes=1024,
                duration_ms=duration_ms,
                has_audio=has_audio,
                status=status.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return identifier


def _seed_segments(count: int, *, offset_ms: int = 0) -> list[AsrSegment]:
    """Return ``count`` segments spaced one second apart, shifted by ``offset_ms``."""
    return [
        AsrSegment(
            start_ms=offset_ms + 1000 + index * 1000,
            end_ms=offset_ms + 1800 + index * 1000,
            text=f"segment {index}",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        )
        for index in range(count)
    ]


def _seed_transcript(
    session_factory: sessionmaker[Session],
    video_id: str,
    *,
    completed_chunks: int = 0,
    segment_count: int = 3,
    finish: bool = False,
) -> str:
    """Create a transcript with chunk rows and optional finished chunks."""
    with session_factory() as session:
        transcript = transcript_service.get_or_create_transcript(
            session,
            video_id,
            language="en",
            model_name="fake-whisper",
            duration_ms=DURATION_MS,
            chunk_seconds=CHUNK_SECONDS,
            options={"language": "en"},
        )
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_service.start_transcript(session, transcript)
        for index in range(completed_chunks):
            transcript_service.mark_chunk_running(session, transcript.id, index)
            transcript_service.save_chunk_result(
                session,
                transcript.id,
                index,
                _seed_segments(segment_count, offset_ms=index * 300_000),
            )
        if finish:
            transcript_service.complete_transcript(session, transcript.id)
        transcript_id = transcript.id
        session.commit()
    return transcript_id


def test_get_transcript_returns_404_before_any_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)

    response = client.get(f"{API}/videos/{video_id}/transcript")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "TRANSCRIPT_NOT_FOUND"
    assert body["error"]["details"]["video_id"] == video_id


def test_post_transcribe_queues_a_job(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)

    response = client.post(f"{API}/videos/{video_id}/transcribe", json={})

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["video_id"] == video_id
    assert body["status"] == TranscriptStatus.PENDING.value
    assert body["language"] == "en"
    assert body["segment_count"] == 0
    assert body["progress"]["chunks_total"] == 0
    assert body["job_id"]

    job = client.get(f"{API}/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["type"] == "transcribe"
    assert job.json()["status"] == JobStatus.QUEUED.value
    assert job.json()["video_id"] == video_id

    fetched = client.get(f"{API}/videos/{video_id}/transcript")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]
    assert fetched.json()["job_id"] == body["job_id"]


def test_post_transcribe_without_body_uses_defaults(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)

    response = client.post(f"{API}/videos/{video_id}/transcribe")

    assert response.status_code == 202, response.text
    assert response.json()["status"] == TranscriptStatus.PENDING.value


def test_post_transcribe_refuses_a_second_active_job(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    first = client.post(f"{API}/videos/{video_id}/transcribe", json={})
    assert first.status_code == 202

    second = client.post(f"{API}/videos/{video_id}/transcribe", json={"force": True})

    assert second.status_code == 409
    body = second.json()
    assert body["error"]["code"] == "TRANSCRIPT_CONFLICT"
    assert body["error"]["details"]["job_id"] == first.json()["job_id"]


@pytest.mark.parametrize(
    ("status", "duration_ms", "has_audio", "expected_status", "expected_code"),
    [
        (VideoStatus.PENDING, DURATION_MS, 1, 409, "INVALID_STATE_TRANSITION"),
        (VideoStatus.READY, DURATION_MS, 0, 422, "NO_AUDIO_STREAM"),
        (VideoStatus.READY, None, 1, 422, "PROBE_FAILED"),
    ],
)
def test_post_transcribe_rejects_unusable_videos(
    client: TestClient,
    session_factory: sessionmaker[Session],
    status: VideoStatus,
    duration_ms: int | None,
    has_audio: int,
    expected_status: int,
    expected_code: str,
) -> None:
    video_id = _insert_video(
        session_factory, status=status, duration_ms=duration_ms, has_audio=has_audio
    )

    response = client.post(f"{API}/videos/{video_id}/transcribe", json={})

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    with session_factory() as session:
        assert transcript_service.find_transcript(session, video_id) is None


def test_completed_transcript_requires_force(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    transcript_id = _seed_transcript(
        session_factory, video_id, completed_chunks=3, segment_count=2, finish=True
    )

    refused = client.post(f"{API}/videos/{video_id}/transcribe", json={})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "TRANSCRIPT_CONFLICT"
    assert refused.json()["error"]["details"]["transcript_id"] == transcript_id

    forced = client.post(f"{API}/videos/{video_id}/transcribe", json={"force": True})
    assert forced.status_code == 202, forced.text
    body = forced.json()
    assert body["id"] == transcript_id
    assert body["status"] == TranscriptStatus.PENDING.value
    assert body["segment_count"] == 0
    assert body["progress"]["chunks_total"] == 0
    assert body["job_id"]


def test_forced_transcript_purges_an_interrupted_transcript(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    transcript_id = _seed_transcript(session_factory, video_id, completed_chunks=1)

    forced = client.post(f"{API}/videos/{video_id}/transcribe", json={"force": True})

    assert forced.status_code == 202, forced.text
    body = forced.json()
    assert body["id"] == transcript_id
    assert body["status"] == TranscriptStatus.PENDING.value
    assert body["segment_count"] == 0
    assert body["progress"]["chunks_total"] == 0
    assert body["job_id"]

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        assert transcript.status == TranscriptStatus.PENDING.value
        chunk_count = session.scalar(
            select(func.count())
            .select_from(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript.id)
        )
        assert chunk_count == 0
        assert transcript_service.count_segments(session, transcript_id) == 0


def test_interrupted_transcript_is_resumed_not_purged(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    transcript_id = _seed_transcript(session_factory, video_id, completed_chunks=1)
    with session_factory() as session:
        transcript_service.fail_transcript(
            session, transcript_id, error_code="ASR_FAILED", error_message="decoder died"
        )
        session.commit()

    response = client.post(f"{API}/videos/{video_id}/transcribe", json={})

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["id"] == transcript_id
    assert body["status"] == TranscriptStatus.FAILED.value
    assert body["progress"]["chunks_completed"] == 1
    assert body["progress"]["chunks_total"] == 3
    assert body["segment_count"] == 3


def test_get_transcript_reports_progress_and_segments(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    transcript_id = _seed_transcript(session_factory, video_id, completed_chunks=2)

    response = client.get(f"{API}/videos/{video_id}/transcript")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == transcript_id
    assert body["status"] == TranscriptStatus.RUNNING.value
    assert body["chunk_seconds"] == CHUNK_SECONDS
    assert body["duration_ms"] == DURATION_MS
    assert body["options"] == {"language": "en"}
    assert body["segment_count"] == 6
    assert body["progress"]["chunks_total"] == 3
    assert body["progress"]["chunks_completed"] == 2
    assert body["progress"]["audio_ms_processed"] == 600_000
    assert body["progress"]["progress"] == pytest.approx(2 / 3)
    assert body["error"] is None
    assert body["job_id"] is None


def test_list_segments_paginates(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    transcript_id = _seed_transcript(session_factory, video_id, completed_chunks=1)
    with session_factory() as session:
        transcript_service.mark_chunk_running(session, transcript_id, 1)
        transcript_service.save_chunk_result(
            session, transcript_id, 1, _seed_segments(2, offset_ms=300_000)
        )
        session.commit()

    full = client.get(f"{API}/videos/{video_id}/transcript/segments")
    assert full.status_code == 200
    assert full.json()["total"] == 5
    assert len(full.json()["items"]) == 5

    page = client.get(
        f"{API}/videos/{video_id}/transcript/segments", params={"limit": 2, "offset": 1}
    )
    assert page.status_code == 200
    body = page.json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert len(body["items"]) == 2
    assert body["items"][0]["text"] == "segment 1"
    assert body["items"][0]["chunk_index"] == 0


def test_list_segments_filters_by_overlap(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id, completed_chunks=1)

    window = client.get(
        f"{API}/videos/{video_id}/transcript/segments",
        params={"start_ms": 2000, "end_ms": 3000},
    )

    assert window.status_code == 200
    texts = [item["text"] for item in window.json()["items"]]
    # Segments are [1000, 1800), [2000, 2800), [3000, 3800): overlap includes
    # the middle one only (the third starts exactly at the exclusive end).
    assert texts == ["segment 1"]


def test_list_segments_rejects_inverted_window(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id, completed_chunks=1)

    response = client.get(
        f"{API}/videos/{video_id}/transcript/segments",
        params={"start_ms": 5000, "end_ms": 1000},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_segments_for_video_without_transcript_returns_404(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)

    response = client.get(f"{API}/videos/{video_id}/transcript/segments")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRANSCRIPT_NOT_FOUND"


def test_transcribe_unknown_video_returns_404(client: TestClient) -> None:
    response = client.post(f"{API}/videos/does-not-exist/transcribe", json={})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_NOT_FOUND"


def _jobs_for_video(session_factory: sessionmaker[Session], video_id: str) -> list[Job]:
    """Return every job of a video, newest first."""
    with session_factory() as session:
        return list(
            session.scalars(
                select(Job).where(Job.video_id == video_id).order_by(Job.created_at.desc())
            ).all()
        )


def test_force_requeues_exactly_one_job(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory, video_id, completed_chunks=3, segment_count=1, finish=True
    )

    first = client.post(f"{API}/videos/{video_id}/transcribe", json={"force": True})
    assert first.status_code == 202
    second = client.post(f"{API}/videos/{video_id}/transcribe", json={"force": True})
    assert second.status_code == 409

    jobs = _jobs_for_video(session_factory, video_id)
    assert len(jobs) == 1
    assert jobs[0].type == "transcribe"
    assert jobs[0].status == JobStatus.QUEUED.value


def test_transcript_options_come_from_settings(
    client: TestClient, session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _insert_video(session_factory)

    response = client.post(f"{API}/videos/{video_id}/transcribe", json={})

    assert response.status_code == 202
    options = response.json()["options"]
    assert options["language"] == settings.asr_language
    assert options["device"] == settings.asr_device
    assert options["compute_type"] == settings.asr_compute_type
    assert options["chunk_seconds"] == settings.asr_chunk_seconds
    assert response.json()["model_name"] == settings.asr_model
