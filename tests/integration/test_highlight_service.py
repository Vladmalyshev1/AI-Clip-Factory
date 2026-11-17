"""``highlight_service``: analysis lifecycle and candidate persistence (M4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode
from app.core.status import HighlightStatus, JobStatus, SourceKind, VideoStatus
from app.models.highlight import Highlight, HighlightCandidate
from app.models.job import Job, new_id
from app.models.video import Video
from app.services import highlight_service, job_service
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db


@dataclass(frozen=True)
class Cand:
    """Minimal candidate accepted by ``replace_candidates``."""

    start_ms: int
    end_ms: int
    score: float
    reason: str = "emotional reaction"
    transcript_text: str = "wow no way"
    metadata: dict[str, Any] = field(default_factory=dict)


def _insert_video(
    session_factory: sessionmaker[Session], *, video_id: str | None = None
) -> str:
    """Insert one probed source recording and return its id."""
    identifier = video_id or new_id()
    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Video(
                id=identifier,
                title="VOD",
                original_filename="vod.mp4",
                source_path="C:/media/vod.mp4",
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=1024,
                duration_ms=600_000,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return identifier


def _analysis(session_factory: sessionmaker[Session], video_id: str) -> Highlight:
    """Create the analysis row of ``video_id`` and return a detached copy."""
    with session_factory() as session:
        analysis = highlight_service.get_or_create_analysis(
            session, video_id, options={"min_score": 0.35}
        )
        session.commit()
        return analysis


def test_get_or_create_is_idempotent(session_factory: sessionmaker[Session]) -> None:
    video_id = _insert_video(session_factory)
    first = _analysis(session_factory, video_id)
    second = _analysis(session_factory, video_id)
    assert first.id == second.id
    with session_factory() as session:
        total = session.scalar(select(func.count()).select_from(Highlight))
    assert total == 1


def test_get_or_create_requires_an_existing_video(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session, pytest.raises(AppError) as excinfo:
        highlight_service.get_or_create_analysis(session, "missing-video")
    assert excinfo.value.code is ErrorCode.VIDEO_NOT_FOUND


def test_analysis_lifecycle_and_completion_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        started = highlight_service.start_analysis(session, row)
        assert started.status == HighlightStatus.RUNNING.value

        completed = highlight_service.complete_analysis(session, analysis.id)
        again = highlight_service.complete_analysis(session, analysis.id)
        session.commit()
    assert completed.status == HighlightStatus.COMPLETED.value
    assert again.status == HighlightStatus.COMPLETED.value


def test_starting_a_completed_analysis_is_a_conflict(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        highlight_service.complete_analysis(session, analysis.id)
        session.commit()

    with session_factory() as session, pytest.raises(AppError) as excinfo:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
    assert excinfo.value.code is ErrorCode.HIGHLIGHT_CONFLICT
    assert excinfo.value.status_code == 409


def test_failed_analysis_can_be_resumed(session_factory: sessionmaker[Session]) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        highlight_service.fail_analysis(
            session, analysis.id, error_code="HIGHLIGHT_FAILED", error_message="boom"
        )
        session.commit()

    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        resumed = highlight_service.start_analysis(session, row)
        assert resumed.status == HighlightStatus.RUNNING.value
        assert resumed.error_code is None
        session.commit()


def test_pause_returns_to_pending_and_rejects_completed(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        paused = highlight_service.pause_analysis(session, analysis.id)
        assert paused.status == HighlightStatus.PENDING.value
        session.commit()

    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        highlight_service.complete_analysis(session, analysis.id)
        session.commit()

    with session_factory() as session, pytest.raises(AppError) as excinfo:
        highlight_service.pause_analysis(session, analysis.id)
    assert excinfo.value.code is ErrorCode.HIGHLIGHT_CONFLICT


def test_replace_candidates_never_duplicates(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    first = [
        Cand(start_ms=0, end_ms=20_000, score=0.9),
        Cand(start_ms=40_000, end_ms=60_000, score=0.5),
    ]
    with session_factory() as session:
        stored = highlight_service.replace_candidates(
            session, analysis.id, video_id, first
        )
        session.commit()
    assert stored == 2

    with session_factory() as session:
        stored = highlight_service.replace_candidates(
            session, analysis.id, video_id, [Cand(start_ms=10_000, end_ms=30_000, score=0.7)]
        )
        session.commit()
    assert stored == 1

    with session_factory() as session:
        items, total = highlight_service.list_candidates(session, analysis.id)
        assert total == 1
        assert items[0].score == pytest.approx(0.7)


def test_candidates_are_ordered_by_score_and_paged(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    candidates = [
        Cand(start_ms=0, end_ms=20_000, score=0.4),
        Cand(start_ms=30_000, end_ms=50_000, score=0.9),
        Cand(start_ms=60_000, end_ms=80_000, score=0.6),
    ]
    with session_factory() as session:
        highlight_service.replace_candidates(session, analysis.id, video_id, candidates)
        session.commit()

    with session_factory() as session:
        first_page, total = highlight_service.list_candidates(
            session, analysis.id, limit=2, offset=0
        )
        second_page, _ = highlight_service.list_candidates(
            session, analysis.id, limit=2, offset=2
        )
    assert total == 3
    assert [item.score for item in first_page] == [pytest.approx(0.9), pytest.approx(0.6)]
    assert [item.score for item in second_page] == [pytest.approx(0.4)]


def test_reset_with_purge_drops_candidates_and_status(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        highlight_service.replace_candidates(
            session, analysis.id, video_id, [Cand(0, 20_000, 0.9)]
        )
        highlight_service.complete_analysis(session, analysis.id)
        session.commit()

    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.reset_analysis(session, row.id, purge=True)
        session.commit()

    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        assert row.status == HighlightStatus.PENDING.value
        assert row.finished_at is None
        total = session.scalar(select(func.count()).select_from(HighlightCandidate))
        assert total == 0


def test_reset_without_purge_keeps_candidates(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.start_analysis(session, row)
        highlight_service.replace_candidates(
            session, analysis.id, video_id, [Cand(0, 20_000, 0.9)]
        )
        highlight_service.complete_analysis(session, analysis.id)
        row = highlight_service.get_analysis_by_id(session, analysis.id)
        highlight_service.reset_analysis(session, row.id, purge=False)
        session.commit()

    with session_factory() as session:
        stats = highlight_service.candidate_stats(session, analysis.id)
    assert stats.total == 1


def test_get_candidate_rejects_unknown_or_foreign_ids(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    other_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        highlight_service.replace_candidates(
            session, analysis.id, video_id, [Cand(0, 20_000, 0.9)]
        )
        session.commit()
        items, _ = highlight_service.list_candidates(session, analysis.id)
        candidate_id = items[0].id

    with session_factory() as session:
        found = highlight_service.get_candidate(session, video_id, candidate_id)
        assert found.id == candidate_id

    for wrong_video, wrong_candidate in (
        (video_id, "missing"),
        (other_id, candidate_id),
    ):
        with session_factory() as session, pytest.raises(AppError) as excinfo:
            highlight_service.get_candidate(session, wrong_video, wrong_candidate)
        assert excinfo.value.code is ErrorCode.HIGHLIGHT_NOT_FOUND


def test_active_and_latest_highlight_jobs(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    with session_factory() as session:
        job = job_service.enqueue(
            session, type=highlight_service.HIGHLIGHT_JOB_TYPE, video_id=video_id
        )
        session.commit()
        job_id = job.id

    with session_factory() as session:
        active = highlight_service.find_active_highlight_job(session, video_id)
        latest = highlight_service.latest_highlight_job(session, video_id)
    assert active is not None and active.id == job_id
    assert latest is not None and latest.id == job_id

    with session_factory() as session:
        row = session.get(Job, job_id)
        assert row is not None
        row.status = JobStatus.SUCCEEDED.value
        session.commit()

    with session_factory() as session:
        assert highlight_service.find_active_highlight_job(session, video_id) is None
        latest = highlight_service.latest_highlight_job(session, video_id)
    assert latest is not None and latest.id == job_id


def test_candidate_stats_reports_the_best_score(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _insert_video(session_factory)
    analysis = _analysis(session_factory, video_id)
    with session_factory() as session:
        highlight_service.replace_candidates(
            session,
            analysis.id,
            video_id,
            [Cand(0, 20_000, 0.4), Cand(30_000, 50_000, 0.8)],
        )
        session.commit()
        stats = highlight_service.candidate_stats(session, analysis.id)
    assert stats.total == 2
    assert stats.best_score == pytest.approx(0.8)
