"""Clip service: persistence, idempotency, queueing and file bookkeeping (M6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.status import ClipStatus, JobStatus, SourceKind, VideoStatus
from app.models.clip import Clip
from app.models.highlight import HighlightCandidate
from app.models.job import Job, new_id
from app.models.video import Video
from app.services import clip_service, highlight_service, job_service
from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

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


def _seed_video(
    session_factory: sessionmaker[Session],
    *,
    duration_ms: int = DURATION_MS,
    status: str = VideoStatus.READY.value,
) -> str:
    now = utc_now_ms()
    video_id = new_id()
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
    with session_factory() as session:
        analysis = highlight_service.get_or_create_analysis(session, video_id)
        ids = [
            row.id
            for row in session.scalars(
                select(HighlightCandidate).where(HighlightCandidate.video_id == video_id)
            )
        ]
        if ids:
            session.commit()
            return ids
        highlight_service.start_analysis(session, analysis)
        highlight_service.replace_candidates(session, analysis.id, video_id, candidates)
        if completed:
            highlight_service.complete_analysis(session, analysis.id)
        session.commit()
        return [
            row.id
            for row in session.scalars(
                select(HighlightCandidate)
                .where(HighlightCandidate.video_id == video_id)
                .order_by(HighlightCandidate.score.desc())
            )
        ]


def _create(
    session_factory: sessionmaker[Session],
    settings: Settings,
    video_id: str,
    candidates: list[str],
) -> list[Clip]:
    with session_factory() as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        clips = clip_service.create_clips(
            session,
            video_id,
            options=clip_service.ClipOptions.from_settings(settings),
            settings=settings,
            candidate_ids=candidates,
        )
        session.commit()
        return clips


def test_create_clip_from_a_candidate_persists_bounds_score_and_job(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(125_000, 145_000, 0.87)])[0]

    clips = _create(session_factory, settings, video_id, [candidate_id])

    assert len(clips) == 1
    clip = clips[0]
    assert (clip.start_ms, clip.end_ms, clip.duration_ms) == (120_000, 155_000, 35_000)
    assert clip.score == pytest.approx(0.87)
    assert clip.highlight_candidate_id == candidate_id
    assert clip.status == ClipStatus.PENDING.value
    assert clip.encoding_mode == "accurate"
    assert clip.output_format == "vertical"
    assert clip.crop_mode == "pad"
    with session_factory() as session:
        job = session.get(Job, clip.job_id)
        assert job is not None
        assert job.type == clip_service.CLIP_JOB_TYPE
        assert job.status == JobStatus.QUEUED.value
        assert job_service.load_json_dict(job.params_json) == {"clip_id": clip.id}


def test_repeated_generate_is_idempotent(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]

    first = _create(session_factory, settings, video_id, [candidate_id])
    second = _create(session_factory, settings, video_id, [candidate_id])

    assert [clip.id for clip in first] == [clip.id for clip in second]
    with session_factory() as session:
        assert len(session.scalars(select(Clip)).all()) == 1
        assert len(session.scalars(select(Job).where(Job.type == clip_service.CLIP_JOB_TYPE)).all()) == 1


def test_unique_candidate_index_blocks_a_duplicate_row(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    _create(session_factory, settings, video_id, [candidate_id])

    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Clip(
                video_id=video_id,
                highlight_candidate_id=candidate_id,
                status=ClipStatus.PENDING.value,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                created_at=now,
                updated_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_failed_clip_is_rearmed_with_a_new_job(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]
    old_job = clip.job_id
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.FAILED.value
        stored.error_code = "RENDER_FAILED"
        job = session.get(Job, stored.job_id)
        assert job is not None
        job.status = JobStatus.FAILED.value
        session.commit()

    again = _create(session_factory, settings, video_id, [candidate_id])[0]

    assert again.id == clip.id
    assert again.status == ClipStatus.PENDING.value
    assert again.error_code is None
    assert again.job_id != old_job


def test_min_score_selects_candidates_in_bulk(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.4)],
    )

    with session_factory() as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        clips = clip_service.create_clips(
            session,
            video_id,
            options=clip_service.ClipOptions.from_settings(settings),
            settings=settings,
            min_score=0.65,
        )
        session.commit()

    assert len(clips) == 1
    assert clips[0].score == pytest.approx(0.9)


def test_bulk_generate_without_min_score_is_rejected(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])

    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, video_id, [])
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_incomplete_highlight_analysis_is_rejected(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(
        session_factory, video_id, [Candidate(10_000, 20_000, 0.5)], completed=False
    )[0]

    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, video_id, [candidate_id])
    assert excinfo.value.code is ErrorCode.HIGHLIGHT_INCOMPLETE


def test_unknown_video_and_candidate_are_rejected(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, "missing", [])
    assert excinfo.value.code is ErrorCode.VIDEO_NOT_FOUND

    video_id = _seed_video(session_factory)
    _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])
    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, video_id, ["missing-candidate"])
    assert excinfo.value.code is ErrorCode.HIGHLIGHT_NOT_FOUND


def test_unready_video_is_rejected(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory, status=VideoStatus.PENDING.value)

    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, video_id, [])
    assert excinfo.value.code is ErrorCode.CLIP_SOURCE_UNAVAILABLE


def test_impossible_bounds_are_rejected(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    # 130 s of interest exceeds the 120 s maximum clip duration.
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(0, 130_000, 0.5)])[0]

    with pytest.raises(AppError) as excinfo:
        _create(session_factory, settings, video_id, [candidate_id])
    assert excinfo.value.code is ErrorCode.CLIP_BOUNDS_INVALID


def test_active_job_cap_is_enforced(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    capped = settings.model_copy(update={"clip_max_concurrent_jobs": 1})

    with pytest.raises(AppError) as excinfo:
        _create(session_factory, capped, video_id, ids)
    assert excinfo.value.code is ErrorCode.CLIP_CONFLICT


def test_canceled_clip_with_an_active_job_is_not_rearmed(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    """A cancel only requests termination: the renderer may still be writing."""
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.CANCELED.value
        session.commit()

    again = _create(session_factory, settings, video_id, [candidate_id])[0]

    assert again.id == clip.id
    assert again.job_id == clip.job_id
    assert again.status == ClipStatus.CANCELED.value
    with session_factory() as session:
        jobs = session.scalars(
            select(Job).where(Job.type == clip_service.CLIP_JOB_TYPE)
        ).all()
        assert len(jobs) == 1


def test_completed_clip_is_returned_untouched(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.COMPLETED.value
        stored.output_path = "C:/media/clips/done.mp4"
        session.commit()

    again = _create(session_factory, settings, video_id, [candidate_id])[0]

    assert again.id == clip.id
    assert again.job_id == clip.job_id
    assert again.status == ClipStatus.COMPLETED.value
    assert again.output_path == "C:/media/clips/done.mp4"


def test_lost_worker_fails_the_clip_instead_of_leaving_it_processing(
    session_factory: sessionmaker[Session], settings: Settings, engine: Engine
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]
    assert clip.job_id is not None
    with session_factory() as session:
        job = session.get(Job, clip.job_id)
        assert job is not None
        job.max_attempts = 1
        session.commit()

    assert job_service.claim_next(engine, "worker-a") is not None
    with session_factory() as session:
        clip_service.start_processing(session, clip.id, job_id=clip.job_id)
        session.execute(
            update(Job)
            .where(Job.id == clip.job_id)
            .values(heartbeat_at=utc_now_ms() - 10 * 60 * 1000)
        )
        session.commit()

    recovered = job_service.recover_stale(engine, 60.0, utc_now_ms())

    assert recovered == [clip.job_id]
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        assert stored.status == ClipStatus.FAILED.value
        assert stored.error_code == ErrorCode.WORKER_LOST.value
        assert stored.finished_at is not None


def test_recovered_cancel_request_cancels_the_clip(
    session_factory: sessionmaker[Session], settings: Settings, engine: Engine
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]
    assert clip.job_id is not None
    assert job_service.claim_next(engine, "worker-a") is not None
    with session_factory() as session:
        clip_service.start_processing(session, clip.id, job_id=clip.job_id)
        job_service.request_cancel(session, clip.job_id)
        session.execute(
            update(Job)
            .where(Job.id == clip.job_id)
            .values(heartbeat_at=utc_now_ms() - 10 * 60 * 1000)
        )
        session.commit()

    recovered = job_service.recover_stale(engine, 60.0, utc_now_ms())

    assert recovered == [clip.job_id]
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        assert stored.status == ClipStatus.CANCELED.value
        assert stored.finished_at is not None


def test_retry_requires_a_failed_or_canceled_clip(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        with pytest.raises(AppError) as excinfo:
            clip_service.retry_clip(session, stored)
        assert excinfo.value.code is ErrorCode.CLIP_CONFLICT
        stored.status = ClipStatus.FAILED.value
        stored.error_code = "RENDER_FAILED"
        job = session.get(Job, clip.job_id)
        assert job is not None
        job.status = JobStatus.FAILED.value
        session.flush()
        new_job_id = clip_service.retry_clip(session, stored).job_id
        assert stored.status == ClipStatus.PENDING.value
        assert stored.error_code is None
        assert new_job_id != clip.job_id
        session.commit()


def test_cancel_pending_clip_cancels_its_job(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        clip_service.cancel_clip(session, stored)
        session.commit()
        assert stored.status == ClipStatus.CANCELED.value
        job = session.get(Job, clip.job_id)
        assert job is not None and job.status == JobStatus.CANCELED.value


def test_delete_removes_the_published_file(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    output = settings.clips_dir / video_id / f"{clip.id}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"video")

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.COMPLETED.value
        stored.output_path = str(output)
        session.flush()
        clip_service.delete_clip(session, stored, settings=settings)
        session.commit()

    assert not output.exists()
    with session_factory() as session:
        assert session.get(Clip, clip.id) is None


def test_delete_never_removes_a_file_outside_the_clip_directory(
    session_factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"user data")
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.output_path = str(outside)
        session.flush()
        clip_service.delete_clip(session, stored, settings=settings)
        session.commit()

    assert outside.exists()
    with session_factory() as session:
        assert session.get(Clip, clip.id) is None


def test_delete_refuses_a_processing_clip(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.PROCESSING.value
        session.flush()
        with pytest.raises(AppError) as excinfo:
            clip_service.delete_clip(session, stored, settings=settings)
        assert excinfo.value.code is ErrorCode.CLIP_CONFLICT


def test_stats_count_every_status(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    ids = _seed_candidates(
        session_factory,
        video_id,
        [Candidate(10_000, 20_000, 0.9), Candidate(100_000, 110_000, 0.8)],
    )
    clips = _create(session_factory, settings, video_id, ids)
    with session_factory() as session:
        first = session.get(Clip, clips[0].id)
        second = session.get(Clip, clips[1].id)
        assert first is not None and second is not None
        first.status = ClipStatus.COMPLETED.value
        second.status = ClipStatus.FAILED.value
        session.commit()
        stats = clip_service.clip_stats(session, video_id)

    assert (stats.total, stats.completed, stats.failed) == (2, 1, 1)
    assert (stats.pending, stats.processing, stats.canceled) == (0, 0, 0)
    with session_factory() as session:
        assert clip_service.clip_stats(session, "missing").total == 0


def test_resolve_clip_file_requires_published_output(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    video_id = _seed_video(session_factory)
    candidate_id = _seed_candidates(session_factory, video_id, [Candidate(10_000, 20_000, 0.5)])[0]
    clip = _create(session_factory, settings, video_id, [candidate_id])[0]

    with pytest.raises(AppError) as excinfo:
        clip_service.resolve_clip_file(settings, clip)
    assert excinfo.value.code is ErrorCode.CLIP_NOT_FOUND
