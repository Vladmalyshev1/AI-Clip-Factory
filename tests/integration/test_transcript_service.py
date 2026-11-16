"""Transcript persistence: idempotent plans, atomic chunk saves and resume.

The tests are pure database tests: no ffmpeg, no ffprobe, no audio file. Chunk
plans and ASR segments are replaced by small frozen dataclasses, which is why
the service only depends on structural protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode
from app.core.status import (
    JobStatus,
    SourceKind,
    TranscriptChunkStatus,
    TranscriptStatus,
    VideoStatus,
)
from app.models.job import Job
from app.models.transcript import Transcript, TranscriptChunk, TranscriptSegment
from app.models.video import Video
from app.services import job_service, transcript_service
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

VIDEO_DURATION_MS = 1_260_000
CHUNK_SECONDS = 300
CHUNK_MS = CHUNK_SECONDS * 1000


@dataclass(frozen=True)
class Plan:
    """Minimal stand-in for the media layer's chunk plan."""

    index: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Seg:
    """Minimal stand-in for ``app.ai.asr.AsrSegment``."""

    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float | None = -0.1
    no_speech_prob: float | None = 0.01


def _plans(duration_ms: int = VIDEO_DURATION_MS, chunk_ms: int = CHUNK_MS) -> list[Plan]:
    plans: list[Plan] = []
    index = 0
    start = 0
    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        plans.append(Plan(index=index, start_ms=start, end_ms=end))
        index += 1
        start = end
    return plans


def _make_video(session: Session, *, duration_ms: int = VIDEO_DURATION_MS) -> Video:
    now = utc_now_ms()
    video = Video(
        title="Test VOD",
        original_filename="vod.mp4",
        source_path="C:/media/originals/vod.mp4",
        source_kind=SourceKind.IMPORT_IN_PLACE.value,
        size_bytes=1024,
        duration_ms=duration_ms,
        has_audio=1,
        status=VideoStatus.READY.value,
        created_at=now,
        updated_at=now,
    )
    session.add(video)
    session.flush()
    return video


def _make_transcript(
    session: Session, *, duration_ms: int = VIDEO_DURATION_MS
) -> tuple[Video, Transcript]:
    video = _make_video(session, duration_ms=duration_ms)
    transcript = transcript_service.get_or_create_transcript(
        session,
        video.id,
        language="en",
        model_name="Systran/faster-whisper-small",
        duration_ms=duration_ms,
        chunk_seconds=CHUNK_SECONDS,
        options={"device": "cpu", "compute_type": "int8", "beam_size": 5},
    )
    return video, transcript


def _segments_for(index: int, *, count: int = 2) -> list[Seg]:
    """Build ``count`` segments inside the absolute window of chunk ``index``."""
    base = index * CHUNK_MS
    return [
        Seg(start_ms=base + offset, end_ms=base + offset + 1000, text=f"chunk {index} line {n}")
        for n, offset in enumerate(range(0, count * 1000, 1000))
    ]


@pytest.mark.db
def test_get_or_create_transcript_is_idempotent_and_plans_chunks(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        video = _make_video(session)
        first = transcript_service.get_or_create_transcript(
            session,
            video.id,
            language="en",
            model_name="Systran/faster-whisper-small",
            duration_ms=VIDEO_DURATION_MS,
            chunk_seconds=CHUNK_SECONDS,
            options={"device": "cpu", "beam_size": 5},
        )
        session.commit()
        transcript_id = first.id
        assert first.status == TranscriptStatus.PENDING.value
        assert first.chunk_seconds == CHUNK_SECONDS
        assert first.options_json is not None

    with session_factory() as session:
        again = transcript_service.get_or_create_transcript(
            session,
            video.id,
            language="en",
            model_name="Systran/faster-whisper-base",
            duration_ms=VIDEO_DURATION_MS,
            chunk_seconds=CHUNK_SECONDS,
            options={"device": "cuda"},
        )
        session.commit()
        assert again.id == transcript_id, "a video must never get a second transcript row"
        assert again.model_name == "Systran/faster-whisper-base"
        assert transcript_service.get_transcript(session, video.id).id == transcript_id
        assert transcript_service.get_transcript_by_id(session, transcript_id).id == transcript_id
        assert session.scalar(select(func.count()).select_from(Transcript)) == 1

        count = transcript_service.ensure_chunks(session, again, _plans())
        session.commit()
        assert count == 5

        chunks = list(
            session.scalars(
                select(TranscriptChunk)
                .where(TranscriptChunk.transcript_id == transcript_id)
                .order_by(TranscriptChunk.chunk_index)
            ).all()
        )
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4]
        assert all(chunk.status == TranscriptChunkStatus.PENDING.value for chunk in chunks)
        assert [chunk.duration_ms for chunk in chunks] == [
            CHUNK_MS,
            CHUNK_MS,
            CHUNK_MS,
            CHUNK_MS,
            VIDEO_DURATION_MS - 4 * CHUNK_MS,
        ]
        assert chunks[0].attempts == 0
        assert chunks[0].segment_count == 0

        # Re-planning is idempotent and keeps rows at five.
        assert transcript_service.ensure_chunks(session, again, _plans()) == 5

    with session_factory() as session:
        with pytest.raises(AppError) as unknown_video:
            transcript_service.get_or_create_transcript(
                session,
                "does-not-exist",
                language="en",
                model_name="Systran/faster-whisper-small",
                duration_ms=1000,
                chunk_seconds=CHUNK_SECONDS,
            )
        assert unknown_video.value.code is ErrorCode.VIDEO_NOT_FOUND
        assert unknown_video.value.status_code == 404

        with pytest.raises(AppError) as missing:
            transcript_service.get_transcript(session, video.id + "-nope")
        assert missing.value.code is ErrorCode.TRANSCRIPT_NOT_FOUND
        assert missing.value.status_code == 404
        assert missing.value.details == {"video_id": video.id + "-nope"}


@pytest.mark.db
def test_plan_save_progress_and_complete(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        session.commit()
        transcript_id = transcript.id

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        transcript_service.start_transcript(session, transcript)
        session.commit()
        assert transcript.status == TranscriptStatus.RUNNING.value
        assert transcript.started_at is not None
        started_at = transcript.started_at

    with session_factory() as session:
        for index in (0, 1):
            transcript_service.mark_chunk_running(session, transcript_id, index)
            stored = transcript_service.save_chunk_result(
                session, transcript_id, index, _segments_for(index, count=2)
            )
            assert stored == 2
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        snapshot = transcript_service.progress(session, transcript)
        assert snapshot.chunks_total == 5
        assert snapshot.chunks_completed == 2
        assert snapshot.chunks_running == 0
        assert snapshot.chunks_failed == 0
        assert snapshot.audio_ms_total == VIDEO_DURATION_MS
        assert snapshot.audio_ms_processed == 2 * CHUNK_MS
        assert snapshot.current_chunk_index == 2
        assert snapshot.progress == pytest.approx(2 * CHUNK_MS / VIDEO_DURATION_MS)

    with session_factory() as session:
        with pytest.raises(AppError) as incomplete:
            transcript_service.complete_transcript(session, transcript_id)
        assert incomplete.value.code is ErrorCode.ASR_FAILED
        assert incomplete.value.status_code == 500
        assert incomplete.value.message == "Transcript is incomplete: 3 of 5 chunks processed"

    with session_factory() as session:
        for index in (2, 3, 4):
            transcript_service.mark_chunk_running(session, transcript_id, index)
            transcript_service.save_chunk_result(
                session, transcript_id, index, _segments_for(index, count=1)
            )
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.complete_transcript(session, transcript_id)
        session.commit()
        assert transcript.status == TranscriptStatus.COMPLETED.value
        assert transcript.finished_at is not None
        assert transcript.error_code is None
        assert transcript.started_at == started_at
        assert transcript_service.progress(session, transcript).progress == 1.0
        assert transcript_service.count_segments(session, transcript_id) == 7

        # Starting a completed transcript is a conflict, not a silent downgrade.
        with pytest.raises(AppError) as conflict:
            transcript_service.start_transcript(session, transcript)
        assert conflict.value.code is ErrorCode.TRANSCRIPT_CONFLICT
        assert conflict.value.status_code == 409


@pytest.mark.db
def test_resume_returns_only_unfinished_chunks_in_order(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        transcript_service.mark_chunk_running(session, transcript_id, 0)
        transcript_service.save_chunk_result(
            session, transcript_id, 0, _segments_for(0, count=1)
        )
        transcript_service.mark_chunk_failed(
            session, transcript_id, 1, error_code="ASR_FAILED", error_message="boom"
        )
        transcript_service.mark_chunk_running(session, transcript_id, 3)
        session.commit()

    with session_factory() as session:
        resumable = transcript_service.claimable_chunks(session, transcript_id)
        assert [chunk.chunk_index for chunk in resumable] == [1, 2, 3, 4]
        assert [chunk.status for chunk in resumable] == [
            TranscriptChunkStatus.FAILED.value,
            TranscriptChunkStatus.PENDING.value,
            TranscriptChunkStatus.RUNNING.value,
            TranscriptChunkStatus.PENDING.value,
        ]
        assert resumable[2].attempts == 1, "a chunk left running is retried, not skipped"


@pytest.mark.db
def test_save_chunk_result_replaces_segments_atomically(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        assert transcript_service.save_chunk_result(session, transcript_id, 0, _segments_for(0, count=2)) == 2
        session.commit()
        assert transcript_service.count_segments(session, transcript_id) == 2

        replacement = [
            Seg(start_ms=1000, end_ms=2000, text="  first retry  "),
            Seg(start_ms=2000, end_ms=3000, text="second retry"),
            Seg(start_ms=3000, end_ms=4000, text="third retry"),
        ]
        assert transcript_service.save_chunk_result(session, transcript_id, 0, replacement) == 3
        session.commit()
        assert transcript_service.count_segments(session, transcript_id) == 3

        chunk = session.get(TranscriptChunk, (transcript_id, 0))
        assert chunk is not None
        assert chunk.status == TranscriptChunkStatus.COMPLETED.value
        assert chunk.segment_count == 3
        assert chunk.processed_at is not None
        assert chunk.error_code is None

        items, total = transcript_service.list_segments(session, transcript_id, start_ms=0, end_ms=5000)
        assert total == 3
        assert [item.text for item in items] == ["first retry", "second retry", "third retry"]

        # Saving the very same result again is idempotent.
        assert transcript_service.save_chunk_result(session, transcript_id, 0, replacement) == 3
        session.commit()
        assert transcript_service.count_segments(session, transcript_id) == 3

        # A silent window completes the chunk without any segment row.
        assert transcript_service.save_chunk_result(session, transcript_id, 1, []) == 0
        session.commit()
        silent = session.get(TranscriptChunk, (transcript_id, 1))
        assert silent is not None
        assert silent.status == TranscriptChunkStatus.COMPLETED.value
        assert silent.segment_count == 0
        assert transcript_service.count_segments(session, transcript_id) == 3


@pytest.mark.db
def test_failed_chunk_stays_claimable_and_blocks_completion(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        transcript_service.save_chunk_result(session, transcript_id, 0, _segments_for(0, count=1))
        transcript_service.mark_chunk_failed(
            session,
            transcript_id,
            1,
            error_code=ErrorCode.ASR_FAILED.value,
            error_message="x" * 5000,
        )
        session.commit()

        failed = session.get(TranscriptChunk, (transcript_id, 1))
        assert failed is not None
        assert failed.status == TranscriptChunkStatus.FAILED.value
        assert failed.error_code == ErrorCode.ASR_FAILED.value
        assert failed.error_message is not None
        assert len(failed.error_message) == transcript_service.MAX_ERROR_CHARS

        assert 1 in [chunk.chunk_index for chunk in transcript_service.claimable_chunks(session, transcript_id)]

        with pytest.raises(AppError) as incomplete:
            transcript_service.complete_transcript(session, transcript_id)
        assert incomplete.value.code is ErrorCode.ASR_FAILED
        assert incomplete.value.details == {"unfinished_chunk_indexes": [1, 2, 3, 4]}


@pytest.mark.db
def test_list_segments_paginates_filters_and_validates(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        chunk_zero = [
            Seg(start_ms=0, end_ms=1000, text="zero"),
            Seg(start_ms=1000, end_ms=2500, text="one"),
            Seg(start_ms=2500, end_ms=5000, text="two"),
        ]
        chunk_one = [
            Seg(start_ms=CHUNK_MS, end_ms=CHUNK_MS + 1000, text="three"),
            Seg(start_ms=CHUNK_MS + 1000, end_ms=CHUNK_MS + 3000, text="four"),
        ]
        transcript_service.save_chunk_result(session, transcript_id, 0, chunk_zero)
        transcript_service.save_chunk_result(session, transcript_id, 1, chunk_one)
        session.commit()

        items, total = transcript_service.list_segments(session, transcript_id, offset=0, limit=2)
        assert total == 5
        assert [item.text for item in items] == ["zero", "one"]

        items, total = transcript_service.list_segments(session, transcript_id, offset=2, limit=2)
        assert total == 5
        assert [item.text for item in items] == ["two", "three"]

        # Overlap semantics: a segment that starts before the window still matches.
        items, total = transcript_service.list_segments(
            session, transcript_id, start_ms=1200, end_ms=2600
        )
        assert total == 2
        assert [item.text for item in items] == ["one", "two"]

        items, total = transcript_service.list_segments(session, transcript_id, start_ms=1000)
        assert total == 4
        assert [item.text for item in items] == ["one", "two", "three", "four"]

        items, total = transcript_service.list_segments(session, transcript_id, end_ms=2500)
        assert total == 2
        assert [item.text for item in items] == ["zero", "one"]

        items, total = transcript_service.list_segments(
            session, transcript_id, start_ms=CHUNK_MS + 500, end_ms=CHUNK_MS + 1500
        )
        assert total == 2
        assert [item.text for item in items] == ["three", "four"]

        with pytest.raises(AppError) as invalid:
            transcript_service.list_segments(session, transcript_id, start_ms=5000, end_ms=5000)
        assert invalid.value.code is ErrorCode.VALIDATION_ERROR
        assert invalid.value.status_code == 422


@pytest.mark.db
def test_reset_transcript_purges_or_keeps_chunks(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        transcript_service.start_transcript(session, transcript)
        for index in (0, 1):
            transcript_service.mark_chunk_running(session, transcript_id, index)
            transcript_service.save_chunk_result(
                session, transcript_id, index, _segments_for(index, count=2)
            )
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.reset_transcript(session, transcript_id, purge=False)
        session.commit()
        assert transcript.status == TranscriptStatus.PENDING.value
        assert transcript.finished_at is None
        assert session.scalar(
            select(func.count())
            .select_from(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript_id)
        ) == 5
        assert transcript_service.count_segments(session, transcript_id) == 4
        completed = [chunk.chunk_index for chunk in transcript_service.claimable_chunks(session, transcript_id)]
        assert completed == [2, 3, 4], "completed chunks survive a soft reset"

    with session_factory() as session:
        transcript = transcript_service.reset_transcript(session, transcript_id, purge=True)
        session.commit()
        assert transcript.status == TranscriptStatus.PENDING.value
        assert transcript.started_at is None
        assert session.scalar(
            select(func.count())
            .select_from(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript_id)
        ) == 0
        assert transcript_service.count_segments(session, transcript_id) == 0
        assert transcript_service.claimable_chunks(session, transcript_id) == []


@pytest.mark.db
def test_deleting_a_transcript_cascades_and_keeps_the_video(
    session_factory: sessionmaker[Session], engine: Engine
) -> None:
    with session_factory() as session:
        video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        video_id = video.id
        session.commit()

    with session_factory() as session:
        for index in (0, 1, 2):
            transcript_service.save_chunk_result(
                session, transcript_id, index, _segments_for(index, count=2)
            )
        session.commit()

    with session_factory() as session:
        session.execute(delete(Transcript).where(Transcript.id == transcript_id))
        session.commit()

    with Session(engine) as session:
        assert session.scalar(
            select(func.count())
            .select_from(TranscriptChunk)
            .where(TranscriptChunk.transcript_id == transcript_id)
        ) == 0
        assert session.scalar(
            select(func.count())
            .select_from(TranscriptSegment)
            .where(TranscriptSegment.transcript_id == transcript_id)
        ) == 0
        assert session.get(Video, video_id) is not None, "deleting a transcript must keep the video"
        assert session.get(Transcript, transcript_id) is None


@pytest.mark.db
def test_segments_are_stored_with_absolute_timestamps(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        stored = transcript_service.save_chunk_result(
            session,
            transcript_id,
            1,
            [Seg(start_ms=CHUNK_MS + 500, end_ms=CHUNK_MS + 2500, text="second window")],
        )
        session.commit()
        assert stored == 1

        segment = session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript_id)
        ).one()
        assert segment.chunk_index == 1
        assert segment.start_ms >= CHUNK_MS
        assert segment.end_ms >= segment.start_ms
        assert segment.created_at > 0


@pytest.mark.db
def test_asr_job_lookups_filter_by_type_and_state(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        video = _make_video(session)
        other = job_service.enqueue(session, type="pipeline", video_id=video.id)
        older = job_service.enqueue(
            session, type=transcript_service.ASR_JOB_TYPE, video_id=video.id
        )
        older.status = JobStatus.SUCCEEDED.value
        older.created_at = 1_700_000_000_000
        newer = job_service.enqueue(
            session, type=transcript_service.ASR_JOB_TYPE, video_id=video.id
        )
        # Deterministic ordering: both rows would otherwise share a millisecond
        # and only the UUID tie-breaker would decide which one is "latest".
        newer.created_at = 1_700_000_001_000
        session.commit()
        other_id, older_id, newer_id = other.id, older.id, newer.id

    with session_factory() as session:
        active = transcript_service.find_active_asr_job(session, video.id)
        assert active is not None
        assert active.id == newer_id
        assert isinstance(active, Job)

        latest = transcript_service.latest_asr_job(session, video.id)
        assert latest is not None
        assert latest.id == newer_id

    with session_factory() as session:
        newer = job_service.get_job(session, newer_id)
        newer.status = JobStatus.RUNNING.value
        session.commit()

    with session_factory() as session:
        running = transcript_service.find_active_asr_job(session, video.id)
        assert running is not None and running.id == newer_id

    with session_factory() as session:
        newer = job_service.get_job(session, newer_id)
        newer.status = JobStatus.CANCEL_REQUESTED.value
        session.commit()

    with session_factory() as session:
        cancelling = transcript_service.find_active_asr_job(session, video.id)
        assert cancelling is not None, "a cancel_requested job is still owned by the worker"
        assert cancelling.id == newer_id

    with session_factory() as session:
        newer = job_service.get_job(session, newer_id)
        newer.status = JobStatus.CANCELED.value
        session.commit()

    with session_factory() as session:
        assert transcript_service.find_active_asr_job(session, video.id) is None
        done = transcript_service.latest_asr_job(session, video.id)
        assert done is not None and done.id == newer_id
        assert session.get(Job, older_id) is not None
        pipeline_job = session.get(Job, other_id)
        assert pipeline_job is not None and pipeline_job.type == "pipeline"


@pytest.mark.db
def test_adopt_chunk_grid_keeps_the_recorded_grid(
    session_factory: sessionmaker[Session],
) -> None:
    """A settings change must never re-tile a transcript that already has rows."""
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        assert (
            transcript_service.adopt_chunk_grid(session, transcript, 60) == 60
        ), "a transcript without rows adopts the configured length"
        assert transcript.chunk_seconds == 60

        transcript_service.ensure_chunks(session, transcript, _plans(chunk_ms=60_000))
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript.id)
        assert (
            transcript_service.adopt_chunk_grid(session, transcript, CHUNK_SECONDS)
            == 60
        ), "the recorded grid wins as soon as a chunk row exists"
        assert transcript.chunk_seconds == 60


@pytest.mark.db
def test_ensure_chunks_rejects_a_completed_row_from_another_grid(
    session_factory: sessionmaker[Session],
) -> None:
    """Keeping a completed row from a different tiling would duplicate or lose speech."""
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_service.save_chunk_result(
            session, transcript.id, 0, _segments_for(0, count=1)
        )
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript.id)
        with pytest.raises(AppError) as conflict:
            transcript_service.ensure_chunks(
                session, transcript, _plans(chunk_ms=60_000)
            )

        assert conflict.value.code is ErrorCode.TRANSCRIPT_CONFLICT
        assert conflict.value.status_code == 409
        assert conflict.value.details["chunk_index"] == 0
        assert conflict.value.details["stored"] == {"start_ms": 0, "end_ms": CHUNK_MS}
        assert conflict.value.details["planned"] == {"start_ms": 0, "end_ms": 60_000}


@pytest.mark.db
def test_complete_transcript_requires_contiguous_and_complete_coverage(
    session_factory: sessionmaker[Session],
) -> None:
    """Chunks that do not tile the audio must never be reported as a success."""
    with session_factory() as session:
        _video, transcript = _make_transcript(session, duration_ms=600_000)
        transcript_service.ensure_chunks(session, transcript, _plans(duration_ms=600_000))
        transcript_service.start_transcript(session, transcript)
        transcript_service.save_chunk_result(
            session, transcript.id, 0, _segments_for(0, count=1)
        )
        transcript_service.save_chunk_result(
            session, transcript.id, 1, _segments_for(1, count=1)
        )
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        # The audio is longer than the recorded grid: the tail was never processed.
        with pytest.raises(AppError) as tail:
            transcript_service.complete_transcript(
                session, transcript_id, expected_ms=900_000
            )
        assert tail.value.code is ErrorCode.TRANSCRIPT_CONFLICT
        assert tail.value.status_code == 409
        assert tail.value.details["coverage"]["reason"] == "tail"

    with session_factory() as session:
        second = session.get(TranscriptChunk, (transcript_id, 1))
        assert second is not None
        second.start_ms = 400_000
        second.end_ms = 600_000
        second.duration_ms = 200_000
        session.commit()

        with pytest.raises(AppError) as gap:
            transcript_service.complete_transcript(session, transcript_id)
        assert gap.value.code is ErrorCode.TRANSCRIPT_CONFLICT
        assert gap.value.status_code == 409
        assert gap.value.details["coverage"] == {
            "reason": "gap",
            "after_chunk_index": 0,
            "gap_start_ms": CHUNK_MS,
            "gap_end_ms": 400_000,
        }

        # The failed call must not have changed the transcript.
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        assert transcript.status == TranscriptStatus.RUNNING.value


@pytest.mark.db
def test_complete_transcript_follows_the_status_machine(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session, duration_ms=CHUNK_MS)
        transcript_service.ensure_chunks(session, transcript, _plans(duration_ms=CHUNK_MS))
        transcript_service.save_chunk_result(
            session, transcript.id, 0, _segments_for(0, count=1)
        )
        transcript_id = transcript.id
        session.commit()

    with session_factory() as session:
        # Every chunk is done, but a ``pending`` transcript may not jump to
        # ``completed``: the stage starts a transcript before it works on it.
        with pytest.raises(AppError) as invalid:
            transcript_service.complete_transcript(session, transcript_id)
        assert invalid.value.code is ErrorCode.INVALID_STATE_TRANSITION
        assert invalid.value.status_code == 409

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        transcript_service.start_transcript(session, transcript)
        session.commit()

    with session_factory() as session:
        transcript = transcript_service.complete_transcript(session, transcript_id)
        session.commit()
        assert transcript.status == TranscriptStatus.COMPLETED.value


@pytest.mark.db
def test_resuming_a_failed_transcript_clears_its_finish_time(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _video, transcript = _make_transcript(session)
        transcript_service.ensure_chunks(session, transcript, _plans())
        transcript_service.start_transcript(session, transcript)
        transcript_service.fail_transcript(
            session, transcript.id, error_code="ASR_FAILED", error_message="boom"
        )
        transcript_id = transcript.id
        session.commit()
        assert transcript.finished_at is not None

    with session_factory() as session:
        transcript = transcript_service.get_transcript_by_id(session, transcript_id)
        transcript_service.start_transcript(session, transcript)
        session.commit()
        assert transcript.status == TranscriptStatus.RUNNING.value
        assert transcript.finished_at is None, "a running transcript has not finished"
        assert transcript.error_code is None
