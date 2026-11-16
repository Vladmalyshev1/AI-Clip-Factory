"""``AsrStage``: chunked, resumable speech recognition with a fake engine (M3.4).

Nothing here touches FFmpeg, soundfile or Whisper. Extraction, slicing and WAV
decoding are replaced by tiny fakes, and the engine returns deterministic
absolute-timeline segments, so the tests exercise exactly the stage's own logic:
chunk planning, per-chunk commits, resume, retries, cancellation and failure
bookkeeping against a real SQLite database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.ai import asr as asr_api
from app.ai.asr import AsrChunkResult, AsrError, AsrSegment
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.status import JobStatus, SourceKind, TranscriptStatus, VideoStatus
from app.media import audio as audio_io
from app.media import audio_chunks
from app.media.runner import RenderError, RenderResult
from app.models.transcript import Transcript, TranscriptChunk, TranscriptSegment
from app.models.video import Video
from app.pipeline import orchestrator
from app.pipeline.stages import asr as asr_stage
from app.pipeline.stages.asr import AsrStage
from app.pipeline.stages.base import StageContext
from app.services import job_service, transcript_service
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

CHUNK_SECONDS = 300
CHUNK_MS = CHUNK_SECONDS * 1000
#: 15 minutes -> exactly three 300 s chunks.
VIDEO_MS = 3 * CHUNK_MS
FAKE_MODEL = "fake-whisper"


@dataclass
class FakeEngine:
    """Deterministic stand-in for :class:`app.ai.asr.AsrEngine`."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    #: offset_ms -> number of failures still to raise (``AsrError``).
    failures: dict[int, int] = field(default_factory=dict)
    #: offsets that simulate a cancellation raised from inside decoding.
    cancel_offsets: set[int] = field(default_factory=set)
    model_name: str = FAKE_MODEL

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_ms: int,
        duration_ms: int,
        language: str | None = None,
        should_cancel: Any = None,
    ) -> AsrChunkResult:
        self.calls.append(
            {
                "offset_ms": offset_ms,
                "duration_ms": duration_ms,
                "language": language,
                "samples": int(audio.size),
            }
        )
        if should_cancel is not None:
            should_cancel()
        if offset_ms in self.cancel_offsets:
            raise StageCancelled(f"cancelled during chunk at {offset_ms} ms")
        remaining = self.failures.get(offset_ms, 0)
        if remaining > 0:
            self.failures[offset_ms] = remaining - 1
            raise AsrError(
                f"decoder exploded at {offset_ms} ms", details={"offset_ms": offset_ms}
            )
        return AsrChunkResult(
            segments=(
                AsrSegment(
                    start_ms=offset_ms + 1000,
                    end_ms=offset_ms + 2000,
                    text=f"speech of the chunk at {offset_ms} ms",
                    avg_logprob=-0.25,
                    no_speech_prob=0.02,
                ),
            ),
            language="en",
            language_probability=0.99,
            duration_ms=duration_ms,
            duration_after_vad_ms=duration_ms,
            elapsed_s=0.01,
            model_name=self.model_name,
        )


class RecordingReporter:
    """Collects progress updates and mimics the orchestrator's reporter."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def update(self, progress: float, message: str | None = None) -> None:
        self.updates.append((progress, message))

    def check_cancelled(self) -> None:
        return None


class FakeRunner:
    """Stand-in for :class:`app.media.runner.FFmpegRunner` (never runs FFmpeg)."""

    @classmethod
    def from_settings(cls, settings: Settings) -> FakeRunner:
        return cls()


@dataclass
class MediaCalls:
    """Counters for the media fakes."""

    extract: int = 0
    slice: int = 0
    removed_audio: int = 0
    slice_error: RenderError | None = None
    extract_cancel_check: Any = None
    slice_cancel_check: Any = None


@pytest.fixture()
def media(monkeypatch: pytest.MonkeyPatch) -> MediaCalls:
    """Replace extraction, slicing and decoding with cheap file stubs."""
    calls = MediaCalls()

    def fake_extract(
        runner: Any,
        source_path: Path,
        output_path: Path,
        *,
        duration_ms: int | None = None,
        on_progress: Any = None,
        cancel_event: Any = None,
        cancel_check: Any = None,
    ) -> RenderResult:
        calls.extract += 1
        calls.extract_cancel_check = cancel_check
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFFfake")
        if on_progress is not None:
            on_progress(1.0)
        return RenderResult(returncode=0, elapsed_s=0.01, stderr_tail="", command=[])

    def fake_slice(
        runner: Any,
        source_wav: Path,
        output_wav: Path,
        plan: Any,
        *,
        cancel_check: Any = None,
    ) -> RenderResult:
        calls.slice += 1
        calls.slice_cancel_check = cancel_check
        if calls.slice_error is not None:
            raise calls.slice_error
        target = Path(output_wav)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFFfake")
        return RenderResult(returncode=0, elapsed_s=0.01, stderr_tail="", command=[])

    def fake_remove_audio(path: Path) -> None:
        calls.removed_audio += 1

    monkeypatch.setattr(audio_io, "extract_audio", fake_extract)
    monkeypatch.setattr(audio_io, "wav_matches", lambda path, expected_ms, **kw: False)
    monkeypatch.setattr(audio_io, "wav_duration_ms", lambda path: VIDEO_MS)
    monkeypatch.setattr(
        audio_io, "read_audio_16k", lambda path: np.zeros(1600, dtype=np.float32)
    )
    monkeypatch.setattr(audio_io, "remove_audio", fake_remove_audio)
    monkeypatch.setattr(audio_chunks, "slice_chunk", fake_slice)
    monkeypatch.setattr(asr_stage, "FFmpegRunner", FakeRunner)
    return calls


def _create_video(
    factory: sessionmaker[Session],
    *,
    duration_ms: int | None = VIDEO_MS,
    has_audio: int | None = 1,
    video_id: str = "vid_asr",
) -> str:
    """Insert one probed source recording and return its id."""
    now = utc_now_ms()
    with factory() as session:
        session.add(
            Video(
                id=video_id,
                title="Source recording",
                original_filename="vod.mp4",
                source_path=str(Path("media") / "originals" / "vod.mp4"),
                source_kind=SourceKind.UPLOAD.value,
                size_bytes=1024,
                duration_ms=duration_ms,
                has_audio=has_audio,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def _context(
    settings: Settings, factory: sessionmaker[Session], video_id: str | None
) -> StageContext:
    work_dir = settings.work_dir / "job_asr" / "asr"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        job_id="job_asr",
        video_id=video_id,
        attempt=1,
        work_dir=work_dir,
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=factory,
    )


def _transcript(factory: sessionmaker[Session], video_id: str) -> Transcript:
    with factory() as session:
        return transcript_service.get_transcript(session, video_id)


def _chunks(factory: sessionmaker[Session], transcript_id: str) -> list[TranscriptChunk]:
    with factory() as session:
        return list(
            session.scalars(
                select(TranscriptChunk)
                .where(TranscriptChunk.transcript_id == transcript_id)
                .order_by(TranscriptChunk.chunk_index)
            ).all()
        )


def _segment_texts(factory: sessionmaker[Session], transcript_id: str) -> list[str]:
    with factory() as session:
        return list(
            session.scalars(
                select(TranscriptSegment.text)
                .where(TranscriptSegment.transcript_id == transcript_id)
                .order_by(TranscriptSegment.start_ms)
            ).all()
        )


def test_transcribes_every_chunk_and_completes(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    engine = FakeEngine()
    video_id = _create_video(session_factory)
    reporter = RecordingReporter()

    result = AsrStage(engine=engine).run(
        _context(settings, session_factory, video_id), reporter
    )

    assert [call["offset_ms"] for call in engine.calls] == [0, CHUNK_MS, 2 * CHUNK_MS]
    assert all(call["duration_ms"] == CHUNK_MS for call in engine.calls)
    assert all(call["language"] == settings.asr_language for call in engine.calls)
    assert [call["samples"] for call in engine.calls] == [1600, 1600, 1600]

    transcript = _transcript(session_factory, video_id)
    assert transcript.status == TranscriptStatus.COMPLETED.value
    assert transcript.model_name == FAKE_MODEL
    assert [chunk.status for chunk in _chunks(session_factory, transcript.id)] == [
        "completed",
        "completed",
        "completed",
    ]
    assert len(_segment_texts(session_factory, transcript.id)) == 3

    assert result.output["transcript_id"] == transcript.id
    assert result.output["segments"] == 3
    assert result.output["chunks_total"] == 3
    assert result.output["chunks_completed"] == 3
    assert result.checkpoint == {
        "transcript_id": transcript.id,
        "chunks_total": 3,
        "chunks_completed": 3,
        "segments": 3,
    }
    assert reporter.updates[-1][0] == pytest.approx(1.0)
    assert media.extract == 1
    assert media.removed_audio == 1
    assert list((settings.work_dir / "job_asr" / "asr" / "chunks").glob("*.wav")) == []


def test_resume_skips_completed_chunks(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    video_id = _create_video(session_factory)
    with session_factory() as session:
        transcript = transcript_service.get_or_create_transcript(
            session,
            video_id,
            language="en",
            model_name=FAKE_MODEL,
            duration_ms=VIDEO_MS,
            chunk_seconds=CHUNK_SECONDS,
        )
        plans = audio_chunks.plan_chunks(VIDEO_MS, CHUNK_SECONDS)
        transcript_service.ensure_chunks(session, transcript, plans)
        transcript_service.start_transcript(session, transcript)
        transcript_service.mark_chunk_running(session, transcript.id, 0)
        transcript_service.save_chunk_result(
            session,
            transcript.id,
            0,
            [
                AsrSegment(
                    start_ms=500,
                    end_ms=1500,
                    text="already transcribed speech",
                    avg_logprob=-0.1,
                    no_speech_prob=0.0,
                )
            ],
        )
        # A chunk left ``running`` by a crashed worker must be claimed again.
        transcript_service.mark_chunk_running(session, transcript.id, 1)
        transcript_id = transcript.id
        session.commit()

    engine = FakeEngine()
    result = AsrStage(engine=engine).run(
        _context(settings, session_factory, video_id), RecordingReporter()
    )

    assert [call["offset_ms"] for call in engine.calls] == [CHUNK_MS, 2 * CHUNK_MS]
    assert _segment_texts(session_factory, transcript_id)[0] == "already transcribed speech"
    assert len(_segment_texts(session_factory, transcript_id)) == 3
    assert _transcript(session_factory, video_id).status == TranscriptStatus.COMPLETED.value
    assert result.output["chunks_completed"] == 3


def test_completed_transcript_cannot_be_rerun_without_reset(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    video_id = _create_video(session_factory)
    engine = FakeEngine()
    AsrStage(engine=engine).run(_context(settings, session_factory, video_id), RecordingReporter())
    calls_before = len(engine.calls)

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=engine).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.TRANSCRIPT_CONFLICT
    assert excinfo.value.status_code == 409
    assert len(engine.calls) == calls_before


def test_transient_decode_failure_is_retried(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    retried = settings.model_copy(update={"asr_chunk_max_attempts": 2})
    video_id = _create_video(session_factory)
    engine = FakeEngine(failures={CHUNK_MS: 1})

    AsrStage(engine=engine).run(
        _context(retried, session_factory, video_id), RecordingReporter()
    )

    offsets = [call["offset_ms"] for call in engine.calls]
    assert offsets == [0, CHUNK_MS, CHUNK_MS, 2 * CHUNK_MS]
    transcript = _transcript(session_factory, video_id)
    assert transcript.status == TranscriptStatus.COMPLETED.value
    assert len(_segment_texts(session_factory, transcript.id)) == 3


def test_exhausted_chunk_retries_fail_the_transcript(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    settings = settings.model_copy(update={"asr_chunk_max_attempts": 2})
    video_id = _create_video(session_factory)
    engine = FakeEngine(failures={CHUNK_MS: 5})

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=engine).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    error = excinfo.value
    assert error.code is ErrorCode.ASR_FAILED
    assert error.details is not None
    assert error.details["chunk_index"] == 1
    assert error.details["reason"] == "decode"
    assert error.details["attempts"] == 2
    # Chunk 0 finished, chunk 1 was attempted twice and is recorded as failed.
    assert [call["offset_ms"] for call in engine.calls] == [0, CHUNK_MS, CHUNK_MS]
    transcript = _transcript(session_factory, video_id)
    assert transcript.status == TranscriptStatus.FAILED.value
    assert transcript.error_code == ErrorCode.ASR_FAILED.value
    chunks = _chunks(session_factory, transcript.id)
    assert [chunk.status for chunk in chunks] == ["completed", "failed", "pending"]
    assert chunks[1].error_code == ErrorCode.ASR_FAILED.value
    assert "decoder exploded" in (chunks[1].error_message or "")


def test_cancellation_pauses_and_keeps_finished_work(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    video_id = _create_video(session_factory)
    engine = FakeEngine(cancel_offsets={CHUNK_MS})

    with pytest.raises(StageCancelled):
        AsrStage(engine=engine).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    transcript = _transcript(session_factory, video_id)
    assert transcript.status == TranscriptStatus.PENDING.value
    assert transcript.error_code is None
    chunks = _chunks(session_factory, transcript.id)
    # The chunk in flight stays ``running``, which makes it claimable again.
    assert [chunk.status for chunk in chunks] == ["completed", "running", "pending"]
    assert media.removed_audio == 0
    assert (settings.work_dir / "job_asr" / "asr" / "audio.wav").exists()

    # A second run resumes exactly where the cancelled one stopped.
    engine.cancel_offsets.clear()
    result = AsrStage(engine=engine).run(
        _context(settings, session_factory, video_id), RecordingReporter()
    )
    assert [call["offset_ms"] for call in engine.calls] == [0, CHUNK_MS, CHUNK_MS, 2 * CHUNK_MS]
    assert result.output["chunks_completed"] == 3
    assert _transcript(session_factory, video_id).status == TranscriptStatus.COMPLETED.value


def test_slicing_failure_is_retried_and_reported_as_audio(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    settings = settings.model_copy(update={"asr_chunk_max_attempts": 1})
    video_id = _create_video(session_factory)
    media.slice_error = RenderError("FFmpeg failed with exit code 1", stderr_tail="boom")

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=FakeEngine()).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.ASR_FAILED
    assert excinfo.value.details is not None
    assert excinfo.value.details["reason"] == "audio"
    assert excinfo.value.details["stderr_tail"] == "boom"
    assert _transcript(session_factory, video_id).status == TranscriptStatus.FAILED.value


def test_existing_audio_is_reused(
    settings: Settings,
    session_factory: sessionmaker[Session],
    media: MediaCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_io, "wav_matches", lambda path, expected_ms, **kw: True)
    video_id = _create_video(session_factory)

    AsrStage(engine=FakeEngine()).run(
        _context(settings, session_factory, video_id), RecordingReporter()
    )

    assert media.extract == 0


def test_video_without_audio_is_rejected(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    video_id = _create_video(session_factory, has_audio=0)

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=FakeEngine()).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.NO_AUDIO_STREAM
    assert media.extract == 0
    with session_factory() as session:
        stored = session.scalar(
            select(func.count()).select_from(Transcript).where(Transcript.video_id == video_id)
        )
    assert stored == 0


def test_unprobed_audio_flag_is_rejected(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    """``has_audio`` is NULL for a video that was never probed: refuse it early."""
    video_id = _create_video(session_factory, has_audio=None)

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=FakeEngine()).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.NO_AUDIO_STREAM
    assert media.extract == 0


def test_changed_chunk_seconds_cannot_re_tile_a_resumed_transcript(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    """A resumed transcript keeps its recorded grid, so no chunk is duplicated."""
    video_id = _create_video(session_factory)
    with session_factory() as session:
        transcript = transcript_service.get_or_create_transcript(
            session,
            video_id,
            language="en",
            model_name=FAKE_MODEL,
            duration_ms=VIDEO_MS,
            chunk_seconds=CHUNK_SECONDS,
        )
        transcript_service.ensure_chunks(
            session, transcript, audio_chunks.plan_chunks(VIDEO_MS, CHUNK_SECONDS)
        )
        transcript_service.start_transcript(session, transcript)
        transcript_service.save_chunk_result(
            session,
            transcript.id,
            0,
            [AsrSegment(start_ms=500, end_ms=1500, text="first window", avg_logprob=-0.1, no_speech_prob=0.0)],
        )
        transcript_id = transcript.id
        session.commit()

    changed = settings.model_copy(update={"asr_chunk_seconds": CHUNK_SECONDS // 3})
    engine = FakeEngine()
    AsrStage(engine=engine).run(
        _context(changed, session_factory, video_id), RecordingReporter()
    )

    # The plan still uses the 300 s grid recorded on the transcript, not the
    # newly configured 100 s, so the two remaining windows are untouched.
    assert [call["offset_ms"] for call in engine.calls] == [CHUNK_MS, 2 * CHUNK_MS]
    assert all(call["duration_ms"] == CHUNK_MS for call in engine.calls)
    transcript = _transcript(session_factory, video_id)
    assert transcript.chunk_seconds == CHUNK_SECONDS
    assert transcript.status == TranscriptStatus.COMPLETED.value
    assert len(_segment_texts(session_factory, transcript_id)) == 3


def test_media_operations_poll_the_stage_cancellation(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    """Extraction and slicing must stop even when FFmpeg emits no progress."""
    video_id = _create_video(session_factory)

    class CancellingReporter(RecordingReporter):
        cancelled: bool = False

        def check_cancelled(self) -> None:
            if self.cancelled:
                raise StageCancelled("stop requested")

    reporter = CancellingReporter()
    AsrStage(engine=FakeEngine()).run(
        _context(settings, session_factory, video_id), reporter
    )

    assert media.extract_cancel_check is not None
    assert media.slice_cancel_check is not None
    # Before a cancellation the callback is a cheap no-op.
    media.extract_cancel_check()
    media.slice_cancel_check()

    reporter.cancelled = True
    with pytest.raises(StageCancelled):
        media.extract_cancel_check()
    with pytest.raises(StageCancelled):
        media.slice_cancel_check()


def test_degenerate_extraction_fails_instead_of_hallucinating(
    settings: Settings,
    session_factory: sessionmaker[Session],
    media: MediaCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-length WAV must fail the stage, not decode 30 s of silence."""
    monkeypatch.setattr(audio_io, "wav_duration_ms", lambda path: 0)
    video_id = _create_video(session_factory)
    engine = FakeEngine()

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=engine).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.ASR_FAILED
    assert excinfo.value.details is not None
    assert excinfo.value.details["action"] == "extract"
    assert engine.calls == []
    assert _transcript(session_factory, video_id).status == TranscriptStatus.FAILED.value


def test_video_without_duration_is_rejected(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    video_id = _create_video(session_factory, duration_ms=None)

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=FakeEngine()).run(
            _context(settings, session_factory, video_id), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.PROBE_FAILED
    assert excinfo.value.status_code == 422


def test_job_without_video_is_rejected(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=FakeEngine()).run(
            _context(settings, session_factory, None), RecordingReporter()
        )

    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_stage_requires_settings_and_session_factory(
    settings: Settings, session_factory: sessionmaker[Session], media: MediaCalls
) -> None:
    ctx = StageContext(
        job_id="job_missing",
        video_id="vid_missing",
        attempt=1,
        work_dir=settings.work_dir,
        params={},
        checkpoint=None,
    )

    with pytest.raises(RuntimeError, match="settings"):
        AsrStage(engine=FakeEngine()).run(ctx, RecordingReporter())


def test_run_job_uses_the_transcribe_pipeline(
    engine: Engine,
    settings: Settings,
    session_factory: sessionmaker[Session],
    media: MediaCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEngine()
    monkeypatch.setattr(asr_api, "get_engine", lambda settings: fake)
    video_id = _create_video(session_factory)
    with session_factory() as session:
        job = job_service.enqueue(session, type="transcribe", video_id=video_id)
        job_id = job.id
        session.commit()

    claimed = job_service.claim_next(engine, "worker-asr")
    assert claimed is not None
    assert claimed.type == "transcribe"

    status = orchestrator.run_job(engine, settings, claimed, "worker-asr")

    assert status is JobStatus.SUCCEEDED
    with session_factory() as session:
        assert [stage.name for stage in job_service.list_job_stages(session, job_id)] == ["asr"]
        assert job_service.get_job(session, job_id).status == JobStatus.SUCCEEDED.value
    assert _transcript(session_factory, video_id).status == TranscriptStatus.COMPLETED.value
