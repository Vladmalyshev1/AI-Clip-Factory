"""End-to-end speech-recognition stage over real FFmpeg audio (M3.6).

The fake engine is the only stub: extraction, chunk slicing and WAV decoding are
the production code paths, so this test catches the mistakes a pure fake cannot —
wrong extraction flags, a chunk window that does not tile the audio, a sample
rate mismatch or an off-by-one in the absolute timeline.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.ai.asr import AsrChunkResult, AsrSegment
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError
from app.core.status import SourceKind, TranscriptStatus, VideoStatus
from app.media import audio as audio_io
from app.media.binaries import find_ffmpeg
from app.models.transcript import TranscriptSegment as TranscriptSegmentRow
from app.models.video import Video
from app.pipeline.stages.asr import AsrStage
from app.pipeline.stages.base import StageContext
from app.services import transcript_service
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = [pytest.mark.media, pytest.mark.db]

CHUNK_SECONDS = 30
SOURCE_SECONDS = 61
SOURCE_MS = SOURCE_SECONDS * 1000
SAMPLE_RATE = 16_000


class CountingEngine:
    """Fake engine that records how much decoded audio each chunk produced."""

    model_name = "fake-whisper"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
                "samples": int(audio.size),
                "dtype": str(audio.dtype),
                "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
            }
        )
        start_ms = offset_ms + 250
        end_ms = min(offset_ms + duration_ms, start_ms + 1000)
        return AsrChunkResult(
            segments=(
                AsrSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=f"chunk starting at {offset_ms} ms",
                    avg_logprob=-0.3,
                    no_speech_prob=0.05,
                ),
            ),
            language="en",
            language_probability=0.98,
            duration_ms=duration_ms,
            duration_after_vad_ms=duration_ms,
            elapsed_s=0.0,
            model_name=self.model_name,
        )


class SilentReporter:
    """Progress reporter that never cancels."""

    def update(self, progress: float, message: str | None = None) -> None:
        return None

    def check_cancelled(self) -> None:
        return None


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One 61 s H.264/AAC MP4 with a tone, generated once per module."""
    binary = find_ffmpeg()
    if binary is None:  # pragma: no cover - environment problem
        pytest.skip("ffmpeg is not available in this environment")
    directory = tmp_path_factory.mktemp("asr-media")
    destination = directory / "tone.mp4"
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
            "testsrc=size=160x120:rate=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            str(SOURCE_SECONDS),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
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
    yield destination


def _insert_video(session_factory: sessionmaker[Session], source: Path) -> str:
    """Insert the generated recording as a ready video row."""
    now = utc_now_ms()
    video_id = "vid_asr_media"
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title="Tone recording",
                original_filename="tone.mp4",
                source_path=str(source),
                source_kind=SourceKind.UPLOAD.value,
                size_bytes=source.stat().st_size if source.exists() else 0,
                duration_ms=SOURCE_MS,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def test_stage_transcribes_real_audio_end_to_end(
    settings: Settings, session_factory: sessionmaker[Session], source_video: Path
) -> None:
    settings = settings.model_copy(update={"asr_chunk_seconds": CHUNK_SECONDS})
    video_id = _insert_video(session_factory, source_video)
    engine = CountingEngine()
    work_dir = settings.work_dir / "job_media" / "asr"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        job_id="job_media",
        video_id=video_id,
        attempt=1,
        work_dir=work_dir,
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )

    result = AsrStage(engine=engine).run(ctx, SilentReporter())

    # 61 s at 30 s per chunk -> [0, 30000) + [30000, 60000) + the short tail.
    # AAC padding makes the extracted audio slightly longer than the probe
    # duration (61.02 s here), which is why the tail chunk is not exactly 1 s.
    assert [call["offset_ms"] for call in engine.calls] == [0, 30_000, 60_000]
    assert [call["duration_ms"] for call in engine.calls[:2]] == [30_000, 30_000]
    tail_ms = engine.calls[2]["duration_ms"]
    assert 1_000 <= tail_ms <= 1_100

    # Every full chunk decodes to 30 s of mono float32 at 16 kHz.
    for call in engine.calls[:2]:
        assert call["dtype"] == "float32"
        assert abs(call["samples"] - 30 * SAMPLE_RATE) <= SAMPLE_RATE // 10
        assert call["peak"] > 0.001
    assert abs(engine.calls[2]["samples"] - tail_ms * SAMPLE_RATE // 1000) <= SAMPLE_RATE // 10

    with session_factory() as session:
        transcript = transcript_service.get_transcript(session, video_id)
        assert transcript.status == TranscriptStatus.COMPLETED.value
        segments = list(
            session.scalars(
                select(TranscriptSegmentRow)
                .where(TranscriptSegmentRow.transcript_id == transcript.id)
                .order_by(TranscriptSegmentRow.start_ms)
            ).all()
        )
        assert len(segments) == 3
        assert [segment.chunk_index for segment in segments] == [0, 1, 2]
        assert segments[0].start_ms == 250
        assert segments[2].start_ms == 60_250
        # The extracted audio can be a few milliseconds longer than the probe
        # duration (AAC padding), so the tail segment may end just past it.
        assert all(segment.end_ms <= SOURCE_MS + 100 for segment in segments)

    assert result.output["chunks_completed"] == 3
    assert result.output["segments"] == 3
    # The extracted WAV is removed on success; chunk files never survive a run.
    assert not (work_dir / audio_io.AUDIO_FILENAME).exists()
    assert list((work_dir / "chunks").glob("*.wav")) == []


def test_stage_rejects_a_source_without_audio(
    settings: Settings, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # A video row whose file does not exist still fails before extraction,
    # because the row says there is no audio stream.
    video_id = _insert_video(session_factory, tmp_path / "missing.mp4")
    with session_factory() as session:
        row = session.get(Video, video_id)
        assert row is not None
        row.has_audio = 0
        session.commit()

    ctx = StageContext(
        job_id="job_noaudio",
        video_id=video_id,
        attempt=1,
        work_dir=tmp_path / "work",
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )

    with pytest.raises(AppError) as excinfo:
        AsrStage(engine=CountingEngine()).run(ctx, SilentReporter())

    assert excinfo.value.code.value == "NO_AUDIO_STREAM"
