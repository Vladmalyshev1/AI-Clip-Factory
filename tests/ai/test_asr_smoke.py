"""Opt-in smoke test with the real faster-whisper model (M3.6).

The rest of the ASR suite uses fakes, which is fast and deterministic but says
nothing about whether the model, CTranslate2 and the audio path actually agree on
this machine. This test closes that gap and is therefore **opt-in**:

    set ACF_ASR_SMOKE_AUDIO=C:\\path\\to\\talk.mp4
    .\\.venv\\Scripts\\python.exe -m pytest tests/ai/test_asr_smoke.py -m slow -s

It is skipped (not failed) when the variable is unset, the file is missing, the
model was never fetched or FFmpeg is unavailable, so the default test run stays
hermetic and offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.ai.asr import get_engine
from app.ai.model_manager import model_available
from app.core.config import Settings
from app.media import audio as audio_io
from app.media.binaries import find_ffmpeg
from app.media.runner import FFmpegRunner

pytestmark = [pytest.mark.ai, pytest.mark.slow]

SMOKE_ENV = "ACF_ASR_SMOKE_AUDIO"
SMOKE_SECONDS_ENV = "ACF_ASR_SMOKE_SECONDS"
DEFAULT_SMOKE_SECONDS = 60.0


def _smoke_source() -> Path:
    """Return the configured recording or skip the test."""
    raw = os.environ.get(SMOKE_ENV, "").strip()
    if not raw:
        pytest.skip(f"set {SMOKE_ENV} to a recording with English speech to run this test")
    source = Path(raw).expanduser()
    if not source.is_file():
        pytest.skip(f"{SMOKE_ENV} points to a missing file: {source}")
    return source


def test_real_model_transcribes_real_speech(tmp_path: Path) -> None:
    """Extract, decode and transcribe real speech with the cached model."""
    source = _smoke_source()
    if find_ffmpeg() is None:  # pragma: no cover - environment problem
        pytest.skip("ffmpeg is not available in this environment")
    settings = Settings(_env_file=None, data_dir=tmp_path / "var", log_level="WARNING")
    settings.ensure_dirs()
    if not model_available(settings.asr_model):
        pytest.skip(
            f"model {settings.asr_model} is not cached; run "
            "python -m app.ai.model_manager fetch"
        )
    limit_s = float(os.environ.get(SMOKE_SECONDS_ENV, DEFAULT_SMOKE_SECONDS))

    audio_path = tmp_path / audio_io.AUDIO_FILENAME
    audio_io.extract_audio(
        FFmpegRunner.from_settings(settings), source, audio_path
    )
    duration_ms = audio_io.wav_duration_ms(audio_path)
    assert duration_ms is not None and duration_ms > 0

    limit_ms = min(duration_ms, int(limit_s * 1000))
    samples = audio_io.read_audio_16k(audio_path)
    samples = samples[: int(limit_ms * audio_io.TARGET_SAMPLE_RATE / 1000)]
    assert samples.size > 0

    engine = get_engine(settings)
    result = engine.transcribe(
        samples,
        offset_ms=0,
        duration_ms=limit_ms,
        language=settings.asr_language,
    )

    assert result.model_name == settings.asr_model
    assert result.language == settings.asr_language
    assert result.duration_ms == limit_ms
    assert result.segments, "the model returned no segments for real speech"

    previous_end = 0
    for segment in result.segments:
        assert segment.text.strip()
        assert 0 <= segment.start_ms < segment.end_ms <= limit_ms + 50
        assert segment.start_ms >= previous_end - 50
        previous_end = segment.end_ms
    text = " ".join(segment.text for segment in result.segments)
    assert any(character.isalpha() for character in text)

    rtf = result.elapsed_s / (limit_ms / 1000.0)
    print(
        f"\nsmoke: {len(result.segments)} segments, {len(text)} chars, "
        f"{limit_ms / 1000:.1f}s audio in {result.elapsed_s:.2f}s (RTF {rtf:.3f})"
    )
