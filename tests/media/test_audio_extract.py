"""Tests for audio extraction and the 16 kHz WAV helpers (M3.1).

Extraction drives the real pinned FFmpeg build and is marked ``media``; the
header-only helpers are pure filesystem checks and stay ``unit``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from app.media.audio import (
    AUDIO_FILENAME,
    TARGET_CHANNELS,
    TARGET_CODEC,
    TARGET_SAMPLE_RATE,
    build_extract_audio_command,
    extract_audio,
    read_audio_16k,
    remove_audio,
    wav_duration_ms,
    wav_matches,
)
from app.media.runner import FFmpegRunner, RenderError


def _write_wav(
    path: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = 1,
    seconds: float = 1.0,
) -> Path:
    frames = int(sample_rate * seconds)
    timeline = np.arange(frames, dtype=np.float32) / sample_rate
    mono = (0.25 * np.sin(2 * np.pi * 440.0 * timeline)).astype(np.float32)
    data = mono if channels == 1 else np.column_stack([mono] * channels)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


@pytest.mark.unit
def test_build_extract_audio_command_is_an_argument_list(tmp_path: Path) -> None:
    source = tmp_path / "source & ; 'quoted'.mp4"
    output = tmp_path / "work" / AUDIO_FILENAME

    command = build_extract_audio_command(source, output)

    assert command == [
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        TARGET_CODEC,
        "-f",
        "wav",
        str(output),
    ]
    assert all(isinstance(part, str) for part in command)


@pytest.mark.unit
def test_build_extract_audio_command_selects_an_audio_stream(tmp_path: Path) -> None:
    command = build_extract_audio_command(
        tmp_path / "source.mp4",
        tmp_path / "out.wav",
        sample_rate=48000,
        channels=2,
        audio_stream=3,
    )

    assert command[command.index("-map") + 1] == "0:a:3"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"


@pytest.mark.media
def test_extract_audio_produces_16k_mono_pcm16(
    sample_video: Path,
    ffmpeg_binary: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "work" / AUDIO_FILENAME
    fractions: list[float] = []

    result = extract_audio(
        FFmpegRunner(ffmpeg_binary),
        sample_video,
        output,
        duration_ms=1000,
        on_progress=fractions.append,
    )

    assert result.returncode == 0, result.stderr_tail
    assert output.is_file()
    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == TARGET_SAMPLE_RATE
        assert handle.getnchannels() == TARGET_CHANNELS
        assert handle.getsampwidth() == 2
        frames = handle.getnframes()
    assert frames > 0

    duration_ms = wav_duration_ms(output)
    assert duration_ms is not None
    assert abs(duration_ms - 1000) <= 50

    audio = read_audio_16k(output)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size == frames
    assert np.max(np.abs(audio)) > 0.0

    assert fractions
    assert fractions[-1] == 1.0


@pytest.mark.media
def test_extract_audio_fails_on_a_video_without_audio(
    silent_video: Path,
    ffmpeg_binary: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(RenderError):
        extract_audio(FFmpegRunner(ffmpeg_binary), silent_video, tmp_path / "audio.wav")


@pytest.mark.unit
def test_wav_duration_ms_returns_none_for_missing_and_corrupt_files(tmp_path: Path) -> None:
    assert wav_duration_ms(tmp_path / "missing.wav") is None

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"this is definitely not a wave file")
    assert wav_duration_ms(corrupt) is None

    truncated = tmp_path / "truncated.wav"
    good = _write_wav(tmp_path / "good.wav", seconds=1.0)
    truncated.write_bytes(good.read_bytes()[:24])
    assert wav_duration_ms(truncated) is None


@pytest.mark.unit
def test_wav_duration_ms_reads_the_header(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "audio.wav", seconds=2.5)

    assert wav_duration_ms(path) == 2500


@pytest.mark.unit
def test_wav_matches_checks_duration_and_format(tmp_path: Path) -> None:
    mono = _write_wav(tmp_path / "mono.wav", seconds=1.0)
    stereo = _write_wav(tmp_path / "stereo.wav", channels=2, seconds=1.0)
    wrong_rate = _write_wav(tmp_path / "rate.wav", sample_rate=44100, seconds=1.0)

    assert wav_matches(mono, 1000) is True
    assert wav_matches(mono, 1010) is True
    assert wav_matches(mono, 1400) is False
    assert wav_matches(tmp_path / "missing.wav", 1000) is False
    assert wav_matches(stereo, 1000) is False
    assert wav_matches(wrong_rate, 1000) is False


@pytest.mark.unit
def test_read_audio_16k_rejects_wrong_rate_and_channels(tmp_path: Path) -> None:
    stereo = _write_wav(tmp_path / "stereo.wav", channels=2)
    wrong_rate = _write_wav(tmp_path / "rate.wav", sample_rate=44100)

    with pytest.raises(ValueError, match="16000"):
        read_audio_16k(wrong_rate)
    with pytest.raises(ValueError, match="mono"):
        read_audio_16k(stereo)


@pytest.mark.unit
def test_remove_audio_is_idempotent(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / AUDIO_FILENAME)

    remove_audio(path)
    remove_audio(path)

    assert not path.exists()
