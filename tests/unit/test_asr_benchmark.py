"""Unit tests for the ASR benchmark tool (no model, no FFmpeg)."""

from __future__ import annotations

import json
import wave
from dataclasses import asdict
from pathlib import Path

import asr_benchmark as bench
import pytest


def _write_wav(path: Path, *, rate: int = 16_000, channels: int = 1, seconds: int = 2) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * rate * channels * seconds)
    return path


@pytest.mark.unit
def test_target_wav_detection(tmp_path: Path) -> None:
    assert bench._is_target_wav(_write_wav(tmp_path / "16k-mono.wav"))
    assert not bench._is_target_wav(
        _write_wav(tmp_path / "8k-mono.wav", rate=8_000)
    )
    assert not bench._is_target_wav(
        _write_wav(tmp_path / "16k-stereo.wav", channels=2)
    )
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    assert not bench._is_target_wav(broken)
    assert not bench._is_target_wav(tmp_path / "missing.wav")


@pytest.mark.unit
def test_missing_source_fails_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = bench.main([str(tmp_path / "missing.mp4")])

    assert exit_code == 1
    assert "is not a file" in capsys.readouterr().err


@pytest.mark.unit
def test_too_short_limit_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_wav(tmp_path / "sample.wav")

    exit_code = bench.main([str(source), "--limit-seconds", "1"])

    assert exit_code == 2
    assert "at least" in capsys.readouterr().err


@pytest.mark.unit
def test_parser_defaults() -> None:
    args = bench._build_parser().parse_args(["recording.mp4"])

    assert args.chunk_seconds is None
    assert args.limit_seconds == bench.DEFAULT_LIMIT_SECONDS
    assert args.json is False


@pytest.mark.unit
def test_report_serialises_as_json() -> None:
    report = bench.BenchmarkReport(
        source="recording.mp4",
        source_duration_ms=600_000,
        analysed_ms=300_000,
        chunk_seconds=300,
        model="Systran/faster-whisper-small",
        device="cpu",
        compute_type="int8",
        cpu_threads=6,
        beam_size=5,
        vad_filter=True,
        model_load_s=1.2,
        prepare_s=1.3,
        total_audio_s=300.0,
        total_wall_s=31.5,
        rtf=0.105,
        segments=95,
        chars=4200,
        peak_rss_mb=435.0,
        chunks=[
            bench.ChunkMeasurement(
                index=0,
                start_ms=0,
                end_ms=300_000,
                audio_s=300.0,
                wall_s=31.5,
                decode_s=31.0,
                rtf=0.105,
                segments=95,
                chars=4200,
            )
        ],
    )

    payload = json.loads(json.dumps(asdict(report)))

    assert payload["chunks"][0]["rtf"] == 0.105
    assert payload["rtf"] == 0.105
