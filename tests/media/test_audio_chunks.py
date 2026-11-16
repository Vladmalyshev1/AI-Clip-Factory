"""Tests for chunk planning, slicing and cleanup (M3.1).

Planning is pure; the slice test drives the real FFmpeg build and verifies
sample-exact cutting (PCM re-encode, not stream copy).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from app.media.audio import read_audio_16k, wav_duration_ms
from app.media.audio_chunks import (
    ChunkPlan,
    build_slice_command,
    chunk_count,
    chunk_filename,
    chunk_path,
    cleanup_chunks,
    plan_chunks,
    remove_file,
    slice_chunk,
)
from app.media.runner import FFmpegRunner


def _write_wav(path: Path, *, seconds: float = 3.0, sample_rate: int = 16000) -> Path:
    frames = int(sample_rate * seconds)
    timeline = np.arange(frames, dtype=np.float32) / sample_rate
    data = (0.3 * np.sin(2 * np.pi * 330.0 * timeline)).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


@pytest.mark.unit
def test_plan_chunks_tiles_an_exact_multiple() -> None:
    plans = plan_chunks(600_000, 300)

    assert plans == [
        ChunkPlan(index=0, start_ms=0, end_ms=300_000),
        ChunkPlan(index=1, start_ms=300_000, end_ms=600_000),
    ]
    assert [plan.duration_ms for plan in plans] == [300_000, 300_000]


@pytest.mark.unit
def test_plan_chunks_produces_a_partial_tail() -> None:
    plans = plan_chunks(610_500, 300)

    assert len(plans) == 3
    assert plans[-1] == ChunkPlan(index=2, start_ms=600_000, end_ms=610_500)
    assert plans[-1].duration_ms == 10_500
    assert plans[0].start_s == 0.0
    assert plans[-1].end_s == 610.5


@pytest.mark.unit
def test_plan_chunks_coalesces_a_tail_shorter_than_one_second() -> None:
    plans = plan_chunks(300_001, 300)

    assert [plan.duration_ms for plan in plans] == [300_000, 1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("duration_ms", "chunk_seconds"),
    [(0, 300), (-1, 300), (600_000, 0), (600_000, -5)],
)
def test_plan_chunks_returns_nothing_for_invalid_input(
    duration_ms: int, chunk_seconds: int
) -> None:
    assert plan_chunks(duration_ms, chunk_seconds) == []
    assert chunk_count(duration_ms, chunk_seconds) == 0


@pytest.mark.unit
def test_plan_chunks_never_emits_zero_length_windows() -> None:
    plans = plan_chunks(1, 300)

    assert plans == [ChunkPlan(index=0, start_ms=0, end_ms=1)]
    assert all(plan.duration_ms > 0 for plan in plan_chunks(1_234_567, 300))


@pytest.mark.unit
def test_chunk_count_matches_plan_chunks() -> None:
    for duration_ms in (1, 999, 300_000, 300_001, 900_000, 1_234_567):
        assert chunk_count(duration_ms, 300) == len(plan_chunks(duration_ms, 300))


@pytest.mark.unit
def test_chunk_names_are_sortable_and_paths_are_stable(tmp_path: Path) -> None:
    assert chunk_filename(0) == "chunk-00000.wav"
    assert chunk_filename(12) == "chunk-00012.wav"
    assert chunk_filename(99_999) == "chunk-99999.wav"
    assert chunk_path(tmp_path, 7) == tmp_path / "chunk-00007.wav"


@pytest.mark.unit
def test_chunk_plan_rejects_degenerate_windows() -> None:
    with pytest.raises(ValueError):
        ChunkPlan(index=-1, start_ms=0, end_ms=1000)
    with pytest.raises(ValueError):
        ChunkPlan(index=0, start_ms=-1, end_ms=1000)
    with pytest.raises(ValueError):
        ChunkPlan(index=0, start_ms=1000, end_ms=1000)
    with pytest.raises(ValueError):
        ChunkPlan(index=0, start_ms=2000, end_ms=1000)


@pytest.mark.unit
def test_build_slice_command_seeks_before_the_input(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    output = tmp_path / "chunks" / "chunk-00000.wav"
    plan = ChunkPlan(index=0, start_ms=300_000, end_ms=600_000)

    command = build_slice_command(source, output, plan)

    assert command[:5] == ["-y", "-ss", "300.000", "-t", "300.000"]
    assert command.index("-ss") < command.index("-i")
    assert command.index("-t") < command.index("-i")
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[command.index("-f") + 1] == "wav"
    assert command[-1] == str(output)


@pytest.mark.unit
def test_build_slice_command_honours_the_sample_rate(tmp_path: Path) -> None:
    plan = ChunkPlan(index=1, start_ms=1000, end_ms=3500)

    command = build_slice_command(
        tmp_path / "audio.wav",
        tmp_path / "out.wav",
        plan,
        sample_rate=48000,
    )

    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ss") + 1] == "1.000"
    assert command[command.index("-t") + 1] == "2.500"


@pytest.mark.media
def test_slice_chunk_is_sample_accurate(ffmpeg_binary: Path, tmp_path: Path) -> None:
    source = _write_wav(tmp_path / "audio.wav", seconds=3.0)
    plan = ChunkPlan(index=0, start_ms=1000, end_ms=3000)
    output = chunk_path(tmp_path / "chunks", 0)

    result = slice_chunk(FFmpegRunner(ffmpeg_binary), source, output, plan)

    assert result.returncode == 0, result.stderr_tail
    assert output.is_file()
    assert wav_duration_ms(output) == 2000

    sliced = read_audio_16k(output)
    original = read_audio_16k(source)
    expected = original[16_000:48_000]
    assert sliced.shape == expected.shape
    assert float(np.max(np.abs(sliced - expected))) == 0.0

    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


@pytest.mark.unit
def test_cleanup_chunks_removes_only_chunk_wavs_and_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "chunks"
    directory.mkdir()
    first = _write_wav(directory / chunk_filename(0), seconds=0.1)
    second = _write_wav(directory / chunk_filename(1), seconds=0.1)
    keep = _write_wav(directory / "audio.wav", seconds=0.1)

    cleanup_chunks(directory)

    assert not first.exists()
    assert not second.exists()
    assert keep.is_file()

    cleanup_chunks(directory)
    cleanup_chunks(tmp_path / "missing")


@pytest.mark.unit
def test_remove_file_is_idempotent(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / chunk_filename(0), seconds=0.1)

    remove_file(path)
    remove_file(path)

    assert not path.exists()
