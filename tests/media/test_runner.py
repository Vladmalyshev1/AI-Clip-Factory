"""Tests for the cancellable FFmpeg runner.

Parsing is pure and marked ``unit``; the remaining tests drive the real pinned
FFmpeg build and are marked ``media``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from app.core.errors import ErrorCode
from app.media.runner import (
    GLOBAL_FLAGS,
    FFmpegRunner,
    ProgressUpdate,
    RenderCancelled,
    RenderError,
    RenderResult,
    parse_progress_line,
)


@pytest.mark.unit
def test_parse_progress_line_reads_out_time_us_as_microseconds() -> None:
    update = parse_progress_line("out_time_us=1500000")
    assert isinstance(update, ProgressUpdate)
    assert update.out_time_ms == 1_500_000
    assert update.speed is None
    assert update.fraction == 0.0


@pytest.mark.unit
def test_parse_progress_line_reads_out_time_ms_as_microseconds() -> None:
    # FFmpeg's ``out_time_ms`` key also carries microseconds.
    update = parse_progress_line("out_time_ms=250000")
    assert update is not None
    assert update.out_time_ms == 250_000


@pytest.mark.unit
def test_parse_progress_line_reads_speed() -> None:
    update = parse_progress_line("speed=1.25x")
    assert update is not None
    assert update.speed == "1.25x"
    assert update.out_time_ms is None


@pytest.mark.unit
def test_parse_progress_line_treats_missing_speed_as_none() -> None:
    update = parse_progress_line("speed=N/A")
    assert update is not None
    assert update.speed is None


@pytest.mark.unit
def test_parse_progress_line_marks_the_end() -> None:
    update = parse_progress_line("progress=end")
    assert update is not None
    assert update.fraction == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    ["", "   ", "garbage", "frame=12", "out_time_us=abc", "out_time_us=-1", "progress=continue"],
)
def test_parse_progress_line_ignores_noise(line: str) -> None:
    assert parse_progress_line(line) is None


def _encode_args(destination: Path, *, seconds: str = "2", realtime: bool = False) -> list[str]:
    args: list[str] = []
    if realtime:
        args.append("-re")
    args.extend(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-t",
            seconds,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(destination),
        ]
    )
    return args


@pytest.mark.media
def test_run_reports_monotonic_progress_ending_at_one(
    ffmpeg_binary: Path, tmp_path: Path
) -> None:
    runner = FFmpegRunner(ffmpeg_binary)
    fractions: list[float] = []

    result = runner.run(
        _encode_args(tmp_path / "progress.mp4"),
        duration_ms=2000,
        on_progress=fractions.append,
    )

    assert isinstance(result, RenderResult)
    assert result.returncode == 0
    assert result.elapsed_s > 0
    assert fractions, "the runner must report progress for a 2 s encode"
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in fractions)
    # Throttled to roughly 10 callbacks per second.
    assert len(fractions) <= 40
    assert result.command[: 1 + len(GLOBAL_FLAGS)] == [str(ffmpeg_binary), *GLOBAL_FLAGS]


@pytest.mark.media
def test_run_raises_render_error_with_stderr_tail_for_a_missing_input(
    ffmpeg_binary: Path, tmp_path: Path
) -> None:
    runner = FFmpegRunner(ffmpeg_binary, log_dir=tmp_path)

    with pytest.raises(RenderError) as excinfo:
        runner.run(["-i", str(tmp_path / "missing.mp4"), "-y", str(tmp_path / "out.mp4")])

    error = excinfo.value
    assert error.code is ErrorCode.RENDER_FAILED
    assert error.returncode is not None
    assert error.returncode != 0
    assert error.stderr_tail
    assert error.command[: 1 + len(GLOBAL_FLAGS)] == [str(ffmpeg_binary), *GLOBAL_FLAGS]
    assert list(tmp_path.glob("ffmpeg-stderr-*.log")) == []


@pytest.mark.media
def test_run_stops_quickly_when_cancelled(ffmpeg_binary: Path, tmp_path: Path) -> None:
    runner = FFmpegRunner(ffmpeg_binary)
    cancel_event = threading.Event()
    timer = threading.Timer(0.3, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RenderCancelled):
            runner.run(
                _encode_args(tmp_path / "cancelled.mp4", seconds="30", realtime=True),
                duration_ms=30_000,
                cancel_event=cancel_event,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.0


@pytest.mark.media
def test_run_passes_shell_metacharacters_verbatim(ffmpeg_binary: Path, tmp_path: Path) -> None:
    target = tmp_path / "out put & ;.mp4"
    runner = FFmpegRunner(ffmpeg_binary)

    result = runner.run(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=5",
            "-t",
            "0.2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            'title=a "quoted" & value',
            "-y",
            str(target),
        ]
    )

    assert result.returncode == 0
    assert target.is_file()
    assert result.command[-1] == str(target)
    assert 'title=a "quoted" & value' in result.command
    assert isinstance(result.command, list)
    assert all(isinstance(part, str) for part in result.command)


@pytest.mark.media
def test_run_raises_render_timeout(ffmpeg_binary: Path, tmp_path: Path) -> None:
    runner = FFmpegRunner(ffmpeg_binary)
    started = time.monotonic()

    with pytest.raises(RenderError) as excinfo:
        runner.run(
            _encode_args(tmp_path / "timeout.mp4", seconds="30", realtime=True),
            timeout_s=0.5,
        )

    assert excinfo.value.code is ErrorCode.RENDER_TIMEOUT
    assert excinfo.value.details["timeout_s"] == 0.5
    assert time.monotonic() - started < 5.0


@pytest.mark.media
def test_from_settings_uses_the_resolved_binary(settings) -> None:
    runner = FFmpegRunner.from_settings(settings)
    assert runner.ffmpeg_path.is_file()
    assert runner.log_dir == settings.logs_dir
