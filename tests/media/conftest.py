"""Fixtures for media tests: real FFmpeg-generated sample files.

The files are tiny (1 second, 320x240) so the whole media suite stays fast.
They are generated once per session with the pinned FFmpeg build.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.core.config import Settings
from app.media.binaries import find_ffmpeg


def _ffmpeg() -> Path:
    binary = find_ffmpeg(Settings(_env_file=None))
    if binary is None:
        pytest.skip("ffmpeg is not available in this environment")
    return binary


def _generate(destination: Path, *args: str) -> Path:
    result = subprocess.run(
        [str(_ffmpeg()), "-hide_banner", "-loglevel", "error", "-y", *args, str(destination)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - environment problem
        raise RuntimeError(f"ffmpeg sample generation failed: {result.stderr[-1000:]}")
    return destination


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-second H.264/AAC MP4, 320x240 at 10 fps."""
    directory = tmp_path_factory.mktemp("media-samples")
    return _generate(
        directory / "sample.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "1.0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
    )


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-second video-only MP4 (no audio stream)."""
    directory = tmp_path_factory.mktemp("media-samples-silent")
    return _generate(
        directory / "silent.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=5",
        "-t",
        "1.0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-an",
    )


@pytest.fixture()
def broken_media_file(tmp_path: Path) -> Iterator[Path]:
    """A text file with a media extension, i.e. an unreadable "video"."""
    path = tmp_path / "broken.mp4"
    path.write_text("this is definitely not an mp4 file", encoding="utf-8")
    yield path


@pytest.fixture()
def ffmpeg_binary() -> Path:
    """The resolved ffmpeg executable (skips when absent)."""
    return _ffmpeg()


@pytest.fixture()
def ffprobe_binary(ffmpeg_binary: Path) -> Path:
    """ffprobe next to the resolved ffmpeg (skips when absent)."""
    candidate = ffmpeg_binary.with_name(
        "ffprobe.exe" if ffmpeg_binary.suffix == ".exe" else "ffprobe"
    )
    if not candidate.is_file():
        found = shutil.which("ffprobe")
        if not found:  # pragma: no cover - environment problem
            pytest.skip("ffprobe is not available in this environment")
        return Path(found)
    return candidate
