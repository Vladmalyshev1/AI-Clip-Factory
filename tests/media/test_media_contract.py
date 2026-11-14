"""Contract tests for the shared media layer (binary resolution + probing).

These tests use the real pinned FFmpeg build; they are the guarantee that the
ingest service and the render stages see the same binaries and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.media import binaries, probe
from app.media.binaries import (
    find_ffmpeg,
    find_ffprobe,
    require_ffmpeg,
    require_ffprobe,
    run_command,
)
from app.media.probe import MediaInfo, ProbeError, probe_media

pytestmark = pytest.mark.media


def test_find_ffmpeg_and_ffprobe_use_the_same_directory(settings: Settings) -> None:
    ffmpeg = find_ffmpeg(settings)
    assert ffmpeg is not None
    assert ffmpeg.is_file()

    ffprobe = find_ffprobe(settings, ffmpeg_path=ffmpeg)
    assert ffprobe is not None
    assert ffprobe.is_file()
    assert ffprobe.parent == ffmpeg.parent


def test_configured_ffmpeg_path_wins(settings: Settings, ffmpeg_binary: Path) -> None:
    configured = settings.model_copy(update={"ffmpeg_path": str(ffmpeg_binary)})
    assert find_ffmpeg(configured) == ffmpeg_binary


def test_missing_configured_path_falls_back_to_bundled_build(
    settings: Settings, tmp_path: Path
) -> None:
    configured = settings.model_copy(update={"ffmpeg_path": str(tmp_path / "nope.exe")})
    resolved = find_ffmpeg(configured)
    assert resolved is not None
    assert resolved.is_file()
    assert resolved != tmp_path / "nope.exe"


def test_require_ffmpeg_raises_a_typed_error(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(binaries, "BUNDLED_BIN_DIR", tmp_path / "missing-bin")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    unconfigured = settings.model_copy(update={"ffmpeg_path": None})

    with pytest.raises(AppError) as excinfo:
        require_ffmpeg(unconfigured)
    assert excinfo.value.code is ErrorCode.FFMPEG_NOT_FOUND
    assert excinfo.value.status_code == 500

    with pytest.raises(AppError) as probe_excinfo:
        require_ffprobe(unconfigured)
    assert probe_excinfo.value.code is ErrorCode.FFPROBE_NOT_FOUND


def test_run_command_reports_failures_without_raising() -> None:
    result = run_command(["definitely-not-an-executable-acf"], timeout=5.0)
    assert result.returncode != 0
    assert result.ok is False
    assert result.output


def test_probe_media_reads_container_and_stream_metadata(
    sample_video: Path, ffprobe_binary: Path
) -> None:
    info = probe_media(sample_video, ffprobe_path=ffprobe_binary)

    assert isinstance(info, MediaInfo)
    assert info.path == str(sample_video)
    assert (info.width, info.height) == (320, 240)
    assert 8.0 <= info.fps <= 12.0
    assert info.video_codec == "h264"
    assert info.pix_fmt == "yuv420p"
    assert info.audio_codec == "aac"
    assert info.has_audio is True
    assert info.sample_rate == 44100
    assert info.channels >= 1
    assert 900 <= info.duration_ms <= 1500
    assert info.size_bytes == sample_video.stat().st_size
    assert "mp4" in info.container
    assert info.is_landscape is True
    assert info.is_vertical is False
    assert info.aspect_ratio == pytest.approx(320 / 240)


def test_probe_media_reports_a_video_without_audio(
    silent_video: Path, ffprobe_binary: Path
) -> None:
    info = probe_media(silent_video, ffprobe_path=ffprobe_binary)
    assert info.has_audio is False
    assert info.audio_codec is None
    assert info.sample_rate is None
    assert (info.width, info.height) == (160, 120)


def test_probe_media_resolves_ffprobe_from_settings(
    sample_video: Path, settings: Settings
) -> None:
    info = probe_media(sample_video, settings=settings)
    assert info.duration_ms > 0
    assert info.video_codec == "h264"


def test_probe_media_rejects_a_missing_file(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(ProbeError) as excinfo:
        probe_media(tmp_path / "absent.mp4", settings=settings)
    assert excinfo.value.code is ErrorCode.SOURCE_NOT_FOUND


def test_probe_media_rejects_a_non_media_file(
    broken_media_file: Path, ffprobe_binary: Path
) -> None:
    with pytest.raises(ProbeError) as excinfo:
        probe_media(broken_media_file, ffprobe_path=ffprobe_binary)
    assert excinfo.value.code in {ErrorCode.PROBE_FAILED, ErrorCode.UNSUPPORTED_MEDIA}
    assert excinfo.value.details


def test_probe_media_raises_when_ffprobe_is_missing(
    sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "find_ffprobe", lambda _settings=None: None)
    with pytest.raises(ProbeError) as excinfo:
        probe_media(sample_video)
    assert excinfo.value.code is ErrorCode.FFPROBE_NOT_FOUND


def test_media_info_serialises_to_json(sample_video: Path, ffprobe_binary: Path) -> None:
    payload = probe_media(sample_video, ffprobe_path=ffprobe_binary).to_dict()
    encoded = json.dumps(payload)
    assert json.loads(encoded)["width"] == 320
    assert set(payload) == {
        "path",
        "duration_ms",
        "width",
        "height",
        "fps",
        "video_codec",
        "pix_fmt",
        "audio_codec",
        "has_audio",
        "container",
        "bitrate",
        "size_bytes",
        "rotation",
        "sample_rate",
        "channels",
    }
