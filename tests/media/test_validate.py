"""Tests for rendered-clip validation.

Checks that only need synthetic :class:`MediaInfo` values are marked ``unit``;
checks that run the pinned ffprobe against real sample files are ``media``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from app.core.errors import ErrorCode
from app.media import validate
from app.media.probe import MediaInfo, ProbeError
from app.media.validate import (
    DEFAULT_ASPECT_RATIO,
    RenderValidationError,
    validate_aspect_ratio,
    validate_render,
)


def _info(**overrides: Any) -> MediaInfo:
    params: dict[str, Any] = {
        "path": "clip.mp4",
        "duration_ms": 1000,
        "width": 1080,
        "height": 1920,
        "fps": 30.0,
        "video_codec": "h264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "has_audio": True,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "bitrate": 4_000_000,
        "size_bytes": 500_000,
    }
    params.update(overrides)
    return MediaInfo(**params)


@pytest.mark.media
def test_generated_sample_passes_and_reuses_the_probe_result(
    sample_video: Path,
    ffprobe_binary: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[MediaInfo] = []
    real_probe = validate.probe_media

    def spy(*args: Any, **kwargs: Any) -> MediaInfo:
        info = real_probe(*args, **kwargs)
        probed.append(info)
        return info

    monkeypatch.setattr(validate, "probe_media", spy)

    info = validate_render(
        sample_video,
        expected_duration_ms=1000,
        expected_width=320,
        expected_height=240,
        expect_audio=True,
        ffprobe_path=ffprobe_binary,
    )

    assert isinstance(info, MediaInfo)
    assert len(probed) == 1
    assert info is probed[0]
    assert (info.width, info.height) == (320, 240)
    assert 900 <= info.duration_ms <= 1500
    assert info.has_audio is True


@pytest.mark.media
def test_wrong_expected_duration_reports_both_values(
    sample_video: Path, ffprobe_binary: Path
) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(
            sample_video,
            expected_duration_ms=5000,
            ffprobe_path=ffprobe_binary,
        )

    error = excinfo.value
    assert error.code is ErrorCode.RENDER_FAILED
    assert error.details["expected_ms"] == 5000
    assert 900 <= error.details["actual_ms"] <= 1500
    assert error.details["delta_ms"] > 750


@pytest.mark.media
def test_wrong_expected_width_fails(sample_video: Path, ffprobe_binary: Path) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(sample_video, expected_width=1080, ffprobe_path=ffprobe_binary)
    assert excinfo.value.details == {"expected_width": 1080, "actual_width": 320}


@pytest.mark.media
def test_wrong_expected_height_fails(sample_video: Path, ffprobe_binary: Path) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(sample_video, expected_height=1920, ffprobe_path=ffprobe_binary)
    assert excinfo.value.details == {"expected_height": 1920, "actual_height": 240}


@pytest.mark.media
def test_missing_audio_is_rejected_for_a_silent_video(
    silent_video: Path, ffprobe_binary: Path
) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(silent_video, expect_audio=True, ffprobe_path=ffprobe_binary)
    assert excinfo.value.details["reason"] == "missing_audio"

    info = validate_render(
        silent_video,
        expect_audio=False,
        expected_width=160,
        expected_height=120,
        ffprobe_path=ffprobe_binary,
    )
    assert info.has_audio is False


@pytest.mark.unit
def test_missing_file_is_rejected_before_probing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.mp4"

    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(missing)

    error = excinfo.value
    assert error.code is ErrorCode.RENDER_FAILED
    assert error.details["reason"] == "missing"
    assert error.details["path"] == str(missing)


@pytest.mark.media
def test_non_media_file_is_wrapped_as_a_validation_error(
    broken_media_file: Path, ffprobe_binary: Path
) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(broken_media_file, ffprobe_path=ffprobe_binary)

    error = excinfo.value
    assert isinstance(error.__cause__, ProbeError)
    assert error.details["reason"] == "probe_failed"
    assert error.details["cause"] in {
        str(ErrorCode.PROBE_FAILED),
        str(ErrorCode.UNSUPPORTED_MEDIA),
    }
    assert error.details["probe_details"]


@pytest.mark.unit
def test_tolerance_below_the_duration_delta_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered = tmp_path / "clip.mp4"
    rendered.write_bytes(b"not empty")
    synthetic = _info(path=str(rendered), duration_ms=900, size_bytes=9)
    monkeypatch.setattr(validate, "probe_media", lambda *args, **kwargs: synthetic)

    with pytest.raises(RenderValidationError) as excinfo:
        validate_render(rendered, expected_duration_ms=1000, tolerance_ms=50)

    details = excinfo.value.details
    assert details["expected_ms"] == 1000
    assert details["actual_ms"] == 900
    assert details["delta_ms"] == 100
    assert details["tolerance_ms"] == 50


@pytest.mark.unit
def test_tolerance_above_the_duration_delta_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered = tmp_path / "clip.mp4"
    rendered.write_bytes(b"not empty")
    synthetic = _info(path=str(rendered), duration_ms=900, size_bytes=9)
    monkeypatch.setattr(validate, "probe_media", lambda *args, **kwargs: synthetic)

    info = validate_render(rendered, expected_duration_ms=1000, tolerance_ms=150)

    assert info is synthetic


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height"), [(1080, 1920), (270, 480), (720, 1280)])
def test_aspect_ratio_accepts_vertical_9_by_16(width: int, height: int) -> None:
    info = _info(width=width, height=height, audio_codec=None, has_audio=False)
    validate_aspect_ratio(info)
    assert info.aspect_ratio == pytest.approx(DEFAULT_ASPECT_RATIO)


@pytest.mark.unit
def test_aspect_ratio_rejects_a_landscape_frame() -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_aspect_ratio(_info(width=320, height=240))

    details = excinfo.value.details
    assert details["width"] == 320
    assert details["height"] == 240
    assert details["actual_ratio"] == pytest.approx(4 / 3)


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height"), [(0, 1920), (1080, 0), (0, 0), (-2, 480)])
def test_aspect_ratio_rejects_zero_or_negative_dimensions(width: int, height: int) -> None:
    with pytest.raises(RenderValidationError) as excinfo:
        validate_aspect_ratio(_info(width=width, height=height))

    assert excinfo.value.details == {"width": width, "height": height}
