"""Render modes added in M6: original/vertical, crop/pad, accurate/fast."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from app.media.cut import (
    ClipSpec,
    CropMode,
    EncodingMode,
    OutputFormat,
    build_clip_command,
    render_clip,
)
from app.media.graph import VerticalLayout
from app.media.probe import probe_media
from app.media.runner import FFmpegRunner, RenderResult
from app.media.validate import validate_aspect_ratio, validate_render

SMALL_LAYOUT = VerticalLayout(width=270, height=480)


def _spec(tmp_path: Path, **overrides: object) -> ClipSpec:
    params: dict[str, object] = {
        "source_path": tmp_path / "source.mp4",
        "output_path": tmp_path / "clip.mp4",
        "start_ms": 100,
        "end_ms": 900,
        **overrides,
    }
    return ClipSpec(**params)  # type: ignore[arg-type]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.unit
def test_original_format_maps_the_source_streams_without_a_filtergraph(
    tmp_path: Path,
) -> None:
    command = build_clip_command(_spec(tmp_path, output_format=OutputFormat.ORIGINAL))

    assert "-filter_complex" not in command
    assert _value_after(command, "-map") == "0:v:0"
    assert "0:a?" in command
    assert _value_after(command, "-c:v") == "libx264"
    assert _value_after(command, "-c:a") == "aac"
    assert "crop" not in " ".join(command)


@pytest.mark.unit
def test_vertical_crop_fills_the_canvas_without_blur(tmp_path: Path) -> None:
    command = build_clip_command(
        _spec(tmp_path, layout=SMALL_LAYOUT, crop_mode=CropMode.CROP)
    )

    graph = _value_after(command, "-filter_complex")
    assert "scale=270:480:force_original_aspect_ratio=increase,crop=270:480" in graph
    assert "boxblur" not in graph
    assert _value_after(command, "-map") == "[vfinal]"


@pytest.mark.unit
def test_vertical_pad_keeps_the_blurred_background(tmp_path: Path) -> None:
    graph = _value_after(
        build_clip_command(_spec(tmp_path, layout=SMALL_LAYOUT)), "-filter_complex"
    )
    assert "boxblur" in graph
    assert "force_original_aspect_ratio=decrease" in graph


@pytest.mark.unit
def test_fast_mode_stream_copies_the_original_format(tmp_path: Path) -> None:
    command = build_clip_command(
        _spec(tmp_path, output_format=OutputFormat.ORIGINAL, encoding_mode=EncodingMode.FAST)
    )

    assert "-filter_complex" not in command
    assert _value_after(command, "-c") == "copy"
    assert "-c:v" not in command
    assert "-crf" not in command


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"encoding_mode": EncodingMode.FAST},
        {
            "encoding_mode": EncodingMode.FAST,
            "output_format": OutputFormat.ORIGINAL,
            "subtitles_path": Path("clip.ass"),
        },
        {"output_format": OutputFormat.ORIGINAL, "subtitles_path": Path("clip.ass")},
        {"encoding_mode": "turbo"},
        {"output_format": "square"},
        {"crop_mode": "stretch"},
    ],
)
def test_clip_spec_rejects_impossible_mode_combinations(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        _spec(tmp_path, **overrides)


@pytest.mark.unit
def test_render_clip_forwards_cancellation_and_timeout(tmp_path: Path) -> None:
    received: dict[str, object] = {}

    class RecordingRunner:
        def run(self, args: list[str], **kwargs: object) -> RenderResult:
            received.update(kwargs)
            received["args"] = args
            return RenderResult(returncode=0, elapsed_s=0.0, stderr_tail="", command=args)

    event = threading.Event()
    cancel_check = lambda: None  # noqa: E731 - trivial callback
    render_clip(
        RecordingRunner(),  # type: ignore[arg-type]
        _spec(tmp_path),
        cancel_event=event,
        cancel_check=cancel_check,
        timeout_s=12.5,
    )

    assert received["cancel_event"] is event
    assert received["cancel_check"] is cancel_check
    assert received["timeout_s"] == 12.5
    assert received["duration_ms"] == 800


@pytest.mark.media
def test_original_accurate_render_keeps_the_source_geometry(
    sample_video: Path,
    ffmpeg_binary: Path,
    ffprobe_binary: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "original.mp4"
    render_clip(
        FFmpegRunner(ffmpeg_binary),
        ClipSpec(
            source_path=sample_video,
            output_path=output,
            start_ms=100,
            end_ms=900,
            output_format=OutputFormat.ORIGINAL,
            preset="ultrafast",
        ),
    )

    info = validate_render(
        output, expected_duration_ms=800, expected_width=320, expected_height=240,
        ffprobe_path=ffprobe_binary,
    )
    assert info.video_codec == "h264"
    assert info.fps == pytest.approx(10.0)
    assert info.has_audio is True
    assert info.size_bytes > 1024


@pytest.mark.media
def test_vertical_crop_render_is_9_16(
    sample_video: Path, ffmpeg_binary: Path, ffprobe_binary: Path, tmp_path: Path
) -> None:
    output = tmp_path / "crop.mp4"
    render_clip(
        FFmpegRunner(ffmpeg_binary),
        ClipSpec(
            source_path=sample_video,
            output_path=output,
            start_ms=0,
            end_ms=1000,
            layout=SMALL_LAYOUT,
            crop_mode=CropMode.CROP,
            preset="ultrafast",
        ),
    )

    info = validate_render(
        output, expected_width=270, expected_height=480, ffprobe_path=ffprobe_binary
    )
    validate_aspect_ratio(info, target_ratio=9 / 16)
    assert info.has_audio is True


@pytest.mark.media
def test_vertical_pad_render_is_9_16(
    sample_video: Path, ffmpeg_binary: Path, ffprobe_binary: Path, tmp_path: Path
) -> None:
    output = tmp_path / "pad.mp4"
    render_clip(
        FFmpegRunner(ffmpeg_binary),
        ClipSpec(
            source_path=sample_video,
            output_path=output,
            start_ms=0,
            end_ms=1000,
            layout=SMALL_LAYOUT,
            crop_mode=CropMode.PAD,
            preset="ultrafast",
        ),
    )

    info = probe_media(output, ffprobe_path=ffprobe_binary)
    assert (info.width, info.height) == (270, 480)
    assert info.video_codec == "h264"


@pytest.mark.media
def test_fast_render_stream_copies_an_original_clip(
    sample_video: Path, ffmpeg_binary: Path, ffprobe_binary: Path, tmp_path: Path
) -> None:
    output = tmp_path / "fast.mp4"
    render_clip(
        FFmpegRunner(ffmpeg_binary),
        ClipSpec(
            source_path=sample_video,
            output_path=output,
            start_ms=0,
            end_ms=900,
            output_format=OutputFormat.ORIGINAL,
            encoding_mode=EncodingMode.FAST,
        ),
    )

    info = probe_media(output, ffprobe_path=ffprobe_binary)
    # Stream copy snaps to keyframes, so only the geometry and codec are fixed.
    assert (info.width, info.height) == (320, 240)
    assert info.video_codec == "h264"
    assert 0 < info.duration_ms <= 1500
    assert info.has_audio is True
    assert output.stat().st_size > 1024


@pytest.mark.media
def test_accurate_render_tolerates_a_source_without_audio(
    silent_video: Path, ffmpeg_binary: Path, ffprobe_binary: Path, tmp_path: Path
) -> None:
    output = tmp_path / "silent-clip.mp4"
    render_clip(
        FFmpegRunner(ffmpeg_binary),
        ClipSpec(
            source_path=silent_video,
            output_path=output,
            start_ms=0,
            end_ms=900,
            output_format=OutputFormat.ORIGINAL,
            preset="ultrafast",
        ),
    )

    info = validate_render(output, expect_audio=False, ffprobe_path=ffprobe_binary)
    assert info.has_audio is False
