"""Tests for the clip command builder and the real vertical render.

Command construction is pure and marked ``unit``; the end-to-end render drives
the pinned FFmpeg build (graph + libass burn-in) and is marked ``media``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.media.ass import SubtitleCue, build_ass_document, write_ass_file
from app.media.cut import ClipSpec, build_clip_command, render_clip
from app.media.graph import VerticalLayout, escape_filter_path
from app.media.probe import probe_media
from app.media.runner import FFmpegRunner

START_MS = 90_500
END_MS = 120_500


def _make_spec(tmp_path: Path, **overrides: object) -> ClipSpec:
    params: dict[str, object] = {
        "source_path": tmp_path / "source.mp4",
        "output_path": tmp_path / "clip.mp4",
        "start_ms": START_MS,
        "end_ms": END_MS,
        **overrides,
    }
    return ClipSpec(**params)  # type: ignore[arg-type]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.unit
def test_command_uses_exact_preroll_seek_and_duration(tmp_path: Path) -> None:
    command = build_clip_command(_make_spec(tmp_path))

    assert command[:6] == [
        "-ss",
        "90.500",
        "-i",
        str(tmp_path / "source.mp4"),
        "-t",
        "30.000",
    ]
    assert command.index("-ss") < command.index("-i")
    assert command.count("-ss") == 1
    assert command.count("-t") == 1


@pytest.mark.unit
def test_command_has_the_filtergraph_and_output_maps(tmp_path: Path) -> None:
    command = build_clip_command(_make_spec(tmp_path))

    graph = _value_after(command, "-filter_complex")
    assert "scale=1080:1920" in graph
    assert graph.endswith("format=yuv420p[vout];[vout]null[vfinal]")
    assert command.count("-map") == 2
    assert command[command.index("-map") + 1] == "[vfinal]"
    assert command[command.index("-map") + 3] == "0:a?"
    assert "-map" in command
    assert "[vfinal]" in command


@pytest.mark.unit
def test_command_encode_settings_are_present_in_order(tmp_path: Path) -> None:
    command = build_clip_command(_make_spec(tmp_path))

    assert _value_after(command, "-c:v") == "libx264"
    assert _value_after(command, "-preset") == "medium"
    assert _value_after(command, "-crf") == "19"
    assert _value_after(command, "-pix_fmt") == "yuv420p"
    assert _value_after(command, "-profile:v") == "high"
    assert _value_after(command, "-level") == "4.1"
    assert _value_after(command, "-movflags") == "+faststart"
    assert _value_after(command, "-c:a") == "aac"
    assert _value_after(command, "-b:a") == "128k"

    order = ["-c:v", "-preset", "-crf", "-pix_fmt", "-profile:v", "-level", "-movflags"]
    positions = [command.index(flag) for flag in order]
    assert positions == sorted(positions)
    assert command[-2:] == ["-y", str(tmp_path / "clip.mp4")]


@pytest.mark.unit
def test_command_burns_subtitles_with_an_escaped_path(tmp_path: Path) -> None:
    subtitles = tmp_path / "subs dir" / "clip captions.ass"
    command = build_clip_command(_make_spec(tmp_path, subtitles_path=subtitles))

    graph = _value_after(command, "-filter_complex")
    escaped = escape_filter_path(subtitles)
    assert f"ass={escaped}" in graph
    assert "ass='" not in graph
    assert graph.count("ass=") == 1
    assert graph.endswith("[vfinal]")
    assert command[command.index("-map") + 1] == "[vfinal]"


@pytest.mark.unit
def test_command_escapes_an_apostrophe_in_the_subtitle_path(tmp_path: Path) -> None:
    subtitles = tmp_path / "it's" / "clip's captions.ass"
    command = build_clip_command(_make_spec(tmp_path, subtitles_path=subtitles))

    graph = _value_after(command, "-filter_complex")
    assert "ass='" not in graph
    assert f"ass={escape_filter_path(subtitles)}[vfinal]" in graph


@pytest.mark.unit
def test_command_without_subtitles_uses_a_null_final_label(tmp_path: Path) -> None:
    command = build_clip_command(_make_spec(tmp_path))

    graph = _value_after(command, "-filter_complex")
    assert "[vout]null[vfinal]" in graph
    assert "ass=" not in graph
    assert command[command.index("-map") + 1] == "[vfinal]"


@pytest.mark.unit
def test_command_keeps_metacharacter_paths_as_single_arguments(tmp_path: Path) -> None:
    source = tmp_path / "src & ; 'quoted'.mp4"
    output = tmp_path / 'out & ; "quoted" with spaces.mp4'

    command = build_clip_command(
        _make_spec(tmp_path, source_path=source, output_path=output)
    )

    assert isinstance(command, list)
    assert all(isinstance(part, str) for part in command)
    assert command[-1] == str(output)
    assert command.count(str(output)) == 1
    assert sum(str(output) in part for part in command) == 1
    assert command[command.index("-i") + 1] == str(source)
    assert str(output) not in " ".join(command[:-1])


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"start_ms": -1},
        {"start_ms": 1000, "end_ms": 1000},
        {"start_ms": 2000, "end_ms": 1000},
        {"crf": -1},
        {"crf": 52},
        {"preset": ""},
        {"audio_bitrate": ""},
        {"video_codec": ""},
    ],
)
def test_clip_spec_rejects_invalid_arguments(tmp_path: Path, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _make_spec(tmp_path, **overrides)


@pytest.mark.unit
def test_clip_spec_rejects_the_source_as_output(tmp_path: Path) -> None:
    source = tmp_path / "recording.mp4"
    with pytest.raises(ValueError):
        _make_spec(tmp_path, source_path=source, output_path=source)
    with pytest.raises(ValueError):
        _make_spec(tmp_path, source_path=source, output_path=tmp_path / "sub" / ".." / source.name)


@pytest.mark.unit
def test_clip_spec_accepts_custom_render_settings(tmp_path: Path) -> None:
    spec = _make_spec(
        tmp_path,
        layout=VerticalLayout(width=270, height=480),
        crf=0,
        preset="ultrafast",
    )
    command = build_clip_command(spec)
    assert "crop=270:480" in _value_after(command, "-filter_complex")
    assert _value_after(command, "-crf") == "0"
    assert _value_after(command, "-preset") == "ultrafast"


@pytest.mark.media
def test_render_clip_with_burned_subtitles_end_to_end(
    sample_video: Path,
    ffmpeg_binary: Path,
    ffprobe_binary: Path,
    tmp_path: Path,
) -> None:
    subtitles = tmp_path / "subs" / "clip.ass"
    document = build_ass_document(
        [
            SubtitleCue("first caption", 100, 450),
            SubtitleCue("second caption", 500, 950),
        ],
        video_width=270,
        video_height=480,
    )
    write_ass_file(subtitles, document)
    output = tmp_path / "vertical.mp4"
    spec = ClipSpec(
        source_path=sample_video,
        output_path=output,
        start_ms=0,
        end_ms=1000,
        subtitles_path=subtitles,
        layout=VerticalLayout(width=270, height=480),
        preset="ultrafast",
    )

    result = render_clip(FFmpegRunner(ffmpeg_binary), spec)

    assert result.returncode == 0
    assert output.is_file()
    info = probe_media(output, ffprobe_path=ffprobe_binary)
    assert (info.width, info.height) == (270, 480)
    assert 900 <= info.duration_ms <= 1500
    assert info.has_audio is True
    assert info.video_codec == "h264"
    assert output.stat().st_size > 10 * 1024


@pytest.mark.media
def test_render_clip_burns_subtitles_from_a_path_with_an_apostrophe(
    sample_video: Path,
    ffmpeg_binary: Path,
    tmp_path: Path,
) -> None:
    """Regression: FFmpeg used to fail when the ASS path contained ``'``.

    The old quoted form (``ass='C\\:/.../it\\'s.ass'``) silently dropped the
    apostrophe and libass reported ``fopen failed``. The path is now escaped
    for both parser levels instead of quoted.
    """
    subtitles = tmp_path / "it's" / "clip's captions.ass"
    write_ass_file(
        subtitles,
        build_ass_document(
            [SubtitleCue("it's a caption", 100, 900)],
            video_width=270,
            video_height=480,
        ),
    )
    output = tmp_path / "rendered it's.mp4"
    spec = ClipSpec(
        source_path=sample_video,
        output_path=output,
        start_ms=0,
        end_ms=1000,
        subtitles_path=subtitles,
        layout=VerticalLayout(width=270, height=480),
        preset="ultrafast",
    )

    result = render_clip(FFmpegRunner(ffmpeg_binary), spec)

    assert result.returncode == 0, result.stderr_tail
    assert output.is_file()
    assert output.stat().st_size > 10 * 1024
