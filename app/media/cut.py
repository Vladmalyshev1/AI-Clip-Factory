"""Clip command builder and executor for original and vertical renders (M6).

:func:`build_clip_command` returns only the arguments **after** FFmpeg's global
flags; :class:`app.media.runner.FFmpegRunner` prepends
``-hide_banner -nostdin -loglevel error -progress pipe:1 -nostats``. Keeping the
split here means the progress plumbing lives in exactly one place.

Three independent switches describe a render:

* :class:`OutputFormat` — keep the source geometry (``original``) or fit it on a
  vertical canvas (``vertical``, default);
* :class:`CropMode` — for ``vertical`` only: fill the canvas and cut the sides
  (``crop``) or keep the whole frame on a blurred background (``pad``);
* :class:`EncodingMode` — ``accurate`` re-encodes at the exact requested
  timestamps; ``fast`` stream-copies the original format, so the cut snaps to
  the nearest keyframe and the duration is approximate by design.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from app.media.graph import VerticalLayout, build_clip_filtergraph
from app.media.runner import FFmpegRunner, RenderResult

DEFAULT_CRF: Final = 19
DEFAULT_PRESET: Final = "medium"
DEFAULT_AUDIO_BITRATE: Final = "128k"
DEFAULT_VIDEO_CODEC: Final = "libx264"
MIN_CRF: Final = 0
MAX_CRF: Final = 51


class EncodingMode(StrEnum):
    """How the clip is cut."""

    ACCURATE = "accurate"
    FAST = "fast"


class OutputFormat(StrEnum):
    """Frame geometry of the rendered clip."""

    ORIGINAL = "original"
    VERTICAL = "vertical"


class CropMode(StrEnum):
    """How a vertical clip fills its 9:16 canvas (ignored for ``original``)."""

    CROP = "crop"
    PAD = "pad"


@dataclass(frozen=True)
class ClipSpec:
    """Everything needed to render one clip from a source recording."""

    source_path: Path
    output_path: Path
    start_ms: int
    end_ms: int
    subtitles_path: Path | None = None
    layout: VerticalLayout = field(default_factory=VerticalLayout)
    encoding_mode: EncodingMode = EncodingMode.ACCURATE
    output_format: OutputFormat = OutputFormat.VERTICAL
    crop_mode: CropMode = CropMode.PAD
    crf: int = DEFAULT_CRF
    preset: str = DEFAULT_PRESET
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE
    video_codec: str = DEFAULT_VIDEO_CODEC

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError(f"start_ms must be non-negative, got {self.start_ms}")
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )
        if not MIN_CRF <= self.crf <= MAX_CRF:
            raise ValueError(f"crf must be between {MIN_CRF} and {MAX_CRF}, got {self.crf}")
        if not self.preset:
            raise ValueError("preset must not be empty")
        if not self.audio_bitrate:
            raise ValueError("audio_bitrate must not be empty")
        if not self.video_codec:
            raise ValueError("video_codec must not be empty")
        if Path(self.source_path).resolve() == Path(self.output_path).resolve():
            raise ValueError("output_path must differ from source_path (originals are immutable)")

        # ``StrEnum`` members compare equal to their values, so callers may pass
        # plain strings; anything else is a programming error.
        EncodingMode(self.encoding_mode)
        OutputFormat(self.output_format)
        CropMode(self.crop_mode)
        if self.encoding_mode == EncodingMode.FAST:
            if self.output_format != OutputFormat.ORIGINAL:
                raise ValueError("fast stream copy only supports the original output format")
            if self.subtitles_path is not None:
                raise ValueError("fast stream copy cannot burn subtitles")
        if self.output_format == OutputFormat.ORIGINAL and self.subtitles_path is not None:
            raise ValueError("burning subtitles requires the vertical output format")


def _encoding_args(spec: ClipSpec) -> list[str]:
    """Return the codec flags of an accurate (re-encoding) render."""
    return [
        "-c:v",
        spec.video_codec,
        "-preset",
        spec.preset,
        "-crf",
        str(spec.crf),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        spec.audio_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
    ]


def build_clip_command(spec: ClipSpec) -> list[str]:
    """Return the FFmpeg arguments for ``spec`` (global flags excluded)."""
    start_s = spec.start_ms / 1000.0
    duration_s = (spec.end_ms - spec.start_ms) / 1000.0
    command = [
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(spec.source_path),
        "-t",
        f"{duration_s:.3f}",
    ]

    if spec.encoding_mode == EncodingMode.FAST:
        # Stream copy keeps the original codecs, so only the original geometry
        # is possible. ``-ss`` before ``-i`` seeks to the previous keyframe.
        return [
            *command,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(spec.output_path),
        ]

    if spec.output_format == OutputFormat.VERTICAL:
        graph = build_clip_filtergraph(
            ass_path=spec.subtitles_path,
            layout=spec.layout,
            crop=spec.crop_mode == CropMode.CROP,
        )
        if spec.subtitles_path is None:
            # The clip graph ends in [vout] without subtitles; rename it so the
            # pipeline always maps the same label.
            graph = f"{graph};[vout]null[vfinal]"
        command += ["-filter_complex", graph, "-map", "[vfinal]"]
    else:
        command += ["-map", "0:v:0"]

    return [
        *command,
        "-map",
        "0:a?",
        *_encoding_args(spec),
        "-y",
        str(spec.output_path),
    ]


def render_clip(
    runner: FFmpegRunner,
    spec: ClipSpec,
    *,
    on_progress: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
    cancel_check: Callable[[], None] | None = None,
    timeout_s: float | None = None,
) -> RenderResult:
    """Render ``spec`` with ``runner`` and report progress over the clip duration."""
    Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)
    return runner.run(
        build_clip_command(spec),
        duration_ms=spec.end_ms - spec.start_ms,
        on_progress=on_progress,
        cancel_event=cancel_event,
        cancel_check=cancel_check,
        timeout_s=timeout_s,
    )


__all__ = [
    "DEFAULT_AUDIO_BITRATE",
    "DEFAULT_CRF",
    "DEFAULT_PRESET",
    "DEFAULT_VIDEO_CODEC",
    "ClipSpec",
    "CropMode",
    "EncodingMode",
    "OutputFormat",
    "build_clip_command",
    "render_clip",
]
