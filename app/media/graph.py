"""Vertical 9:16 filtergraph composition (pure string building, no I/O).

The graph takes the first video stream ``[0:v]``, blurs an upscaled copy as the
background, fits the untouched frame into the foreground and overlays it
centred. Subtitles are burned afterwards on the composited stream so they are
never scaled or blurred.

``zoompan`` is intentionally absent: dynamic zooms are out of MVP scope and
would double the render time on the CPU-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_WIDTH: Final = 1080
DEFAULT_HEIGHT: Final = 1920
DEFAULT_BLUR_SIGMA: Final = 20
DEFAULT_BLUR_POWER: Final = 2


@dataclass(frozen=True)
class VerticalLayout:
    """Target canvas for a vertical short plus the background blur settings."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    blur_sigma: int = DEFAULT_BLUR_SIGMA
    blur_power: int = DEFAULT_BLUR_POWER

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if value % 2:
                raise ValueError(f"{name} must be even for yuv420p, got {value}")
        if self.blur_sigma < 0:
            raise ValueError(f"blur_sigma must be non-negative, got {self.blur_sigma}")
        if self.blur_power < 0:
            raise ValueError(f"blur_power must be non-negative, got {self.blur_power}")


def build_vertical_filter(layout: VerticalLayout | None = None) -> str:
    """Return the ``filter_complex`` fragment ``[0:v] -> [vout]``."""
    resolved = layout if layout is not None else VerticalLayout()
    width, height = resolved.width, resolved.height
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"boxblur=luma_radius={resolved.blur_sigma}:luma_power={resolved.blur_power}[bgb];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
    )


def build_crop_filter(layout: VerticalLayout | None = None) -> str:
    """Return the fill-crop ``filter_complex`` fragment ``[0:v] -> [vout]``.

    The frame is scaled up until it covers the canvas and the overflow is cut
    symmetrically, so the aspect ratio is preserved and no black bars appear.
    """
    resolved = layout if layout is not None else VerticalLayout()
    width, height = resolved.width, resolved.height
    return (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},format=yuv420p[vout]"
    )


#: Characters the option-value parser treats specially (level 1).
_OPTION_VALUE_SPECIALS: Final = "\\':"
#: Characters the filtergraph parser treats specially (level 2).
_FILTERGRAPH_SPECIALS: Final = "\\'[],;"


def _escape(text: str, specials: str) -> str:
    return "".join(f"\\{char}" if char in specials else char for char in text)


def escape_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for use as a filter option value.

    ``-filter_complex`` is unescaped twice by FFmpeg: once when the filtergraph
    is split into filters and options, once when each option value is parsed.
    The path is normalised to forward slashes and escaped for both levels, so
    ``C:/a/b.ass`` becomes ``C\\\\:/a/b.ass`` and an apostrophe becomes
    ``\\\\\\'``.

    The result is deliberately **not** quoted: FFmpeg cannot represent a literal
    single quote inside a quoted string (``\\'`` is swallowed without error), so
    quoting a path that contains one silently corrupts it. Escaping every special
    character instead works for quotes, spaces, commas, brackets and semicolons.
    """
    text = str(path).replace("\\", "/")
    return _escape(_escape(text, _OPTION_VALUE_SPECIALS), _FILTERGRAPH_SPECIALS)


def build_ass_filter(ass_path: str | Path) -> str:
    """Return the ``ass`` filter that burns ``ass_path`` (option, unquoted)."""
    return f"ass={escape_filter_path(ass_path)}"


def build_clip_filtergraph(
    *,
    ass_path: str | Path | None = None,
    layout: VerticalLayout | None = None,
    crop: bool = False,
) -> str:
    """Return the full clip graph, ending in ``[vfinal]`` when subtitles are burned."""
    graph = build_crop_filter(layout) if crop else build_vertical_filter(layout)
    if ass_path is None:
        return graph
    return f"{graph};[vout]{build_ass_filter(ass_path)}[vfinal]"


__all__ = [
    "DEFAULT_BLUR_POWER",
    "DEFAULT_BLUR_SIGMA",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "VerticalLayout",
    "build_ass_filter",
    "build_clip_filtergraph",
    "build_crop_filter",
    "build_vertical_filter",
    "escape_filter_path",
]
