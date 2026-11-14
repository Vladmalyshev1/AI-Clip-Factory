"""Animated ASS subtitle generation (pure, no FFmpeg).

The pipeline stores word-level timings and turns them into caption cues; this
module renders those cues into a ``.ass`` document that libass (via FFmpeg's
``ass`` filter) burns into the vertical clip.

Everything here is deterministic and free of I/O except :func:`write_ass_file`,
which writes atomically so a crashed render never leaves a half-written file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SCRIPT_HEADER: Final = "[Script Info]"
STYLES_HEADER: Final = "[V4+ Styles]"
EVENTS_HEADER: Final = "[Events]"
STYLE_FORMAT: Final = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
EVENT_FORMAT: Final = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)
FADE_EFFECT: Final = r"{\fad(80,80)}"
DEFAULT_STYLE_NAME: Final = "Default"


@dataclass(frozen=True)
class SubtitleCue:
    """One caption shown between ``start_ms`` and ``end_ms``."""

    text: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int):
            raise ValueError(f"start_ms must be an integer, got {self.start_ms!r}")
        if isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int):
            raise ValueError(f"end_ms must be an integer, got {self.end_ms!r}")
        if self.start_ms < 0:
            raise ValueError(f"start_ms must be non-negative, got {self.start_ms}")
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )


@dataclass(frozen=True)
class SubtitleStyle:
    """ASS style tuned for a 1080x1920 short."""

    font_name: str = "Arial"
    font_size: int = 64
    primary_colour: str = "&H00FFFFFF"
    secondary_colour: str = "&H000000FF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H80000000"
    bold: int = -1
    border_style: int = 1
    outline: int = 4
    shadow: int = 2
    alignment: int = 2
    margin_l: int = 60
    margin_r: int = 60
    margin_v: int = 220
    encoding: int = 1


def format_ass_time(ms: int) -> str:
    """Format milliseconds as ``H:MM:SS.cc`` (centiseconds truncated)."""
    total_cs = max(0, int(ms)) // 10
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _greedy_lines(words: list[str], limit: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _balanced_lines(words: list[str], line_count: int, limit: int) -> list[str]:
    """Wrap ``words`` into at most ``line_count`` lines of minimal equal width."""
    low = max((len(word) for word in words), default=1)
    high = limit
    best = _greedy_lines(words, limit)
    while low <= high:
        middle = (low + high) // 2
        candidate = _greedy_lines(words, middle)
        if len(candidate) <= line_count:
            best = candidate
            high = middle - 1
        else:
            low = middle + 1
    return best


def _wrap_text(text: str, limit: int, max_lines: int) -> str:
    """Wrap escaped text into at most ``max_lines`` lines joined by ``\\N``."""
    words = text.split()
    if not words:
        return ""
    if len(text) <= limit:
        return text
    lines = _greedy_lines(words, limit)
    if len(lines) <= max_lines:
        if len(lines) > 1:
            lines = _balanced_lines(words, len(lines), limit)
        return r"\N".join(lines)

    kept = lines[: max_lines - 1]
    consumed = sum(len(line.split()) for line in kept)
    remainder = words[consumed:]
    budget = max(1, limit - 3)
    tail = _greedy_lines(remainder, budget)
    last = tail[0] if tail else ""
    if not last and remainder:
        last = remainder[0]
    if len(last) > budget:
        last = last[:budget].rstrip()
    kept.append(f"{last}...")
    return r"\N".join(kept)


def _normalise_cues(cues: Iterable[SubtitleCue]) -> list[SubtitleCue]:
    """Collapse whitespace, drop empty cues, sort by time and clamp overlaps."""
    prepared: list[SubtitleCue] = []
    for cue in cues:
        text = " ".join(cue.text.split())
        if not text:
            continue
        prepared.append(SubtitleCue(text=text, start_ms=cue.start_ms, end_ms=cue.end_ms))
    prepared.sort(key=lambda item: (item.start_ms, item.end_ms))

    result: list[SubtitleCue] = []
    for index, cue in enumerate(prepared):
        end_ms = cue.end_ms
        if index + 1 < len(prepared):
            end_ms = min(end_ms, prepared[index + 1].start_ms)
        if end_ms <= cue.start_ms:
            continue
        result.append(SubtitleCue(text=cue.text, start_ms=cue.start_ms, end_ms=end_ms))
    return result


def _format_text(text: str, limit: int, max_lines: int) -> str:
    """Escape ASS override syntax and wrap the caption body."""
    escaped = text.replace("{", r"\{").replace("}", r"\}")
    return _wrap_text(escaped, limit, max_lines)


def _style_line(style: SubtitleStyle) -> str:
    fields = (
        DEFAULT_STYLE_NAME,
        style.font_name,
        str(style.font_size),
        style.primary_colour,
        style.secondary_colour,
        style.outline_colour,
        style.back_colour,
        str(style.bold),
        "0",  # Italic
        "0",  # Underline
        "0",  # StrikeOut
        "100",  # ScaleX
        "100",  # ScaleY
        "0",  # Spacing
        "0",  # Angle
        str(style.border_style),
        str(style.outline),
        str(style.shadow),
        str(style.alignment),
        str(style.margin_l),
        str(style.margin_r),
        str(style.margin_v),
        str(style.encoding),
    )
    return f"Style: {','.join(fields)}"


def build_ass_document(
    cues: Iterable[SubtitleCue],
    style: SubtitleStyle | None = None,
    *,
    video_width: int = 1080,
    video_height: int = 1920,
    max_chars_per_line: int = 22,
    max_lines: int = 2,
) -> str:
    """Render cues into a complete, libass-compatible ASS document."""
    resolved_style = style if style is not None else SubtitleStyle()
    if video_width <= 0 or video_height <= 0:
        raise ValueError(
            f"video dimensions must be positive, got {video_width}x{video_height}"
        )
    if max_chars_per_line <= 0:
        raise ValueError(f"max_chars_per_line must be positive, got {max_chars_per_line}")
    if max_lines <= 0:
        raise ValueError(f"max_lines must be positive, got {max_lines}")

    lines: list[str] = [
        SCRIPT_HEADER,
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        STYLES_HEADER,
        STYLE_FORMAT,
        _style_line(resolved_style),
        "",
        EVENTS_HEADER,
        EVENT_FORMAT,
    ]
    for cue in _normalise_cues(cues):
        text = _format_text(cue.text, max_chars_per_line, max_lines)
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start_ms)},{format_ass_time(cue.end_ms)},"
            f"{DEFAULT_STYLE_NAME},,0,0,0,,{FADE_EFFECT}{text}"
        )
    return "\n".join(lines) + "\n"


def write_ass_file(path: Path, document: str) -> None:
    """Write ``document`` as UTF-8 (LF endings) atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_STYLE_NAME",
    "EVENTS_HEADER",
    "EVENT_FORMAT",
    "SCRIPT_HEADER",
    "STYLES_HEADER",
    "STYLE_FORMAT",
    "SubtitleCue",
    "SubtitleStyle",
    "build_ass_document",
    "format_ass_time",
    "write_ass_file",
]
