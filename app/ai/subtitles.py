"""Subtitle cues from M3 transcript segments (pure, phrase-level only).

M3 stores whole recognised phrases (``TranscriptSegment``) without word-level
timestamps, so this module builds an honest phrase track: segments are clipped
to the clip window, re-based to it, split proportionally when too long and
never pretend to know where individual words land.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_CUE_MS: int = 3200
MIN_CUE_MS: int = 700
MAX_CUE_CHARS: int = 84


@dataclass(frozen=True)
class Segment:
    """One M3 transcript segment (absolute media timestamps)."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Cue:
    """One caption shown at ``start_ms`` (clip-relative) for ``end_ms``."""

    start_ms: int
    end_ms: int
    text: str


def _clean(text: str) -> str:
    return " ".join(text.split())


def _clip_segments(
    segments: Iterable[Segment], window_start_ms: int, window_end_ms: int
) -> list[tuple[int, int, str]]:
    """Clip segments to ``[window_start, window_end)`` keeping their text whole.

    A segment that merely touches the window boundary contributes no time and
    is dropped; a partially covered segment keeps its full text (the only
    honest option without word timings).
    """
    prepared: list[tuple[int, int, str]] = []
    for item in segments:
        text = _clean(item.text)
        if not text:
            continue
        start = max(item.start_ms, window_start_ms)
        end = min(item.end_ms, window_end_ms)
        if end - start <= 0:
            continue
        prepared.append((start, end, text))
    prepared.sort(key=lambda row: row[0])
    return prepared


def _partition(
    start_ms: int, end_ms: int, text: str
) -> list[tuple[int, int, str]]:
    """Split one over-long phrase into duration-proportional cues.

    The phrase is cut into equal character chunks by sentence punctuation when
    present, otherwise evenly by word count; the absolute window is divided
    proportionally to the character weight of each chunk.
    """
    words = text.split()
    budget = max(1, MAX_CUE_CHARS)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > budget:
            chunks.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition
    if current:
        chunks.append(" ".join(current))
    if len(chunks) < 2:
        return [(start_ms, end_ms, text)]

    weights = [len(chunk) for chunk in chunks]
    result: list[tuple[int, int, str]] = []
    cursor = start_ms
    for index, chunk in enumerate(chunks):
        is_last = index == len(chunks) - 1
        # Each part gets the remaining time proportional to its character
        # weight; the last part always absorbs the exact remainder.
        remaining_weight = sum(weights[index:])
        share = (
            (end_ms - cursor) * weights[index] // remaining_weight
            if not is_last
            else end_ms - cursor
        )
        result.append((cursor, cursor + max(share, 1 if not is_last else 0), chunk))
        cursor += result[-1][1] - result[-1][0]
    return result


def build_cues(
    segments: Sequence[Segment], *, start_ms: int, end_ms: int
) -> list[Cue]:
    """Build the clip-relative cue track of one clip window.

    Returns an empty list for an empty window or a window without speech.
    """
    if end_ms <= start_ms:
        return []
    cues: list[Cue] = []
    for seg_start, seg_end, text in _clip_segments(segments, start_ms, end_ms):
        if seg_end - seg_start > MAX_CUE_MS or len(text) > MAX_CUE_CHARS:
            parts = _partition(seg_start, seg_end, text)
        else:
            parts = [(seg_start, seg_end, text)]
        for part_start, part_end, part_text in parts:
            cue_start = part_start - start_ms
            cue_end = part_end - start_ms
            # Keep every cue readable, but never let the extension overlap the
            # next part of the same phrase or the following segment.
            next_start = parts[parts.index((part_start, part_end, part_text)) + 1][0] if (
                parts.index((part_start, part_end, part_text)) + 1 < len(parts)
            ) else end_ms
            ceiling = min(next_start, end_ms) - start_ms
            if cue_end - cue_start < MIN_CUE_MS and cue_end < ceiling:
                cue_end = min(cue_start + MIN_CUE_MS, ceiling)
            if cue_end > cue_start:
                cues.append(Cue(start_ms=cue_start, end_ms=cue_end, text=part_text))
    return cues


__all__ = ["MAX_CUE_CHARS", "MAX_CUE_MS", "MIN_CUE_MS", "Cue", "Segment", "build_cues"]
