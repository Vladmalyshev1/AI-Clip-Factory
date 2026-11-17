"""Rule-based highlight detection over a completed transcript (M4).

The detector is deliberately pure and stdlib-only: it consumes recognised
segments with absolute timestamps and produces scored candidates, so a future
ML scorer can replace the scoring behind the same function signature. Whisper
provides no loudness information, so loudness is *not* part of the score —
every signal below is derived from the transcript text and its timings only.

Pipeline: score each segment, smooth the scores over a context window, group
high-signal segments into regions, pad/merge/clamp the regions into candidate
intervals and blend five factors into a final ``[0, 1]`` score:

* intensity — the strongest single segment in the interval;
* support   — how much of the surrounding speech participates;
* density   — characters per second relative to the video's own baseline;
* duration  — trapezoid preference for 20-90 second moments;
* keywords  — (folded into intensity) emotion, laughter, shouting, elongations.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from app.core.config import Settings

#: Words and phrases that mark an emotional reaction (case-insensitive).
EMOTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno way\b",
        r"\boh my (?:god|goodness|gosh|lord)\b",
        r"\bholy (?:cow|moly|crap|shit)\b",
        r"\blet'?s go\b",
        r"\bhere we go\b",
        r"\bwhat the\b",
        r"\bare you (?:kidding|serious)\b",
        r"\bget out\b",
        r"\b(?:unbelievable|insane|incredible|unreal|legendary|godlike|clutch|epic)\b",
        r"\b(?:wow|whoa|woah|sheesh|yay|wtf|omg)\b",
    )
)

#: Laughter and amusement markers (case-insensitive).
LAUGHTER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:ha){2,}\b",
        r"\bhehe\b",
        r"\blol+\b",
        r"\blmao\b",
        r"\blmfao\b",
        r"\brofl\b",
    )
)

#: Upper-case acronyms that never count as shouting.
CAPS_DENYLIST: Final[frozenset[str]] = frozenset(
    {"TV", "OK", "ID", "PC", "FPS", "RPG", "MMO", "NPC", "AFK"}
)

CAPS_WORD_RE: Final = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_RE: Final = re.compile(r"([A-Za-z])\1{2,}")
REPEATED_PUNCT_RE: Final = re.compile(r"[!?]{2,}")

#: Candidate length sweet spot used by the duration factor.
IDEAL_MIN_MS: Final = 20_000
IDEAL_MAX_MS: Final = 90_000

#: How much each signal category contributes to a segment's raw signal.
#: Every category is capped, so one category alone can never saturate it.
SIGNAL_WEIGHTS: Final[dict[str, float]] = {
    "emotion": 0.30,
    "laughter": 0.25,
    "exclamation": 0.15,
    "caps": 0.15,
    "elongation": 0.15,
    "punctuation": 0.10,
}
#: How much each factor contributes to the final candidate score.
SCORE_WEIGHTS: Final[dict[str, float]] = {
    "intensity": 0.45,
    "support": 0.25,
    "density": 0.15,
    "duration_fit": 0.15,
}

#: Hard cap for the stored transcript text of one candidate.
MAX_TRANSCRIPT_CHARS: Final = 4000
#: Hard cap for the stored keyword sample of one candidate.
MAX_KEYWORD_SAMPLES: Final = 10

_LABELS: Final[dict[str, str]] = {
    "emotion": "emotional reaction",
    "laughter": "laughter",
    "exclamation": "exclamation",
    "caps": "shouted words",
    "elongation": "prolonged vowels",
    "punctuation": "repeated punctuation",
}


class SegmentLike(Protocol):
    """Transcript segment attributes the detector needs."""

    start_ms: int
    end_ms: int
    text: str
    no_speech_prob: float | None


@dataclass(frozen=True)
class HighlightConfig:
    """Configurable boundaries of one detection run."""

    padding_ms: int = 2000
    merge_gap_ms: int = 10_000
    min_duration_ms: int = 15_000
    max_duration_ms: int = 120_000
    context_window_ms: int = 30_000
    min_score: float = 0.35
    max_candidates: int = 50

    @classmethod
    def from_settings(cls, settings: Settings) -> HighlightConfig:
        """Build the configuration from the application settings."""
        return cls(
            padding_ms=settings.highlight_padding_ms,
            merge_gap_ms=settings.highlight_merge_gap_ms,
            min_duration_ms=settings.highlight_min_duration_ms,
            max_duration_ms=settings.highlight_max_duration_ms,
            context_window_ms=settings.highlight_context_window_ms,
            min_score=settings.highlight_min_score,
            max_candidates=settings.highlight_max_candidates,
        )


@dataclass(frozen=True)
class HighlightCandidateData:
    """One detected interesting moment, before it is persisted."""

    start_ms: int
    end_ms: int
    score: float
    reason: str
    transcript_text: str
    metadata: dict[str, float | int | list[str]] = field(default_factory=dict)


def _segment_counts(text: str) -> dict[str, int]:
    """Count the signal categories present in one segment text."""
    return {
        "emotion": sum(1 for pattern in EMOTION_PATTERNS if pattern.search(text)),
        "laughter": sum(1 for pattern in LAUGHTER_PATTERNS if pattern.search(text)),
        "exclamation": min(text.count("!"), 2),
        "caps": sum(
            1 for word in CAPS_WORD_RE.findall(text) if word not in CAPS_DENYLIST
        ),
        "elongation": len(ELONGATION_RE.findall(text)),
        "punctuation": len(REPEATED_PUNCT_RE.findall(text)),
    }


def _segment_signal(text: str, no_speech_prob: float | None) -> float:
    """Return the raw ``[0, 1]`` interest signal of one segment.

    ``no_speech_prob`` (when the ASR engine provides it) scales the signal
    down: a window Whisper considers probably silent must not produce a
    highlight, whatever its text looks like.
    """
    counts = _segment_counts(text)
    raw = sum(SIGNAL_WEIGHTS[name] * min(counts[name], 2) for name in SIGNAL_WEIGHTS)
    signal = min(1.0, raw)
    if no_speech_prob is not None:
        signal *= max(0.0, 1.0 - float(no_speech_prob))
    return signal


def _smoothed_signals(
    starts: list[int], signals: list[float], window_ms: int
) -> list[float]:
    """Average every signal over the segments starting in the context window.

    The window is keyed on ``starts`` — not on segment centers — because
    ``starts`` is the one list that is sorted, and overlapping ASR segments
    would make any derived list non-monotonic (and a bisect over it garbage).
    """
    half = window_ms / 2.0
    prefix = [0.0]
    for signal in signals:
        prefix.append(prefix[-1] + signal)
    smoothed: list[float] = []
    for start in starts:
        low = bisect_left(starts, start - half)
        high = bisect_right(starts, start + half)
        count = high - low
        smoothed.append((prefix[high] - prefix[low]) / count if count else 0.0)
    return smoothed


def _regions(
    starts: list[int],
    ends: list[int],
    smoothed: list[float],
    merge_gap_ms: int,
) -> list[tuple[int, int]]:
    """Group high-signal segments into ``(start, end)`` regions.

    Consecutive active segments belong to the same region while the gap
    between them stays within ``merge_gap_ms``, which also bridges short
    stretches of calm segments inside one moment.
    """
    regions: list[tuple[int, int]] = []
    for index, signal in enumerate(smoothed):
        if signal > 0.0:
            if regions and starts[index] - regions[-1][1] <= merge_gap_ms:
                # An overlapping segment may end earlier than the region it
                # belongs to; never let it shrink the region.
                regions[-1] = (regions[-1][0], max(regions[-1][1], ends[index]))
            else:
                regions.append((starts[index], ends[index]))
    return regions


def _pad_and_merge(
    regions: Sequence[tuple[int, int]],
    *,
    padding_ms: int,
    duration_ms: int,
) -> list[tuple[int, int]]:
    """Pad regions, clamp them to the video and merge the overlaps."""
    padded = sorted(
        (max(0, start - padding_ms), min(duration_ms, end + padding_ms))
        for start, end in regions
    )
    merged: list[tuple[int, int]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration_fit(duration_ms: int, max_duration_ms: int) -> float:
    """Return ``[0, 1]`` preference for a candidate length (trapezoid)."""
    if duration_ms <= 0:
        return 0.0
    if duration_ms < IDEAL_MIN_MS:
        return duration_ms / IDEAL_MIN_MS
    if duration_ms <= IDEAL_MAX_MS:
        return 1.0
    if max_duration_ms <= IDEAL_MAX_MS:
        return 0.0
    span = max_duration_ms - IDEAL_MAX_MS
    return max(0.0, (max_duration_ms - duration_ms) / span)


def _matched_keywords(text: str) -> list[str]:
    """Return the emotion/laughter phrases matched in ``text``."""
    found = [
        match.group(0)
        for pattern in (*EMOTION_PATTERNS, *LAUGHTER_PATTERNS)
        for match in pattern.finditer(text)
    ]
    return found[:MAX_KEYWORD_SAMPLES]


def detect_highlights(
    segments: Sequence[SegmentLike],
    *,
    duration_ms: int,
    config: HighlightConfig | None = None,
) -> list[HighlightCandidateData]:
    """Detect the interesting moments of a completed transcript.

    Returns the candidates sorted by score (descending), never more than
    ``config.max_candidates`` and never with a score below
    ``config.min_score``. An empty or silent transcript yields ``[]``.
    """
    config = config or HighlightConfig()
    rows = sorted(
        (segment for segment in segments if segment.end_ms > segment.start_ms),
        key=lambda segment: segment.start_ms,
    )
    if not rows or duration_ms <= 0:
        return []

    starts = [row.start_ms for row in rows]
    ends = [row.end_ms for row in rows]
    texts = [row.text for row in rows]
    signals = [_segment_signal(row.text, row.no_speech_prob) for row in rows]
    smoothed = _smoothed_signals(starts, signals, config.context_window_ms)
    regions = _regions(starts, ends, smoothed, config.merge_gap_ms)
    intervals = _pad_and_merge(
        regions, padding_ms=config.padding_ms, duration_ms=duration_ms
    )

    # Baseline: how many characters per second of speech the whole video has.
    total_chars = sum(len(text) for text in texts)
    total_seconds = sum((row.end_ms - row.start_ms) / 1000 for row in rows)
    baseline_cps = total_chars / total_seconds if total_seconds > 0 else 0.0

    candidates: list[HighlightCandidateData] = []
    for start, end in intervals:
        if end - start < config.min_duration_ms:
            continue
        if end - start > config.max_duration_ms:
            end = start + config.max_duration_ms

        high = bisect_left(starts, end)
        window = [index for index in range(high) if ends[index] > start]
        if not window:
            continue

        intensity = max(signals[index] for index in window)
        support = sum(smoothed[index] for index in window) / len(window)
        span_seconds = (end - start) / 1000
        if span_seconds <= 0:  # pragma: no cover - guarded by min_duration_ms
            continue
        window_chars = sum(len(texts[index]) for index in window)
        density_ratio = (
            (window_chars / span_seconds) / baseline_cps if baseline_cps > 0 else 0.0
        )
        density = min(1.0, density_ratio / 2)
        duration_fit = _duration_fit(end - start, config.max_duration_ms)
        score = min(
            1.0,
            SCORE_WEIGHTS["intensity"] * intensity
            + SCORE_WEIGHTS["support"] * support
            + SCORE_WEIGHTS["density"] * density
            + SCORE_WEIGHTS["duration_fit"] * duration_fit,
        )
        if score < config.min_score:
            continue

        # Labels are counted per segment: joining the window's texts would
        # synthesise matches across segment boundaries ("no" + "way").
        keywords: list[str] = []
        labels: list[str] = []
        for index in window:
            counts = _segment_counts(texts[index])
            for keyword in _matched_keywords(texts[index]):
                if len(keywords) < MAX_KEYWORD_SAMPLES and keyword not in keywords:
                    keywords.append(keyword)
            for name, label in _LABELS.items():
                if counts[name] and label not in labels:
                    labels.append(label)
        if density_ratio > 1.2:
            labels.append("high speech density")
        transcript_text = " ".join(texts[index] for index in window)

        candidates.append(
            HighlightCandidateData(
                start_ms=start,
                end_ms=end,
                score=round(score, 4),
                reason=", ".join(labels) if labels else "keyword match",
                transcript_text=transcript_text[:MAX_TRANSCRIPT_CHARS],
                metadata={
                    "intensity": round(intensity, 4),
                    "support": round(support, 4),
                    "density_ratio": round(density_ratio, 4),
                    "duration_fit": round(duration_fit, 4),
                    "segments": len(window),
                    "keywords": keywords,
                },
            )
        )

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.start_ms))
    return candidates[: config.max_candidates]


__all__ = [
    "EMOTION_PATTERNS",
    "IDEAL_MAX_MS",
    "IDEAL_MIN_MS",
    "LAUGHTER_PATTERNS",
    "MAX_TRANSCRIPT_CHARS",
    "SCORE_WEIGHTS",
    "SIGNAL_WEIGHTS",
    "HighlightCandidateData",
    "HighlightConfig",
    "SegmentLike",
    "detect_highlights",
]
