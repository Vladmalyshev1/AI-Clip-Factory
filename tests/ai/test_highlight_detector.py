"""Highlight detector: signals, scoring, boundaries and merging (M4).

Every test feeds plain segment objects through the pure detector, so no
database, media or model is involved: the assertions describe the algorithm's
contract, not an implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from app.ai.highlights import HighlightConfig, detect_highlights
from app.core.config import Settings

pytestmark = pytest.mark.unit

EXCITED = "wow no way insane!"
CALM = "we walk to the next objective"
LAUGH = "hahaha lol that was funny"


@dataclass(frozen=True)
class Seg:
    """Minimal segment accepted by the detector."""

    start_ms: int
    end_ms: int
    text: str
    no_speech_prob: float | None = None


CONFIG = HighlightConfig(
    padding_ms=2000,
    merge_gap_ms=5000,
    min_duration_ms=5000,
    max_duration_ms=60_000,
    context_window_ms=10_000,
    min_score=0.35,
    max_candidates=50,
)


def _burst(
    start_ms: int,
    *,
    count: int = 8,
    step_ms: int = 1000,
    text: str = EXCITED,
    no_speech_prob: float | None = None,
) -> list[Seg]:
    """Return ``count`` excited segments one second apart from ``start_ms``."""
    return [
        Seg(
            start_ms=start_ms + index * step_ms,
            end_ms=start_ms + index * step_ms + 900,
            text=text,
            no_speech_prob=no_speech_prob,
        )
        for index in range(count)
    ]


def _calm(start_ms: int, *, count: int = 10, step_ms: int = 1000) -> list[Seg]:
    """Return ``count`` neutral segments one second apart from ``start_ms``."""
    return [
        Seg(
            start_ms=start_ms + index * step_ms,
            end_ms=start_ms + index * step_ms + 900,
            text=CALM,
        )
        for index in range(count)
    ]


def test_empty_transcript_produces_no_candidates() -> None:
    assert detect_highlights([], duration_ms=60_000, config=CONFIG) == []
    assert detect_highlights([], duration_ms=0, config=CONFIG) == []


def test_a_zero_duration_video_produces_no_candidates() -> None:
    segments = _burst(0)
    assert detect_highlights(segments, duration_ms=0, config=CONFIG) == []


def test_neutral_speech_is_not_a_highlight() -> None:
    assert detect_highlights(_calm(0), duration_ms=60_000, config=CONFIG) == []


def test_emotional_burst_is_detected() -> None:
    candidates = detect_highlights(_burst(10_000), duration_ms=60_000, config=CONFIG)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.start_ms <= 10_000
    assert candidate.end_ms >= 17_000
    assert "emotional reaction" in candidate.reason
    assert "wow" in candidate.transcript_text
    assert candidate.metadata["keywords"]


def test_laughter_is_detected_as_its_own_signal() -> None:
    candidates = detect_highlights(
        _burst(10_000, text=LAUGH), duration_ms=60_000, config=CONFIG
    )
    assert len(candidates) == 1
    assert "laughter" in candidates[0].reason


def test_scores_are_bounded_and_sorted() -> None:
    segments = [
        *_burst(10_000),
        *(_burst(30_000, count=12, text="oh my god hahaha!!")),
        *_burst(60_000, count=4),
    ]
    candidates = detect_highlights(segments, duration_ms=90_000, config=CONFIG)
    assert len(candidates) >= 2
    scores = [candidate.score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_candidate_boundaries_are_clamped_to_the_video() -> None:
    at_start = detect_highlights(_burst(0, count=6), duration_ms=30_000, config=CONFIG)
    assert len(at_start) == 1
    assert at_start[0].start_ms == 0

    at_end = detect_highlights(
        _burst(22_000, count=6), duration_ms=28_000, config=CONFIG
    )
    assert len(at_end) == 1
    assert at_end[0].end_ms <= 28_000


def test_short_moments_are_dropped() -> None:
    single = [Seg(start_ms=10_000, end_ms=10_500, text="wow!")]
    strict = replace(CONFIG, min_duration_ms=8000)
    assert detect_highlights(single, duration_ms=60_000, config=strict) == []


def test_long_moments_are_truncated_to_the_maximum() -> None:
    candidates = detect_highlights(
        _burst(10_000, count=40),
        duration_ms=120_000,
        config=replace(CONFIG, max_duration_ms=30_000),
    )
    assert len(candidates) == 1
    assert candidates[0].end_ms - candidates[0].start_ms <= 30_000


def test_nearby_moments_are_merged() -> None:
    segments = [*_burst(10_000, count=6), *_burst(18_000, count=6)]
    candidates = detect_highlights(segments, duration_ms=60_000, config=CONFIG)
    assert len(candidates) == 1
    assert candidates[0].start_ms <= 10_000
    assert candidates[0].end_ms >= 23_000


def test_distant_moments_stay_separate() -> None:
    segments = [*_burst(10_000, count=6), *_burst(40_000, count=6)]
    candidates = detect_highlights(segments, duration_ms=90_000, config=CONFIG)
    assert len(candidates) == 2


def test_identical_overlapping_segments_do_not_duplicate_a_candidate() -> None:
    segment = Seg(start_ms=10_000, end_ms=18_000, text=EXCITED)
    candidates = detect_highlights(
        [segment, segment, segment], duration_ms=60_000, config=CONFIG
    )
    assert len(candidates) == 1


def test_an_overlapping_segment_neither_shrinks_nor_hides_a_moment() -> None:
    """Regression test: the window search must key on sorted ``starts``.

    A short segment overlapping a long one makes the segment *center* list
    non-monotonic; bisecting over it used to smooth the long segment to zero
    and let the short one shrink the merged region. The fix must still detect
    one full moment ending at ``45_000 + padding``.
    """
    segments = [
        Seg(start_ms=0, end_ms=45_000, text=EXCITED),
        Seg(start_ms=10_000, end_ms=11_000, text=EXCITED),
    ]
    candidates = detect_highlights(segments, duration_ms=60_000, config=CONFIG)
    assert len(candidates) == 1
    assert candidates[0].start_ms == 0
    assert candidates[0].end_ms == 47_000


def test_probably_silent_segments_are_deweighted() -> None:
    assert (
        detect_highlights(
            _burst(10_000, no_speech_prob=0.99),
            duration_ms=60_000,
            config=CONFIG,
        )
        == []
    )


def test_detection_is_deterministic() -> None:
    segments = [*_calm(0, count=5), *_burst(10_000), *_calm(30_000, count=5)]
    first = detect_highlights(segments, duration_ms=60_000, config=CONFIG)
    second = detect_highlights(segments, duration_ms=60_000, config=CONFIG)
    assert first == second


def test_max_candidates_keeps_the_best_scores() -> None:
    segments = [
        *_burst(5_000, count=8),
        *_burst(25_000, count=10, text="oh my god hahaha!!"),
        *_burst(50_000, count=6),
    ]
    candidates = detect_highlights(
        segments, duration_ms=90_000, config=replace(CONFIG, max_candidates=1)
    )
    assert len(candidates) == 1
    assert "oh my god" in candidates[0].transcript_text


def test_metadata_reports_every_factor() -> None:
    candidates = detect_highlights(_burst(10_000), duration_ms=60_000, config=CONFIG)
    metadata = candidates[0].metadata
    assert {"intensity", "support", "density_ratio", "duration_fit", "segments"} <= set(
        metadata
    )
    assert 0 <= float(metadata["intensity"]) <= 1
    assert int(metadata["segments"]) > 0


def test_config_is_built_from_settings(settings: Settings) -> None:
    config = HighlightConfig.from_settings(settings)
    assert config.padding_ms == settings.highlight_padding_ms
    assert config.merge_gap_ms == settings.highlight_merge_gap_ms
    assert config.min_score == settings.highlight_min_score
