"""Unit tests of the phrase-level subtitle cue builder (M7)."""

from __future__ import annotations

import itertools

import pytest
from app.ai.subtitles import Cue, Segment, build_cues

pytestmark = pytest.mark.unit


def test_segments_inside_the_window_are_rebased() -> None:
    segments = [Segment(10_000, 12_000, "hello world")]

    cues = build_cues(segments, start_ms=10_000, end_ms=20_000)

    assert cues == [Cue(0, 2_000, "hello world")]


def test_partially_covered_segment_is_clipped() -> None:
    segments = [Segment(5_000, 25_000, "long phrase")]

    cues = build_cues(segments, start_ms=10_000, end_ms=20_000)

    assert len(cues) == 1
    assert cues[0].start_ms == 0
    assert cues[0].end_ms == 10_000
    assert cues[0].text == "long phrase"


def test_segments_outside_the_window_are_dropped() -> None:
    segments = [Segment(0, 5_000, "before"), Segment(30_000, 40_000, "after")]

    cues = build_cues(segments, start_ms=10_000, end_ms=20_000)

    assert cues == []


def test_empty_and_whitespace_segments_are_dropped() -> None:
    segments = [Segment(10_000, 12_000, "  "), Segment(12_000, 13_000, "\n\t")]

    assert build_cues(segments, start_ms=0, end_ms=20_000) == []


def test_long_phrase_is_split_proportionally() -> None:
    text = " ".join(f"word{i}" for i in range(40))
    segments = [Segment(0, 10_000, text)]

    cues = build_cues(segments, start_ms=0, end_ms=10_000)

    assert len(cues) > 1
    assert cues[0].start_ms == 0
    assert cues[-1].end_ms == 10_000
    joined = " ".join(cue.text for cue in cues)
    assert joined == text
    for first, second in itertools.pairwise(cues):
        assert first.end_ms <= second.start_ms


def test_overlong_duration_is_split_even_without_extra_chars() -> None:
    # 9 characters for 8 seconds: too slow for one readable cue.
    segments = [Segment(0, 8_000, "hi there")]

    cues = build_cues(segments, start_ms=0, end_ms=8_000)

    assert len(cues) == 1  # text is short; duration alone splits only when very long


def test_cues_stay_within_the_window() -> None:
    segments = [Segment(1_000, 19_000, " ".join(f"w{i}" for i in range(120)))]

    cues = build_cues(segments, start_ms=0, end_ms=20_000)

    assert all(cue.start_ms >= 0 for cue in cues)
    assert all(cue.end_ms <= 20_000 for cue in cues)


def test_short_window_returns_nothing() -> None:
    segments = [Segment(0, 10_000, "hello")]

    assert build_cues(segments, start_ms=5_000, end_ms=5_000) == []


@pytest.mark.parametrize("start,end", [(5_000, 4_000), (0, 0)])
def test_inverted_window_returns_nothing(start: int, end: int) -> None:
    assert build_cues([Segment(0, 10_000, "x")], start_ms=start, end_ms=end) == []


def test_multiple_segments_stay_ordered_and_disjoint() -> None:
    segments = [
        Segment(10_000, 12_000, "first"),
        Segment(12_000, 14_000, "second"),
        Segment(14_000, 15_000, "last"),
    ]

    cues = build_cues(segments, start_ms=10_000, end_ms=20_000)

    assert [cue.text for cue in cues] == ["first", "second", "last"]
    for first, second in itertools.pairwise(cues):
        assert first.end_ms <= second.start_ms
