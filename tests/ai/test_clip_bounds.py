"""Unit tests for the pure clip boundary calculation (M6.2)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from app.ai.clips import (
    ClipBounds,
    ClipBoundsConfig,
    ClipBoundsError,
    calculate_bounds,
)

pytestmark = pytest.mark.unit

VIDEO_MS = 3_600_000
CONFIG = ClipBoundsConfig(
    pre_padding_ms=5000,
    post_padding_ms=10_000,
    min_duration_ms=5000,
    max_duration_ms=60_000,
)


def _bounds(start_ms: int, end_ms: int, **overrides: object) -> ClipBounds:
    config = replace(CONFIG, **overrides)  # type: ignore[arg-type]
    return calculate_bounds(
        start_ms=start_ms, end_ms=end_ms, video_duration_ms=VIDEO_MS, config=config
    )


def test_candidate_mode_adds_asymmetric_padding() -> None:
    bounds = _bounds(125_000, 145_000)
    assert bounds == ClipBounds(start_ms=120_000, end_ms=155_000)
    assert bounds.duration_ms == 35_000


def test_start_is_clamped_to_zero_without_shortening_the_clip() -> None:
    bounds = _bounds(2_000, 20_000)
    assert bounds.start_ms == 0
    assert bounds.duration_ms == 33_000


def test_end_is_clamped_to_the_video_duration_by_shifting_the_window() -> None:
    bounds = _bounds(VIDEO_MS - 20_000, VIDEO_MS - 5_000)
    assert bounds.end_ms == VIDEO_MS
    assert bounds.duration_ms == 30_000


def test_candidate_end_past_the_video_is_trimmed_not_rejected() -> None:
    bounds = _bounds(VIDEO_MS - 10_000, VIDEO_MS + 5_000, max_duration_ms=30_000)
    assert bounds.end_ms == VIDEO_MS
    assert bounds.start_ms == VIDEO_MS - 25_000


def test_short_window_is_grown_to_the_minimum_duration() -> None:
    bounds = _bounds(100_000, 101_000, pre_padding_ms=100, post_padding_ms=100)
    assert bounds.duration_ms == 5000
    assert bounds.start_ms < 100_000 < bounds.end_ms


def test_padding_over_the_maximum_is_trimmed_but_the_candidate_stays_inside() -> None:
    bounds = _bounds(100_000, 150_000, max_duration_ms=60_000)
    assert bounds.duration_ms == 60_000
    assert bounds.start_ms <= 100_000
    assert bounds.end_ms >= 150_000


def test_fixed_mode_centres_a_target_window_on_the_candidate() -> None:
    bounds = _bounds(125_000, 145_000, duration_mode="fixed", target_duration_ms=20_000)
    assert bounds == ClipBounds(start_ms=125_000, end_ms=145_000)


def test_fixed_mode_never_cuts_a_candidate_longer_than_the_target() -> None:
    bounds = _bounds(100_000, 130_000, duration_mode="fixed", target_duration_ms=20_000)
    assert bounds.start_ms <= 100_000
    assert bounds.end_ms >= 130_000
    assert bounds.duration_ms == 30_000


def test_fixed_mode_defaults_to_the_maximum_duration() -> None:
    bounds = _bounds(100_000, 110_000, duration_mode="fixed", max_duration_ms=30_000)
    assert bounds.duration_ms == 30_000
    assert bounds.start_ms == 90_000


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "message"),
    [
        (-1, 10_000, "negative"),
        (10_000, 10_000, "greater than start"),
        (10_000, 5_000, "greater than start"),
        (VIDEO_MS, VIDEO_MS + 5_000, "outside"),
        (VIDEO_MS + 1, VIDEO_MS + 5_000, "outside"),
    ],
)
def test_invalid_candidate_bounds_are_rejected(
    start_ms: int, end_ms: int, message: str
) -> None:
    with pytest.raises(ClipBoundsError, match=message):
        calculate_bounds(
            start_ms=start_ms,
            end_ms=end_ms,
            video_duration_ms=VIDEO_MS,
            config=CONFIG,
        )


def test_unknown_video_duration_is_rejected() -> None:
    with pytest.raises(ClipBoundsError, match="duration is unknown"):
        calculate_bounds(start_ms=0, end_ms=1000, video_duration_ms=0, config=CONFIG)


def test_interest_longer_than_the_maximum_is_rejected() -> None:
    with pytest.raises(ClipBoundsError, match="longer than the 60000 ms maximum"):
        _bounds(100_000, 200_000)


def test_video_shorter_than_the_minimum_is_rejected() -> None:
    with pytest.raises(ClipBoundsError, match="shorter than the 5000 ms minimum"):
        calculate_bounds(
            start_ms=0, end_ms=1000, video_duration_ms=3000, config=CONFIG
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"pre_padding_ms": -1},
        {"post_padding_ms": -1},
        {"min_duration_ms": 0},
        {"max_duration_ms": 1000, "min_duration_ms": 5000},
        {"duration_mode": "auto"},
        {"target_duration_ms": 1000},
        {"target_duration_ms": 130_000},
    ],
)
def test_configuration_is_validated(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ClipBoundsConfig(**overrides)  # type: ignore[arg-type]


def test_config_from_settings_uses_the_clip_defaults() -> None:
    from app.core.config import Settings

    config = ClipBoundsConfig.from_settings(Settings(_env_file=None))
    assert config == ClipBoundsConfig(
        pre_padding_ms=5000,
        post_padding_ms=10_000,
        min_duration_ms=5000,
        max_duration_ms=120_000,
    )
