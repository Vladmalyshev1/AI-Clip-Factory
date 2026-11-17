"""Pure clip boundary calculation (M6).

M4 stores candidates with absolute media timestamps in integer milliseconds;
M6 turns one candidate into the ``[start_ms, end_ms)`` window that is handed to
FFmpeg. All arithmetic stays in milliseconds — no unit mixing anywhere.

Two duration modes exist:

* ``candidate`` — the candidate plus its pre/post padding, shrunk around the
  candidate centre only when the padded window would exceed the maximum;
* ``fixed`` — a window of ``target_duration_ms`` centred on the candidate
  centre; the interesting interval is never cut, so a longer candidate widens
  the window instead of being trimmed.

Whatever the mode, the window is clamped to the source recording, shifted (not
truncated) when it touches a boundary, and grown to the minimum duration when
the padded window is too short. Anything that cannot be satisfied raises
:class:`ClipBoundsError` so the API can answer with ``CLIP_BOUNDS_INVALID``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.core.config import Settings

DEFAULT_DURATION_MODE: Final = "candidate"
DURATION_MODES: Final[frozenset[str]] = frozenset({"candidate", "fixed"})


class ClipBoundsError(ValueError):
    """A candidate cannot be turned into a clip inside the configured limits."""


@dataclass(frozen=True)
class ClipBoundsConfig:
    """Padding and duration limits of one clip request."""

    pre_padding_ms: int = 5000
    post_padding_ms: int = 10_000
    min_duration_ms: int = 5000
    max_duration_ms: int = 120_000
    duration_mode: str = DEFAULT_DURATION_MODE
    target_duration_ms: int | None = None

    def __post_init__(self) -> None:
        if self.pre_padding_ms < 0:
            raise ValueError(f"pre_padding_ms must be non-negative, got {self.pre_padding_ms}")
        if self.post_padding_ms < 0:
            raise ValueError(f"post_padding_ms must be non-negative, got {self.post_padding_ms}")
        if self.min_duration_ms <= 0:
            raise ValueError(f"min_duration_ms must be positive, got {self.min_duration_ms}")
        if self.max_duration_ms < self.min_duration_ms:
            raise ValueError(
                f"max_duration_ms ({self.max_duration_ms}) must not be smaller than "
                f"min_duration_ms ({self.min_duration_ms})"
            )
        if self.duration_mode not in DURATION_MODES:
            raise ValueError(
                f"duration_mode must be one of {sorted(DURATION_MODES)}, "
                f"got {self.duration_mode!r}"
            )
        if self.target_duration_ms is not None and not (
            self.min_duration_ms <= self.target_duration_ms <= self.max_duration_ms
        ):
            raise ValueError(
                f"target_duration_ms ({self.target_duration_ms}) must be between "
                f"{self.min_duration_ms} and {self.max_duration_ms}"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> ClipBoundsConfig:
        """Build the default configuration from application settings."""
        return cls(
            pre_padding_ms=settings.clip_pre_padding_ms,
            post_padding_ms=settings.clip_post_padding_ms,
            min_duration_ms=settings.clip_min_duration_ms,
            max_duration_ms=settings.clip_max_duration_ms,
        )


@dataclass(frozen=True)
class ClipBounds:
    """Resolved half-open clip window in absolute milliseconds."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        """Window length in milliseconds."""
        return self.end_ms - self.start_ms


def calculate_bounds(
    *,
    start_ms: int,
    end_ms: int,
    video_duration_ms: int,
    config: ClipBoundsConfig,
) -> ClipBounds:
    """Return the clip window for one candidate.

    Raises:
        ClipBoundsError: the candidate is invalid, lies fully outside the video,
            or the limits cannot be met without cutting the interesting interval.
    """
    if video_duration_ms <= 0:
        raise ClipBoundsError(f"Source video duration is unknown ({video_duration_ms} ms)")
    if start_ms < 0:
        raise ClipBoundsError(f"Candidate start is negative ({start_ms} ms)")
    if end_ms <= start_ms:
        raise ClipBoundsError(
            f"Candidate end ({end_ms} ms) must be greater than start ({start_ms} ms)"
        )
    if start_ms >= video_duration_ms:
        raise ClipBoundsError(
            f"Candidate starts at {start_ms} ms, outside the {video_duration_ms} ms video"
        )
    if video_duration_ms < config.min_duration_ms:
        raise ClipBoundsError(
            f"Video is {video_duration_ms} ms long, shorter than the "
            f"{config.min_duration_ms} ms minimum clip duration"
        )

    # A candidate may end a hair past the duration (timestamp rounding); pad the
    # end inwards instead of failing the whole request.
    interest_end = min(end_ms, video_duration_ms)
    interest_len = interest_end - start_ms
    if interest_len > config.max_duration_ms:
        raise ClipBoundsError(
            f"Candidate is {interest_len} ms long, longer than the "
            f"{config.max_duration_ms} ms maximum clip duration"
        )

    if config.duration_mode == "fixed":
        target = config.target_duration_ms or config.max_duration_ms
        length = max(target, interest_len)
    else:
        length = interest_len + config.pre_padding_ms + config.post_padding_ms

    length = min(max(length, config.min_duration_ms), config.max_duration_ms)
    # Clamping to the video never violates the minimum: the video itself was
    # checked against it above.
    length = min(length, video_duration_ms)

    centre = (start_ms + interest_end) // 2
    if config.duration_mode == "candidate":
        window_start = start_ms - config.pre_padding_ms
        window_end = interest_end + config.post_padding_ms
        if window_end - window_start != length:
            # Padding exceeded the maximum, or the window was grown to the
            # minimum: keep the candidate centred in the resolved length.
            window_start = centre - length // 2
            window_end = window_start + length
    else:
        window_start = centre - length // 2
        window_end = window_start + length

    if window_start < 0:
        window_start = 0
        window_end = window_start + length
    if window_end > video_duration_ms:
        window_end = video_duration_ms
        window_start = window_end - length

    if window_end <= window_start:  # pragma: no cover - guarded by the checks above
        raise ClipBoundsError("Calculated clip window is empty")
    return ClipBounds(start_ms=window_start, end_ms=window_end)


__all__ = [
    "DEFAULT_DURATION_MODE",
    "DURATION_MODES",
    "ClipBounds",
    "ClipBoundsConfig",
    "ClipBoundsError",
    "calculate_bounds",
]
