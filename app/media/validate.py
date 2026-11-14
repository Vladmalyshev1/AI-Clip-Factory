"""Validation of rendered clips before they are published or shown in the UI.

Every check raises :class:`RenderValidationError` so the pipeline has exactly
one error type to catch, regardless of whether the failure came from ffprobe or
from the render itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.media.probe import MediaInfo, ProbeError, probe_media

DEFAULT_DURATION_TOLERANCE_MS = 750
DEFAULT_ASPECT_RATIO = 9 / 16
DEFAULT_ASPECT_TOLERANCE = 0.01


class RenderValidationError(Exception):
    """A rendered file does not match the expected output contract."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.RENDER_FAILED,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def validate_render(
    path: str | Path,
    *,
    expected_duration_ms: int | None = None,
    tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expect_audio: bool = True,
    settings: Settings | None = None,
    ffprobe_path: Path | None = None,
) -> MediaInfo:
    """Probe ``path`` and check it against the expected clip contract.

    Returns the :class:`MediaInfo` on success so callers can persist it without
    probing twice.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise RenderValidationError(
            f"Rendered file not found: {resolved}",
            details={"path": str(resolved), "reason": "missing"},
        )
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:  # pragma: no cover - race with a deleting process
        raise RenderValidationError(
            f"Rendered file is not readable: {resolved}",
            details={"path": str(resolved), "reason": "unreadable", "error": str(exc)},
        ) from exc
    if size_bytes == 0:
        raise RenderValidationError(
            f"Rendered file is empty: {resolved}",
            details={"path": str(resolved), "reason": "empty", "size_bytes": 0},
        )

    try:
        info = probe_media(resolved, ffprobe_path=ffprobe_path, settings=settings)
    except ProbeError as exc:
        raise RenderValidationError(
            f"Rendered file could not be probed: {exc.message}",
            details={
                "path": str(resolved),
                "reason": "probe_failed",
                "cause": str(exc.code),
                "probe_details": exc.details,
            },
        ) from exc

    if expected_duration_ms is not None:
        delta = abs(info.duration_ms - expected_duration_ms)
        if delta > tolerance_ms:
            raise RenderValidationError(
                f"Rendered duration {info.duration_ms} ms is outside "
                f"{tolerance_ms} ms of the expected {expected_duration_ms} ms",
                details={
                    "expected_ms": expected_duration_ms,
                    "actual_ms": info.duration_ms,
                    "tolerance_ms": tolerance_ms,
                    "delta_ms": delta,
                },
            )

    if expected_width is not None and info.width != expected_width:
        raise RenderValidationError(
            f"Rendered width {info.width} px does not match the expected {expected_width} px",
            details={"expected_width": expected_width, "actual_width": info.width},
        )
    if expected_height is not None and info.height != expected_height:
        raise RenderValidationError(
            f"Rendered height {info.height} px does not match the expected {expected_height} px",
            details={"expected_height": expected_height, "actual_height": info.height},
        )

    if expect_audio and not info.has_audio:
        raise RenderValidationError(
            "Rendered file has no audio stream",
            details={"path": str(resolved), "reason": "missing_audio"},
        )
    return info


def validate_aspect_ratio(
    info: MediaInfo,
    *,
    target_ratio: float = DEFAULT_ASPECT_RATIO,
    tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> None:
    """Raise when ``info`` is not (approximately) the target aspect ratio."""
    if info.height <= 0 or info.width <= 0:
        raise RenderValidationError(
            "Rendered file has invalid dimensions",
            details={"width": info.width, "height": info.height},
        )
    actual = info.aspect_ratio
    if abs(actual - target_ratio) > tolerance:
        raise RenderValidationError(
            f"Rendered aspect ratio {actual:.4f} is not within {tolerance} of {target_ratio:.4f}",
            details={
                "expected_ratio": target_ratio,
                "actual_ratio": actual,
                "tolerance": tolerance,
                "width": info.width,
                "height": info.height,
            },
        )


__all__ = [
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_ASPECT_TOLERANCE",
    "DEFAULT_DURATION_TOLERANCE_MS",
    "RenderValidationError",
    "validate_aspect_ratio",
    "validate_render",
]
