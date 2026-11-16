"""Chunk planning and slicing for resumable speech recognition (M3.1).

Planning is pure: :func:`plan_chunks` turns a duration and a chunk size into
:class:`ChunkPlan` windows that tile ``[0, duration_ms)`` without gaps or
zero-length pieces. Slicing is a thin FFmpeg wrapper around
:class:`app.media.runner.FFmpegRunner`; the input WAV is read by FFmpeg and the
output chunk is **streamed to disk**, so only one short chunk ever exists in
memory while transcribing.

This module deliberately knows nothing about the database, settings or ASR: the
pipeline stage owns persistence and decides which plans still need work.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.media.runner import FFmpegRunner, RenderResult

logger = logging.getLogger(__name__)

CHUNK_FILENAME_PREFIX: Final = "chunk-"
CHUNK_FILENAME_SUFFIX: Final = ".wav"
CHUNK_FILENAME_DIGITS: Final = 5
CHUNK_GLOB: Final = "chunk-*.wav"


@dataclass(frozen=True)
class ChunkPlan:
    """One half-open ``[start_ms, end_ms)`` window of the extracted audio."""

    index: int
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")
        if self.start_ms < 0:
            raise ValueError(f"start_ms must be non-negative, got {self.start_ms}")
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )

    @property
    def duration_ms(self) -> int:
        """Length of the window in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def start_s(self) -> float:
        """Window start in seconds, for FFmpeg's ``-ss``."""
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        """Window end in seconds (useful for logging and tests)."""
        return self.end_ms / 1000.0


def plan_chunks(duration_ms: int, chunk_seconds: int) -> list[ChunkPlan]:
    """Tile ``[0, duration_ms)`` into chunks of at most ``chunk_seconds``.

    Chunk ``i`` covers ``[i * chunk_seconds * 1000, min(duration_ms,
    (i + 1) * chunk_seconds * 1000))``. The final chunk is partial when the
    duration is not an exact multiple. A non-positive duration or chunk size
    yields no work (``[]``); zero-length chunks are never produced.
    """
    if duration_ms <= 0 or chunk_seconds <= 0:
        return []
    step_ms = chunk_seconds * 1000
    plans: list[ChunkPlan] = []
    start_ms = 0
    index = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + step_ms)
        if end_ms > start_ms:
            plans.append(ChunkPlan(index=index, start_ms=start_ms, end_ms=end_ms))
        start_ms = end_ms
        index += 1
    return plans


def chunk_count(duration_ms: int, chunk_seconds: int) -> int:
    """Return how many chunks :func:`plan_chunks` would produce."""
    if duration_ms <= 0 or chunk_seconds <= 0:
        return 0
    step_ms = chunk_seconds * 1000
    return (duration_ms + step_ms - 1) // step_ms


def chunk_filename(index: int) -> str:
    """Return the stable, sortable file name of chunk ``index``."""
    return f"{CHUNK_FILENAME_PREFIX}{index:0{CHUNK_FILENAME_DIGITS}d}{CHUNK_FILENAME_SUFFIX}"


def chunk_path(directory: Path, index: int) -> Path:
    """Return the path of chunk ``index`` inside ``directory``."""
    return Path(directory) / chunk_filename(index)


def build_slice_command(
    source_wav: Path,
    output_wav: Path,
    plan: ChunkPlan,
    *,
    sample_rate: int = 16000,
) -> list[str]:
    """Return FFmpeg arguments that cut ``plan`` out of ``source_wav``.

    ``-ss`` and ``-t`` are placed **before** ``-i`` (input seeking) and the
    chunk is re-encoded to PCM16 rather than stream-copied: verified slicing is
    sample-accurate that way, while ``-c copy`` grows the tail by ~8 ms.
    """
    return [
        "-y",
        "-ss",
        f"{plan.start_s:.3f}",
        "-t",
        f"{plan.duration_ms / 1000.0:.3f}",
        "-i",
        str(source_wav),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(output_wav),
    ]


def slice_chunk(
    runner: FFmpegRunner,
    source_wav: Path,
    output_wav: Path,
    plan: ChunkPlan,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> RenderResult:
    """Cut one chunk with ``runner`` (the output is streamed to disk)."""
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    return runner.run(
        build_slice_command(source_wav, output, plan),
        duration_ms=plan.duration_ms,
        cancel_check=cancel_check,
    )


def remove_file(path: Path) -> None:
    """Delete one file idempotently, tolerating a transient Windows lock."""
    target = Path(path)
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete chunk file %s: %s", target, exc)
    else:
        logger.debug("Deleted chunk file %s", target)


def cleanup_chunks(directory: Path) -> None:
    """Delete every ``chunk-*.wav`` in ``directory``; safe to call repeatedly."""
    target = Path(directory)
    if not target.is_dir():
        return
    for path in sorted(target.glob(CHUNK_GLOB)):
        if path.is_file():
            remove_file(path)


__all__ = [
    "CHUNK_GLOB",
    "ChunkPlan",
    "build_slice_command",
    "chunk_count",
    "chunk_filename",
    "chunk_path",
    "cleanup_chunks",
    "plan_chunks",
    "remove_file",
    "slice_chunk",
]
