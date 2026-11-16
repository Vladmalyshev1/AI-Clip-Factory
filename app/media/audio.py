"""Audio extraction and 16 kHz WAV helpers (M3.1).

FFmpeg produces a normalised mono 16 kHz PCM16 WAV that speech recognition can
consume. The decoded audio is **streamed to disk, never loaded whole** by the
extraction path: FFmpeg writes ``audio.wav`` incrementally, so a three-hour VOD
never has to fit in RAM. Only the short, already-sliced chunk WAVs are read into
memory later by :func:`read_audio_16k` (a 5 minute 16 kHz mono chunk is ~19 MB).

FFmpeg's sine/AAC sample rates and container padding make the extracted duration
differ slightly from the video duration, which is why callers compare existing
audio with :func:`wav_matches` instead of trusting a database column.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

from app.media.runner import FFmpegRunner, RenderResult

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: Final = 16000
TARGET_CHANNELS: Final = 1
TARGET_CODEC: Final = "pcm_s16le"
AUDIO_FILENAME: Final = "audio.wav"
TARGET_SAMPLE_WIDTH: Final = 2
DEFAULT_DURATION_TOLERANCE_MS: Final = 50


def build_extract_audio_command(
    source_path: Path,
    output_path: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    audio_stream: int = 0,
) -> list[str]:
    """Return FFmpeg arguments for extracting one audio stream to PCM WAV.

    The list excludes the global flags (``-hide_banner``, ``-progress``, ...)
    that :class:`app.media.runner.FFmpegRunner` prepends. Video and subtitle
    streams are dropped explicitly so the output is audio-only.
    """
    return [
        "-y",
        "-i",
        str(source_path),
        "-map",
        f"0:a:{audio_stream}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        TARGET_CODEC,
        "-f",
        "wav",
        str(output_path),
    ]


def extract_audio(
    runner: FFmpegRunner,
    source_path: Path,
    output_path: Path,
    *,
    duration_ms: int | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> RenderResult:
    """Extract ``source_path``'s first audio stream into a mono 16 kHz WAV.

    Audio is **streamed to disk by FFmpeg, never loaded whole**: the source may
    be a multi-hour recording and only ``output_path`` grows on disk. The parent
    directory is created first (``incoming/`` or a per-job work directory may
    not exist yet). ``duration_ms`` is the *source* duration, used only to turn
    FFmpeg progress into a fraction.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return runner.run(
        build_extract_audio_command(source_path, output),
        duration_ms=duration_ms,
        on_progress=on_progress,
        cancel_event=cancel_event,
        cancel_check=cancel_check,
    )


def _wav_metadata(path: Path) -> tuple[int, int, int] | None:
    """Return ``(framerate, channels, frames)`` or ``None`` when unreadable."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getframerate(), handle.getnchannels(), handle.getnframes()
    except (OSError, EOFError, ValueError, wave.Error):
        return None


def wav_duration_ms(path: Path) -> int | None:
    """Return a WAV's duration in milliseconds, or ``None`` when unreadable.

    Uses the standard library ``wave`` module only (header + frame count), so
    this never decodes samples. Missing, truncated and non-WAV files all return
    ``None`` instead of raising.
    """
    metadata = _wav_metadata(path)
    if metadata is None:
        return None
    framerate, _channels, frames = metadata
    if framerate <= 0 or frames < 0:
        return None
    return round(frames * 1000 / framerate)


def wav_matches(
    path: Path,
    expected_ms: int,
    *,
    tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
) -> bool:
    """Return whether ``path`` is a reusable 16 kHz mono WAV of the right length.

    Existing audio may be reused only when its duration is within
    ``tolerance_ms`` of ``expected_ms`` (AAC padding shifts the real duration by
    a few milliseconds) *and* it already has the mono 16 kHz PCM16 shape that
    :func:`read_audio_16k` requires.
    """
    metadata = _wav_metadata(path)
    if metadata is None:
        return False
    framerate, channels, frames = metadata
    if framerate != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS:
        return False
    duration_ms = round(frames * 1000 / framerate)
    return abs(duration_ms - expected_ms) <= tolerance_ms


def read_audio_16k(path: Path) -> np.ndarray:
    """Decode a mono 16 kHz WAV into a 1-D ``float32`` array.

    No resampling happens here: a file with any other sample rate, or with more
    than one channel, is rejected so a mis-encoded chunk cannot silently feed
    Whisper at the wrong speed. Only short sliced chunks are read this way; the
    full extracted track is streamed to disk by FFmpeg.
    """
    data, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
    if samplerate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, got {samplerate} Hz: {path}"
        )
    audio = np.asarray(data)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}: {path}")
    return audio


def remove_audio(path: Path) -> None:
    """Delete an extracted WAV, tolerating a missing file or a Windows lock."""
    target = Path(path)
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete audio file %s: %s", target, exc)
    else:
        logger.debug("Deleted audio file %s", target)


__all__ = [
    "AUDIO_FILENAME",
    "DEFAULT_DURATION_TOLERANCE_MS",
    "TARGET_CHANNELS",
    "TARGET_CODEC",
    "TARGET_SAMPLE_RATE",
    "build_extract_audio_command",
    "extract_audio",
    "read_audio_16k",
    "remove_audio",
    "wav_duration_ms",
    "wav_matches",
]
