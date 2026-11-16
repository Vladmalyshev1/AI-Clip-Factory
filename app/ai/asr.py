"""Speech recognition around a lazily loaded faster-whisper model (M3.2).

This module performs **no file IO**: the stage decodes a chunk WAV with
``soundfile`` and hands :class:`WhisperAsrEngine` a 1-D ``float32`` array at
16 kHz. That contract exists because faster-whisper 1.2.1 cannot decode a path
or ``BinaryIO`` here (its PyAV call passes an unsupported ``metadata_errors``
keyword), while accepting a NumPy array works.

Timeline handling is explicit: ``vad_filter=True`` restores the original
timeline, so absolute time is ``offset_ms + round(segment.start * 1000)``.
``avg_logprob`` and ``no_speech_prob`` come from faster-whisper's 30 second
decode window and are stored as-is, not "fixed" per segment.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

from app.ai.model_manager import AsrModelManager, get_manager
from app.core.config import Settings

logger = logging.getLogger(__name__)

CANCEL_CHECK_EVERY_SEGMENTS: Final = 10
CANCEL_CHECK_INTERVAL_S: Final = 2.0
TEMPERATURE_FALLBACKS: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class AsrSegment:
    """One transcribed segment on the *absolute* source timeline."""

    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None


@dataclass(frozen=True)
class AsrChunkResult:
    """Everything one ASR call produced for one audio chunk."""

    segments: tuple[AsrSegment, ...]
    language: str | None
    language_probability: float | None
    duration_ms: int
    duration_after_vad_ms: int
    elapsed_s: float
    model_name: str


class RawSegment(Protocol):
    """The subset of faster-whisper's ``Segment`` this layer relies on."""

    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


class AsrEngine(Protocol):
    """Interface the pipeline stage depends on (monkeypatched in tests)."""

    model_name: str

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_ms: int,
        duration_ms: int,
        language: str | None = None,
        should_cancel: Callable[[], None] | None = None,
    ) -> AsrChunkResult:
        """Transcribe one chunk and return absolute-timeline segments."""
        ...


class AsrError(Exception):
    """A transcription call or its lazy segment generator failed."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def normalize_segments(
    raw: Iterable[RawSegment],
    *,
    offset_ms: int,
    chunk_end_ms: int,
) -> list[AsrSegment]:
    """Convert chunk-relative faster-whisper segments to absolute milliseconds.

    Empty/whitespace text is dropped, times are clamped into
    ``[offset_ms, chunk_end_ms]``, and segments that start at or after
    ``chunk_end_ms`` are discarded. A degenerate segment (``end <= start``) gets
    a 1 ms duration, capped at ``chunk_end_ms``. Input order is preserved.
    """
    normalized: list[AsrSegment] = []
    for segment in raw:
        text = segment.text.strip()
        if not text:
            continue
        start_ms = offset_ms + round(segment.start * 1000)
        end_ms = offset_ms + round(segment.end * 1000)
        if start_ms >= chunk_end_ms:
            continue
        start_ms = max(offset_ms, start_ms)
        end_ms = min(chunk_end_ms, end_ms)
        if end_ms <= start_ms:
            end_ms = min(chunk_end_ms, start_ms + 1)
        normalized.append(
            AsrSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
            )
        )
    return normalized


class WhisperAsrEngine:
    """faster-whisper backed :class:`AsrEngine` with cancellable decoding."""

    def __init__(
        self,
        manager: AsrModelManager,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
        language: str | None = "en",
    ) -> None:
        self.manager = manager
        self.model_name = manager.model
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.language = language

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_ms: int,
        duration_ms: int,
        language: str | None = None,
        should_cancel: Callable[[], None] | None = None,
    ) -> AsrChunkResult:
        """Transcribe ``audio`` and return absolute-timeline segments.

        ``should_cancel`` is called every :data:`CANCEL_CHECK_EVERY_SEGMENTS`
        segments and at least every :data:`CANCEL_CHECK_INTERVAL_S` seconds; its
        exception (the stage's cancellation signal) propagates unchanged after
        the lazy generator is closed. Decode failures become :class:`AsrError`.
        """
        started = time.monotonic()
        model = self.manager.get_model()
        effective_language = language if language is not None else self.language
        error_details: dict[str, Any] = {
            "model": self.model_name,
            "offset_ms": offset_ms,
            "duration_ms": duration_ms,
        }
        try:
            segments, info = model.transcribe(
                audio,
                language=effective_language,
                beam_size=self.beam_size,
                best_of=self.beam_size,
                vad_filter=self.vad_filter,
                without_timestamps=False,
                word_timestamps=False,
                condition_on_previous_text=self.condition_on_previous_text,
                log_progress=False,
                temperature=list(TEMPERATURE_FALLBACKS),
            )
        except Exception as exc:
            raise AsrError(
                f"ASR failed to start for the chunk at {offset_ms} ms: {exc}",
                details={
                    **error_details,
                    "reason": "transcribe_call",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ) from exc

        raw_segments: list[RawSegment] = []
        segments_since_check = 0
        last_cancel_check = time.monotonic()
        with contextlib.closing(segments) as stream:
            iterator = iter(stream)
            while True:
                try:
                    segment = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    raise AsrError(
                        f"ASR failed while decoding the chunk at {offset_ms} ms: {exc}",
                        details={
                            **error_details,
                            "reason": "segment_iteration",
                            "segments": len(raw_segments),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    ) from exc
                raw_segments.append(segment)
                segments_since_check += 1
                now = time.monotonic()
                if should_cancel is not None and (
                    segments_since_check >= CANCEL_CHECK_EVERY_SEGMENTS
                    or now - last_cancel_check >= CANCEL_CHECK_INTERVAL_S
                ):
                    segments_since_check = 0
                    last_cancel_check = now
                    should_cancel()

        normalized = normalize_segments(
            raw_segments,
            offset_ms=offset_ms,
            chunk_end_ms=offset_ms + duration_ms,
        )
        after_vad = _optional_float(getattr(info, "duration_after_vad", None))
        duration_after_vad_ms = (
            max(0, round(after_vad * 1000)) if after_vad is not None else duration_ms
        )
        result = AsrChunkResult(
            segments=tuple(normalized),
            language=getattr(info, "language", None),
            language_probability=_optional_float(getattr(info, "language_probability", None)),
            duration_ms=duration_ms,
            duration_after_vad_ms=duration_after_vad_ms,
            elapsed_s=time.monotonic() - started,
            model_name=self.model_name,
        )
        logger.debug(
            "ASR chunk at %d ms: %d segments in %.2fs (%s)",
            offset_ms,
            len(result.segments),
            result.elapsed_s,
            self.model_name,
        )
        return result


def get_engine(settings: Settings) -> AsrEngine:
    """Build the process-wide-engine-style ASR engine for the pipeline stage."""
    return WhisperAsrEngine(
        get_manager(settings),
        beam_size=settings.asr_beam_size,
        vad_filter=settings.asr_vad_filter,
        condition_on_previous_text=settings.asr_condition_on_previous_text,
        language=settings.asr_language,
    )


def transcription_options(settings: Settings) -> dict[str, Any]:
    """Return the ASR configuration that produces a transcript.

    Stored on the transcript row (and echoed by the API) so a finished result
    can be explained later: a transcript made with ``vad_filter`` or a different
    model is not reproducible from the code alone.
    """
    return {
        "language": settings.asr_language,
        "beam_size": settings.asr_beam_size,
        "vad_filter": settings.asr_vad_filter,
        "condition_on_previous_text": settings.asr_condition_on_previous_text,
        "device": settings.asr_device,
        "compute_type": settings.asr_compute_type,
        "chunk_seconds": settings.asr_chunk_seconds,
    }


__all__ = [
    "CANCEL_CHECK_EVERY_SEGMENTS",
    "CANCEL_CHECK_INTERVAL_S",
    "TEMPERATURE_FALLBACKS",
    "AsrChunkResult",
    "AsrEngine",
    "AsrError",
    "AsrSegment",
    "RawSegment",
    "WhisperAsrEngine",
    "get_engine",
    "normalize_segments",
    "transcription_options",
]
