"""Tests for segment normalisation and the cancellable Whisper engine (M3.2).

No model is constructed and no audio is decoded: a fake model produces raw
segments and records the exact kwargs the engine passes to faster-whisper.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from app.ai.asr import (
    TEMPERATURE_FALLBACKS,
    AsrChunkResult,
    AsrError,
    AsrSegment,
    WhisperAsrEngine,
    get_engine,
    normalize_segments,
)
from app.ai.model_manager import reset_manager
from app.core.config import Settings
from app.core.errors import StageCancelled


@dataclass
class FakeRawSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.25
    no_speech_prob: float = 0.01


@dataclass
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration_after_vad: float = 4.0


class FakeModel:
    def __init__(
        self,
        segments: list[FakeRawSegment],
        *,
        info: FakeInfo | None = None,
        fail_after: int | None = None,
        fail_transcribe: bool = False,
    ) -> None:
        self.segments = segments
        self.info = info or FakeInfo()
        self.fail_after = fail_after
        self.fail_transcribe = fail_transcribe
        self.calls: list[dict[str, Any]] = []
        self.consumed = 0
        self.closed = False

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, FakeInfo]:
        self.calls.append({"audio": audio, **kwargs})
        if self.fail_transcribe:
            raise RuntimeError("model refused to decode")
        return self._generate(), self.info

    def _generate(self) -> Any:
        try:
            for index, segment in enumerate(self.segments):
                if self.fail_after is not None and index >= self.fail_after:
                    raise RuntimeError("decode exploded")
                self.consumed += 1
                yield segment
        finally:
            self.closed = True


class FakeManager:
    def __init__(self, model: FakeModel) -> None:
        self.model = "fake-model"
        self._model = model

    def get_model(self) -> FakeModel:
        return self._model


def _engine(model: FakeModel, **kwargs: Any) -> WhisperAsrEngine:
    return WhisperAsrEngine(FakeManager(model), **kwargs)


def _audio() -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Iterator[None]:
    reset_manager()
    yield
    reset_manager()


@pytest.mark.unit
def test_normalize_segments_offsets_the_spec_example() -> None:
    raw = [FakeRawSegment(15.0, 18.0, " hello ")]

    segments = normalize_segments(raw, offset_ms=300_000, chunk_end_ms=600_000)

    assert segments == [AsrSegment(315_000, 318_000, "hello", -0.25, 0.01)]


@pytest.mark.unit
def test_normalize_segments_clamps_both_edges() -> None:
    raw = [
        FakeRawSegment(-1.0, 1.0, "early"),
        FakeRawSegment(299.5, 305.0, "late"),
    ]

    segments = normalize_segments(raw, offset_ms=300_000, chunk_end_ms=600_000)

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (300_000, 301_000),
        (599_500, 600_000),
    ]


@pytest.mark.unit
def test_normalize_segments_drops_empty_text_and_keeps_order() -> None:
    raw = [
        FakeRawSegment(0.0, 1.0, "first"),
        FakeRawSegment(1.0, 2.0, "   "),
        FakeRawSegment(2.0, 3.0, "\n\t"),
        FakeRawSegment(3.0, 4.0, " second "),
    ]

    segments = normalize_segments(raw, offset_ms=0, chunk_end_ms=600_000)

    assert [segment.text for segment in segments] == ["first", "second"]
    assert [segment.start_ms for segment in segments] == [0, 3000]


@pytest.mark.unit
def test_normalize_segments_forces_a_positive_duration() -> None:
    raw = [
        FakeRawSegment(10.0, 10.0, "zero"),
        FakeRawSegment(20.0, 19.0, "backwards"),
    ]

    segments = normalize_segments(raw, offset_ms=0, chunk_end_ms=600_000)

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (10_000, 10_001),
        (20_000, 20_000 + 1),
    ]


@pytest.mark.unit
def test_normalize_segments_never_reaches_beyond_the_chunk_end() -> None:
    raw = [FakeRawSegment(599.999, 700.0, "tail")]

    segments = normalize_segments(raw, offset_ms=0, chunk_end_ms=600_000)

    assert [(item.start_ms, item.end_ms) for item in segments] == [(599_999, 600_000)]


@pytest.mark.unit
def test_normalize_segments_drops_segments_starting_at_or_after_the_end() -> None:
    raw = [
        FakeRawSegment(9.999, 11.0, "starts before"),
        FakeRawSegment(10.0, 12.0, "starts at"),
        FakeRawSegment(12.0, 13.0, "starts after"),
    ]

    segments = normalize_segments(raw, offset_ms=0, chunk_end_ms=10_000)

    assert [segment.text for segment in segments] == ["starts before"]
    assert (segments[0].start_ms, segments[0].end_ms) == (9_999, 10_000)


@pytest.mark.unit
def test_whisper_engine_passes_every_argument_explicitly() -> None:
    model = FakeModel(
        [
            FakeRawSegment(0.5, 1.5, " hello "),
            FakeRawSegment(1.5, 2.0, "   "),
        ]
    )
    engine = _engine(model)
    audio = _audio()

    result = engine.transcribe(audio, offset_ms=300_000, duration_ms=300_000)

    assert isinstance(result, AsrChunkResult)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["audio"] is audio
    assert call["language"] == "en"
    assert call["beam_size"] == 5
    assert call["best_of"] == 5
    assert call["vad_filter"] is True
    assert call["without_timestamps"] is False
    assert call["word_timestamps"] is False
    assert call["condition_on_previous_text"] is True
    assert call["log_progress"] is False
    assert call["temperature"] == list(TEMPERATURE_FALLBACKS)
    assert set(call) == {
        "audio",
        "language",
        "beam_size",
        "best_of",
        "vad_filter",
        "without_timestamps",
        "word_timestamps",
        "condition_on_previous_text",
        "log_progress",
        "temperature",
    }

    assert result.segments == (AsrSegment(300_500, 301_500, "hello", -0.25, 0.01),)
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.99)
    assert result.duration_ms == 300_000
    assert result.duration_after_vad_ms == 4000
    assert result.model_name == "fake-model"
    assert result.elapsed_s >= 0.0


@pytest.mark.unit
def test_whisper_engine_honours_an_explicit_language() -> None:
    model = FakeModel([FakeRawSegment(0.0, 1.0, "hallo")])
    engine = _engine(model, language="en")

    engine.transcribe(_audio(), offset_ms=0, duration_ms=1000, language="de")

    assert model.calls[0]["language"] == "de"


@pytest.mark.unit
def test_whisper_engine_uses_configured_options() -> None:
    model = FakeModel([])
    engine = _engine(
        model,
        beam_size=2,
        vad_filter=False,
        condition_on_previous_text=False,
        language="fr",
    )

    engine.transcribe(_audio(), offset_ms=0, duration_ms=1000)

    call = model.calls[0]
    assert call["beam_size"] == 2
    assert call["best_of"] == 2
    assert call["vad_filter"] is False
    assert call["condition_on_previous_text"] is False
    assert call["language"] == "fr"


@pytest.mark.unit
def test_cancellation_propagates_and_closes_the_generator() -> None:
    model = FakeModel(
        [FakeRawSegment(float(index), index + 0.5, f"part {index}") for index in range(25)]
    )
    engine = _engine(model)
    checks: list[int] = []

    def should_cancel() -> None:
        checks.append(1)
        raise StageCancelled

    with pytest.raises(StageCancelled):
        engine.transcribe(
            _audio(),
            offset_ms=0,
            duration_ms=300_000,
            should_cancel=should_cancel,
        )

    assert len(checks) == 1
    assert model.consumed == 10
    assert model.closed is True


@pytest.mark.unit
def test_generator_failure_maps_to_asr_error() -> None:
    model = FakeModel(
        [FakeRawSegment(float(index), index + 0.5, f"part {index}") for index in range(5)],
        fail_after=2,
    )
    engine = _engine(model)

    with pytest.raises(AsrError) as caught:
        engine.transcribe(_audio(), offset_ms=0, duration_ms=300_000)

    error = caught.value
    assert "decode exploded" in error.message
    assert error.details["reason"] == "segment_iteration"
    assert error.details["segments"] == 2
    assert error.details["model"] == "fake-model"
    assert model.closed is True


@pytest.mark.unit
def test_transcribe_failure_maps_to_asr_error() -> None:
    model = FakeModel([], fail_transcribe=True)
    engine = _engine(model)

    with pytest.raises(AsrError) as caught:
        engine.transcribe(_audio(), offset_ms=1000, duration_ms=2000)

    assert caught.value.details["reason"] == "transcribe_call"
    assert caught.value.details["offset_ms"] == 1000


@pytest.mark.unit
def test_get_engine_reads_the_asr_settings() -> None:
    settings = Settings(
        _env_file=None,
        asr_beam_size=7,
        asr_vad_filter=False,
        asr_condition_on_previous_text=False,
        asr_language="de",
    )

    engine = get_engine(settings)

    assert isinstance(engine, WhisperAsrEngine)
    assert engine.model_name == settings.asr_model
    assert engine.beam_size == 7
    assert engine.vad_filter is False
    assert engine.condition_on_previous_text is False
    assert engine.language == "de"
