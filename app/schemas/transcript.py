"""Request/response schemas for speech-recognition transcripts (M3).

The API exposes the transcript row plus two derived aggregates: progress over
the chunk rows (so a 4 hour VOD can show "chunk 27 of 48") and the number of
stored segments (so a client knows whether paging is worth it).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.models.transcript import Transcript, TranscriptSegment


class TranscriptErrorRead(BaseModel):
    """Persisted recognition error of a transcript or a chunk."""

    code: str
    message: str


class TranscriptProgressRead(BaseModel):
    """Chunk-level progress snapshot of a transcript."""

    chunks_total: int
    chunks_completed: int
    chunks_running: int
    chunks_failed: int
    audio_ms_total: int
    audio_ms_processed: int
    progress: float
    current_chunk_index: int | None = None


class TranscriptSegmentRead(BaseModel):
    """One recognised segment with absolute media timestamps."""

    id: int
    chunk_index: int
    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TranscriptRead(BaseModel):
    """Full transcript representation returned by the API."""

    id: str
    video_id: str
    language: str
    model_name: str
    status: str
    duration_ms: int
    chunk_seconds: int
    options: dict[str, Any] | None = None
    error: TranscriptErrorRead | None = None
    segment_count: int
    progress: TranscriptProgressRead
    job_id: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    created_at_ms: int
    updated_at_ms: int


class TranscribeRequest(BaseModel):
    """Body of ``POST /videos/{id}/transcribe``.

    ``force`` throws away a finished transcript and starts over; without it a
    completed transcript is left alone (409) and an interrupted one is resumed.
    """

    force: bool = Field(default=False)


def segment_to_read(segment: TranscriptSegment) -> TranscriptSegmentRead:
    """Convert a segment row into its API representation."""
    return TranscriptSegmentRead(
        id=segment.id,
        chunk_index=segment.chunk_index,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        text=segment.text,
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
    )


def transcript_to_read(
    transcript: Transcript,
    *,
    progress: TranscriptProgressRead,
    segment_count: int,
    job_id: str | None = None,
) -> TranscriptRead:
    """Convert a transcript row into its API representation."""
    error = None
    if transcript.error_code or transcript.error_message:
        error = TranscriptErrorRead(
            code=transcript.error_code or "INTERNAL_ERROR",
            message=transcript.error_message or "",
        )
    return TranscriptRead(
        id=transcript.id,
        video_id=transcript.video_id,
        language=transcript.language,
        model_name=transcript.model_name,
        status=transcript.status,
        duration_ms=transcript.duration_ms,
        chunk_seconds=transcript.chunk_seconds,
        options=load_options(transcript.options_json),
        error=error,
        segment_count=segment_count,
        progress=progress,
        job_id=job_id,
        started_at_ms=transcript.started_at,
        finished_at_ms=transcript.finished_at,
        created_at_ms=transcript.created_at,
        updated_at_ms=transcript.updated_at,
    )


def load_options(value: str | None) -> dict[str, Any] | None:
    """Parse the stored ``options_json`` snapshot, tolerating malformed JSON."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "TranscribeRequest",
    "TranscriptErrorRead",
    "TranscriptProgressRead",
    "TranscriptRead",
    "TranscriptSegmentRead",
    "load_options",
    "segment_to_read",
    "transcript_to_read",
]
