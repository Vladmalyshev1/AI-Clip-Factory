"""Job and stage lifecycle enums plus the allowed transition tables.

This module is deliberately dependency-free so ORM models and the worker can
import the statuses without pulling in the web framework. It is the single
source of truth for status literals.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class JobStatus(StrEnum):
    """Lifecycle states of a job row."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StageStatus(StrEnum):
    """Lifecycle states of a job stage row."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


JOB_TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.CANCELED, JobStatus.FAILED}
)


class VideoStatus(StrEnum):
    """Lifecycle states of an ingested source recording."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


VIDEO_TRANSITIONS: Final[dict[VideoStatus, frozenset[VideoStatus]]] = {
    VideoStatus.PENDING: frozenset({VideoStatus.READY, VideoStatus.FAILED}),
    VideoStatus.READY: frozenset(),
    VideoStatus.FAILED: frozenset({VideoStatus.PENDING}),
}


class UploadStatus(StrEnum):
    """Lifecycle states of a resumable chunked upload session."""

    UPLOADING = "uploading"
    COMPLETED = "completed"
    CANCELED = "canceled"
    EXPIRED = "expired"


UPLOAD_TERMINAL_STATUSES: Final[frozenset[UploadStatus]] = frozenset(
    {UploadStatus.COMPLETED, UploadStatus.CANCELED, UploadStatus.EXPIRED}
)

UPLOAD_TRANSITIONS: Final[dict[UploadStatus, frozenset[UploadStatus]]] = {
    UploadStatus.UPLOADING: frozenset(
        {UploadStatus.COMPLETED, UploadStatus.CANCELED, UploadStatus.EXPIRED}
    ),
    UploadStatus.COMPLETED: frozenset(),
    UploadStatus.CANCELED: frozenset(),
    UploadStatus.EXPIRED: frozenset(),
}


class SourceKind(StrEnum):
    """Provenance of an immutable source recording."""

    UPLOAD = "upload"
    IMPORT = "import"
    IMPORT_IN_PLACE = "import_in_place"


class TranscriptStatus(StrEnum):
    """Lifecycle states of a speech-to-text transcript of one video."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TranscriptChunkStatus(StrEnum):
    """Lifecycle states of one resumable audio chunk of a transcript."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


#: Chunks that still need work (or need to be retried).
UNFINISHED_CHUNK_STATUSES: Final[frozenset[TranscriptChunkStatus]] = frozenset(
    {
        TranscriptChunkStatus.PENDING,
        TranscriptChunkStatus.RUNNING,
        TranscriptChunkStatus.FAILED,
    }
)

TRANSCRIPT_TRANSITIONS: Final[dict[TranscriptStatus, frozenset[TranscriptStatus]]] = {
    TranscriptStatus.PENDING: frozenset(
        {TranscriptStatus.RUNNING, TranscriptStatus.CANCELED, TranscriptStatus.FAILED}
    ),
    TranscriptStatus.RUNNING: frozenset(
        {
            # A transcript left ``running`` by a crashed worker is resumed.
            TranscriptStatus.RUNNING,
            TranscriptStatus.COMPLETED,
            TranscriptStatus.FAILED,
            TranscriptStatus.CANCELED,
            TranscriptStatus.PENDING,
        }
    ),
    # A failed or interrupted transcript is resumed, never recreated.
    TranscriptStatus.FAILED: frozenset(
        {
            TranscriptStatus.PENDING,
            TranscriptStatus.RUNNING,
            # Recording why a run failed again must stay idempotent.
            TranscriptStatus.FAILED,
        }
    ),
    TranscriptStatus.CANCELED: frozenset(
        {
            TranscriptStatus.PENDING,
            # A canceled run that then hits an infrastructure error.
            TranscriptStatus.FAILED,
        }
    ),
    # Only an explicit force re-run resets a completed transcript.
    TranscriptStatus.COMPLETED: frozenset({TranscriptStatus.PENDING}),
}

CHUNK_TRANSITIONS: Final[dict[TranscriptChunkStatus, frozenset[TranscriptChunkStatus]]] = {
    TranscriptChunkStatus.PENDING: frozenset(
        {
            TranscriptChunkStatus.RUNNING,
            TranscriptChunkStatus.COMPLETED,
            TranscriptChunkStatus.FAILED,
            TranscriptChunkStatus.PENDING,
        }
    ),
    TranscriptChunkStatus.RUNNING: frozenset(
        {
            # A chunk left ``running`` by a crashed worker is claimed again.
            TranscriptChunkStatus.RUNNING,
            TranscriptChunkStatus.COMPLETED,
            TranscriptChunkStatus.FAILED,
            TranscriptChunkStatus.PENDING,
        }
    ),
    # Retries overwrite the previous attempt atomically.
    TranscriptChunkStatus.FAILED: frozenset(
        {
            TranscriptChunkStatus.RUNNING,
            TranscriptChunkStatus.COMPLETED,
            # Recording the same failure twice must stay idempotent.
            TranscriptChunkStatus.FAILED,
        }
    ),
    # A completed chunk is terminal for claiming purposes, but its segments may
    # be replaced idempotently by ``save_chunk_result``.
    TranscriptChunkStatus.COMPLETED: frozenset({TranscriptChunkStatus.COMPLETED}),
}

JOB_TRANSITIONS: Final[dict[JobStatus, frozenset[JobStatus]]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCEL_REQUESTED}
    ),
    JobStatus.CANCEL_REQUESTED: frozenset({JobStatus.CANCELED, JobStatus.FAILED}),
    JobStatus.CANCELED: frozenset({JobStatus.QUEUED}),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
    JobStatus.SUCCEEDED: frozenset(),
}

STAGE_TRANSITIONS: Final[dict[StageStatus, frozenset[StageStatus]]] = {
    StageStatus.PENDING: frozenset(
        {StageStatus.RUNNING, StageStatus.SKIPPED, StageStatus.CANCELED}
    ),
    StageStatus.RUNNING: frozenset(
        {
            StageStatus.SUCCEEDED,
            StageStatus.FAILED,
            StageStatus.CANCELED,
            StageStatus.PENDING,
        }
    ),
    StageStatus.FAILED: frozenset({StageStatus.PENDING}),
    StageStatus.SKIPPED: frozenset(),
    StageStatus.SUCCEEDED: frozenset(),
    StageStatus.CANCELED: frozenset({StageStatus.PENDING}),
}


def parse_job_status(value: JobStatus | str) -> JobStatus:
    """Coerce a stored value into :class:`JobStatus` (raises ``ValueError``)."""
    if isinstance(value, JobStatus):
        return value
    return JobStatus(value)


def parse_stage_status(value: StageStatus | str) -> StageStatus:
    """Coerce a stored value into :class:`StageStatus` (raises ``ValueError``)."""
    if isinstance(value, StageStatus):
        return value
    return StageStatus(value)


def parse_video_status(value: VideoStatus | str) -> VideoStatus:
    """Coerce a stored value into :class:`VideoStatus` (raises ``ValueError``)."""
    if isinstance(value, VideoStatus):
        return value
    return VideoStatus(value)


def can_transition_job(current: JobStatus | str, target: JobStatus | str) -> bool:
    """Return whether a job may move from ``current`` to ``target``."""
    return parse_job_status(target) in JOB_TRANSITIONS[parse_job_status(current)]


def can_transition_stage(current: StageStatus | str, target: StageStatus | str) -> bool:
    """Return whether a stage may move from ``current`` to ``target``."""
    return parse_stage_status(target) in STAGE_TRANSITIONS[parse_stage_status(current)]


def can_transition_video(current: VideoStatus | str, target: VideoStatus | str) -> bool:
    """Return whether a video may move from ``current`` to ``target``."""
    return parse_video_status(target) in VIDEO_TRANSITIONS[parse_video_status(current)]


def parse_upload_status(value: UploadStatus | str) -> UploadStatus:
    """Coerce a stored value into :class:`UploadStatus` (raises ``ValueError``)."""
    if isinstance(value, UploadStatus):
        return value
    return UploadStatus(value)


def can_transition_upload(current: UploadStatus | str, target: UploadStatus | str) -> bool:
    """Return whether an upload may move from ``current`` to ``target``."""
    return parse_upload_status(target) in UPLOAD_TRANSITIONS[parse_upload_status(current)]


def parse_transcript_status(value: TranscriptStatus | str) -> TranscriptStatus:
    """Coerce a stored value into :class:`TranscriptStatus` (raises ``ValueError``)."""
    if isinstance(value, TranscriptStatus):
        return value
    return TranscriptStatus(value)


def parse_chunk_status(value: TranscriptChunkStatus | str) -> TranscriptChunkStatus:
    """Coerce a stored value into :class:`TranscriptChunkStatus`."""
    if isinstance(value, TranscriptChunkStatus):
        return value
    return TranscriptChunkStatus(value)


def can_transition_transcript(
    current: TranscriptStatus | str, target: TranscriptStatus | str
) -> bool:
    """Return whether a transcript may move from ``current`` to ``target``."""
    return parse_transcript_status(target) in TRANSCRIPT_TRANSITIONS[
        parse_transcript_status(current)
    ]


def can_transition_chunk(
    current: TranscriptChunkStatus | str, target: TranscriptChunkStatus | str
) -> bool:
    """Return whether a chunk may move from ``current`` to ``target``."""
    return parse_chunk_status(target) in CHUNK_TRANSITIONS[parse_chunk_status(current)]

