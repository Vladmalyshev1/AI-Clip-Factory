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


def can_transition_job(current: JobStatus | str, target: JobStatus | str) -> bool:
    """Return whether a job may move from ``current`` to ``target``."""
    return parse_job_status(target) in JOB_TRANSITIONS[parse_job_status(current)]


def can_transition_stage(current: StageStatus | str, target: StageStatus | str) -> bool:
    """Return whether a stage may move from ``current`` to ``target``."""
    return parse_stage_status(target) in STAGE_TRANSITIONS[parse_stage_status(current)]
