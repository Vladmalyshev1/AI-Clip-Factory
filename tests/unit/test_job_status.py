"""Job/stage status transitions and error codes."""

from __future__ import annotations

import pytest
from app.core.errors import (
    AppError,
    ErrorCode,
    ensure_job_transition,
    ensure_stage_transition,
    parse_job_status,
    parse_stage_status,
)
from app.core.status import (
    JOB_TERMINAL_STATUSES,
    JobStatus,
    StageStatus,
    can_transition_job,
    can_transition_stage,
)

ALLOWED_JOB_TRANSITIONS = [
    (JobStatus.QUEUED, JobStatus.RUNNING),
    (JobStatus.QUEUED, JobStatus.CANCELED),
    (JobStatus.RUNNING, JobStatus.SUCCEEDED),
    (JobStatus.RUNNING, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.CANCELED),
    (JobStatus.CANCELED, JobStatus.QUEUED),
    (JobStatus.FAILED, JobStatus.QUEUED),
]

FORBIDDEN_JOB_TRANSITIONS = [
    (JobStatus.QUEUED, JobStatus.SUCCEEDED),
    (JobStatus.QUEUED, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.QUEUED),
    (JobStatus.SUCCEEDED, JobStatus.QUEUED),
    (JobStatus.SUCCEEDED, JobStatus.RUNNING),
    (JobStatus.CANCELED, JobStatus.RUNNING),
    (JobStatus.FAILED, JobStatus.SUCCEEDED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.QUEUED),
]


@pytest.mark.unit
@pytest.mark.parametrize(("current", "target"), ALLOWED_JOB_TRANSITIONS)
def test_allowed_job_transitions(current: JobStatus, target: JobStatus) -> None:
    assert can_transition_job(current, target) is True
    ensure_job_transition(current, target)


@pytest.mark.unit
@pytest.mark.parametrize(("current", "target"), FORBIDDEN_JOB_TRANSITIONS)
def test_forbidden_job_transitions_raise_invalid_state(current: JobStatus, target: JobStatus) -> None:
    assert can_transition_job(current, target) is False
    with pytest.raises(AppError) as excinfo:
        ensure_job_transition(current, target)
    assert excinfo.value.code is ErrorCode.INVALID_STATE_TRANSITION
    assert excinfo.value.status_code == 409
    assert "Invalid job transition" in excinfo.value.message


@pytest.mark.unit
def test_terminal_job_statuses_allow_only_retry() -> None:
    assert {
        JobStatus.SUCCEEDED,
        JobStatus.CANCELED,
        JobStatus.FAILED,
    } == JOB_TERMINAL_STATUSES
    succeeded_targets = [target for target in JobStatus if can_transition_job(JobStatus.SUCCEEDED, target)]
    assert succeeded_targets == []
    for status in (JobStatus.CANCELED, JobStatus.FAILED):
        targets = [target for target in JobStatus if can_transition_job(status, target)]
        assert targets == [JobStatus.QUEUED]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("current", "target"),
    [
        (StageStatus.PENDING, StageStatus.RUNNING),
        (StageStatus.PENDING, StageStatus.CANCELED),
        (StageStatus.RUNNING, StageStatus.SUCCEEDED),
        (StageStatus.RUNNING, StageStatus.FAILED),
        (StageStatus.RUNNING, StageStatus.CANCELED),
        (StageStatus.RUNNING, StageStatus.PENDING),
        (StageStatus.FAILED, StageStatus.PENDING),
        (StageStatus.CANCELED, StageStatus.PENDING),
    ],
)
def test_allowed_stage_transitions(current: StageStatus, target: StageStatus) -> None:
    assert can_transition_stage(current, target) is True
    ensure_stage_transition(current, target)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("current", "target"),
    [
        (StageStatus.PENDING, StageStatus.SUCCEEDED),
        (StageStatus.SUCCEEDED, StageStatus.RUNNING),
        (StageStatus.SUCCEEDED, StageStatus.PENDING),
        (StageStatus.SKIPPED, StageStatus.RUNNING),
        (StageStatus.FAILED, StageStatus.SUCCEEDED),
    ],
)
def test_forbidden_stage_transitions_raise_invalid_state(
    current: StageStatus, target: StageStatus
) -> None:
    assert can_transition_stage(current, target) is False
    with pytest.raises(AppError) as excinfo:
        ensure_stage_transition(current, target)
    assert excinfo.value.code is ErrorCode.INVALID_STATE_TRANSITION
    assert excinfo.value.status_code == 409


@pytest.mark.unit
def test_error_codes_have_expected_values_and_default_statuses() -> None:
    assert ErrorCode.JOB_NOT_FOUND == "JOB_NOT_FOUND"
    assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
    assert ErrorCode.INVALID_STATE_TRANSITION.value == "INVALID_STATE_TRANSITION"

    not_found = AppError(ErrorCode.JOB_NOT_FOUND, "missing")
    assert not_found.status_code == 404
    invalid_state = AppError(ErrorCode.INVALID_STATE_TRANSITION, "bad state")
    assert invalid_state.status_code == 409
    internal = AppError(ErrorCode.INTERNAL_ERROR, "boom")
    assert internal.status_code == 500


@pytest.mark.unit
def test_status_values_are_unique_and_lowercase() -> None:
    job_values = [status.value for status in JobStatus]
    stage_values = [status.value for status in StageStatus]
    assert len(job_values) == len(set(job_values))
    assert len(stage_values) == len(set(stage_values))
    assert all(value == value.lower() for value in job_values + stage_values)


@pytest.mark.unit
def test_parse_helpers_accept_enums_strings_and_reject_unknown_values() -> None:
    assert parse_job_status("running") is JobStatus.RUNNING
    assert parse_job_status(JobStatus.FAILED) is JobStatus.FAILED
    assert parse_stage_status("succeeded") is StageStatus.SUCCEEDED
    with pytest.raises(ValueError):
        parse_job_status("exploded")
    with pytest.raises(ValueError):
        parse_stage_status("exploded")
