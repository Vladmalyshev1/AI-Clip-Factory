"""Video lifecycle transitions (the enum is shared by ingest and the pipeline)."""

from __future__ import annotations

import pytest
from app.core.errors import AppError, ErrorCode, ensure_video_transition
from app.core.status import (
    UPLOAD_TERMINAL_STATUSES,
    SourceKind,
    UploadStatus,
    VideoStatus,
    can_transition_upload,
    can_transition_video,
    parse_video_status,
)


@pytest.mark.unit
def test_video_status_values_are_stable() -> None:
    assert {status.value for status in VideoStatus} == {"pending", "ready", "failed"}


@pytest.mark.unit
def test_allowed_and_forbidden_video_transitions() -> None:
    assert can_transition_video(VideoStatus.PENDING, VideoStatus.READY)
    assert can_transition_video("pending", "failed")
    assert can_transition_video(VideoStatus.FAILED, VideoStatus.PENDING)
    assert not can_transition_video(VideoStatus.READY, VideoStatus.PENDING)
    assert not can_transition_video(VideoStatus.READY, VideoStatus.FAILED)


@pytest.mark.unit
def test_ensure_video_transition_raises_a_409() -> None:
    with pytest.raises(AppError) as excinfo:
        ensure_video_transition(VideoStatus.READY, VideoStatus.FAILED)
    assert excinfo.value.code is ErrorCode.INVALID_STATE_TRANSITION
    assert excinfo.value.status_code == 409


@pytest.mark.unit
def test_parse_video_status_rejects_unknown_values() -> None:
    assert parse_video_status("ready") is VideoStatus.READY
    with pytest.raises(ValueError):
        parse_video_status("archived")


@pytest.mark.unit
def test_upload_status_is_terminal_once_finished() -> None:
    assert {status.value for status in UploadStatus} == {
        "uploading",
        "completed",
        "canceled",
        "expired",
    }
    assert can_transition_upload(UploadStatus.UPLOADING, UploadStatus.COMPLETED)
    assert can_transition_upload("uploading", "canceled")
    assert not can_transition_upload(UploadStatus.COMPLETED, UploadStatus.UPLOADING)
    assert {
        UploadStatus.COMPLETED,
        UploadStatus.CANCELED,
        UploadStatus.EXPIRED,
    } == UPLOAD_TERMINAL_STATUSES


@pytest.mark.unit
def test_source_kind_values_are_stable() -> None:
    assert {kind.value for kind in SourceKind} == {"upload", "import", "import_in_place"}
