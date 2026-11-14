"""Disk-full detection and the canonical error it maps onto."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest
from app.core.errors import ErrorCode, is_disk_full, storage_error


class _WindowsDiskFullError(OSError):
    """``OSError`` carrying the Windows "disk full" code."""

    winerror = 112


@pytest.mark.unit
def test_enospc_and_winerror_112_are_disk_full() -> None:
    assert is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    assert is_disk_full(_WindowsDiskFullError(errno.EIO, "The disk is full"))


@pytest.mark.unit
def test_other_oserror_is_not_disk_full() -> None:
    assert not is_disk_full(OSError(errno.EACCES, "Permission denied"))
    assert not is_disk_full(OSError(errno.EIO, "I/O error"))


@pytest.mark.unit
def test_storage_error_maps_a_full_disk_to_insufficient_disk() -> None:
    error = storage_error(OSError(errno.ENOSPC, "No space left"), path=Path("C:/var/x.part"))

    assert error.code is ErrorCode.INSUFFICIENT_DISK
    assert error.status_code == 507
    assert error.details["reason"] == "disk_full"
    assert error.details["path"] == str(Path("C:/var/x.part"))
    assert "No space left" in error.details["error"]


@pytest.mark.unit
def test_storage_error_keeps_other_failures_internal() -> None:
    error = storage_error(OSError(errno.EIO, "I/O error"))

    assert error.code is ErrorCode.INTERNAL_ERROR
    assert error.status_code == 500
    assert "reason" not in error.details
