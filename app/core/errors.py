"""Application errors and the canonical error envelope handlers."""

from __future__ import annotations

import errno
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.request_id import REQUEST_ID_HEADER, current_request_id, new_request_id
from app.core.status import (
    JOB_TERMINAL_STATUSES,
    JOB_TRANSITIONS,
    STAGE_TRANSITIONS,
    UPLOAD_TERMINAL_STATUSES,
    UPLOAD_TRANSITIONS,
    VIDEO_TRANSITIONS,
    JobStatus,
    SourceKind,
    StageStatus,
    UploadStatus,
    VideoStatus,
    can_transition_job,
    can_transition_stage,
    can_transition_upload,
    can_transition_video,
    parse_job_status,
    parse_stage_status,
    parse_upload_status,
    parse_video_status,
)

logger = logging.getLogger(__name__)

__all__ = [
    "JOB_TERMINAL_STATUSES",
    "JOB_TRANSITIONS",
    "STAGE_TRANSITIONS",
    "UPLOAD_TERMINAL_STATUSES",
    "UPLOAD_TRANSITIONS",
    "VIDEO_TRANSITIONS",
    "AppError",
    "ErrorCode",
    "JobStatus",
    "SourceKind",
    "StageCancelled",
    "StageStatus",
    "UploadStatus",
    "VideoStatus",
    "can_transition_job",
    "can_transition_stage",
    "can_transition_upload",
    "can_transition_video",
    "ensure_job_transition",
    "ensure_stage_transition",
    "ensure_video_transition",
    "error_response",
    "is_disk_full",
    "parse_job_status",
    "parse_stage_status",
    "parse_upload_status",
    "parse_video_status",
    "register_exception_handlers",
    "resolve_request_id",
    "storage_error",
]


class ErrorCode(StrEnum):
    """Stable machine-readable error identifiers."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    STAGE_NOT_FOUND = "STAGE_NOT_FOUND"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    WORKER_LOST = "WORKER_LOST"
    WORKER_ERROR = "WORKER_ERROR"
    STAGE_FAILED = "STAGE_FAILED"
    VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
    UPLOAD_NOT_FOUND = "UPLOAD_NOT_FOUND"
    UPLOAD_CONFLICT = "UPLOAD_CONFLICT"
    UPLOAD_INCOMPLETE = "UPLOAD_INCOMPLETE"
    UPLOAD_EXPIRED = "UPLOAD_EXPIRED"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
    PROBE_FAILED = "PROBE_FAILED"
    INSUFFICIENT_DISK = "INSUFFICIENT_DISK"
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"
    FFPROBE_NOT_FOUND = "FFPROBE_NOT_FOUND"
    RENDER_FAILED = "RENDER_FAILED"
    RENDER_TIMEOUT = "RENDER_TIMEOUT"
    SUBTITLE_FAILED = "SUBTITLE_FAILED"
    TRANSCRIPT_NOT_FOUND = "TRANSCRIPT_NOT_FOUND"
    TRANSCRIPT_CONFLICT = "TRANSCRIPT_CONFLICT"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    ASR_MODEL_UNAVAILABLE = "ASR_MODEL_UNAVAILABLE"
    ASR_FAILED = "ASR_FAILED"


_DEFAULT_STATUS: Final[dict[ErrorCode, int]] = {
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.METHOD_NOT_ALLOWED: 405,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.STAGE_NOT_FOUND: 409,
    ErrorCode.INVALID_STATE_TRANSITION: 409,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.WORKER_LOST: 500,
    ErrorCode.WORKER_ERROR: 500,
    ErrorCode.STAGE_FAILED: 500,
    ErrorCode.VIDEO_NOT_FOUND: 404,
    ErrorCode.UPLOAD_NOT_FOUND: 404,
    ErrorCode.UPLOAD_CONFLICT: 409,
    ErrorCode.UPLOAD_INCOMPLETE: 409,
    ErrorCode.UPLOAD_EXPIRED: 410,
    ErrorCode.SOURCE_NOT_FOUND: 404,
    ErrorCode.UNSUPPORTED_MEDIA: 415,
    ErrorCode.PROBE_FAILED: 422,
    ErrorCode.INSUFFICIENT_DISK: 507,
    ErrorCode.FFMPEG_NOT_FOUND: 500,
    ErrorCode.FFPROBE_NOT_FOUND: 500,
    ErrorCode.RENDER_FAILED: 500,
    ErrorCode.RENDER_TIMEOUT: 500,
    ErrorCode.SUBTITLE_FAILED: 500,
    ErrorCode.TRANSCRIPT_NOT_FOUND: 404,
    ErrorCode.TRANSCRIPT_CONFLICT: 409,
    ErrorCode.NO_AUDIO_STREAM: 422,
    ErrorCode.ASR_MODEL_UNAVAILABLE: 503,
    ErrorCode.ASR_FAILED: 500,
}


class AppError(Exception):
    """Expected application failure that maps onto an error envelope."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code if status_code is not None else _DEFAULT_STATUS[code]
        self.details = details


class StageCancelled(Exception):  # noqa: N818 - name follows the pipeline vocabulary
    """Internal control flow: the running stage observed a cancellation request."""


#: ``ERROR_DISK_FULL`` on Windows; CPython usually also sets ``errno.ENOSPC``.
DISK_FULL_WINERROR: Final = 112


def is_disk_full(exc: OSError) -> bool:
    """Return whether ``exc`` means the target volume ran out of space."""
    if exc.errno == errno.ENOSPC:
        return True
    return getattr(exc, "winerror", None) == DISK_FULL_WINERROR


def storage_error(exc: OSError, *, path: str | Path | None = None) -> AppError:
    """Map a filesystem failure onto the canonical application error.

    A full volume becomes ``INSUFFICIENT_DISK`` (507) with a stable ``reason``
    so the client can free space and retry; anything else stays an internal
    error instead of leaking a raw ``OSError``.
    """
    details: dict[str, Any] = {"error": str(exc)}
    if path is not None:
        details["path"] = str(path)
    if is_disk_full(exc):
        details["reason"] = "disk_full"
        return AppError(
            ErrorCode.INSUFFICIENT_DISK,
            "Not enough free disk space to store this file",
            status_code=507,
            details=details,
        )
    return AppError(
        ErrorCode.INTERNAL_ERROR,
        f"Could not write the file: {exc}",
        status_code=500,
        details=details,
    )


def ensure_job_transition(current: JobStatus | str, target: JobStatus | str) -> None:
    """Raise :class:`AppError` when the job transition is not allowed."""
    if not can_transition_job(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid job transition from '{parse_job_status(current)}' "
            f"to '{parse_job_status(target)}'",
            status_code=409,
        )


def ensure_stage_transition(current: StageStatus | str, target: StageStatus | str) -> None:
    """Raise :class:`AppError` when the stage transition is not allowed."""
    if not can_transition_stage(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid stage transition from '{parse_stage_status(current)}' "
            f"to '{parse_stage_status(target)}'",
            status_code=409,
        )


def ensure_video_transition(current: VideoStatus | str, target: VideoStatus | str) -> None:
    """Raise :class:`AppError` when the video transition is not allowed."""
    if not can_transition_video(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid video transition from '{parse_video_status(current)}' "
            f"to '{parse_video_status(target)}'",
            status_code=409,
        )


def _jsonable(value: Any) -> Any:
    try:
        return jsonable_encoder(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def resolve_request_id(request: Request) -> str:
    """Return the request id from the middleware, generating one if absent."""
    state_id = getattr(request.state, "request_id", "")
    if isinstance(state_id, str) and state_id:
        return state_id
    context_id = current_request_id()
    return context_id or new_request_id()


def error_response(
    request: Request,
    *,
    code: str,
    message: str,
    status_code: int,
    details: Any | None = None,
) -> JSONResponse:
    """Build the canonical error envelope."""
    request_id = resolve_request_id(request)
    payload = {
        "error": {
            "code": str(code),
            "message": message,
            "details": _jsonable(details),
            "request_id": request_id,
        }
    }
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={REQUEST_ID_HEADER: request_id},
    )


_HTTP_ERROR_CODES: Final[dict[int, str]] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    406: "NOT_ACCEPTABLE",
    409: "CONFLICT",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Map :class:`AppError` onto its HTTP status."""
    logger.warning(
        "Application error %s on %s %s: %s",
        exc.code,
        request.method,
        request.url.path,
        exc.message,
    )
    return error_response(
        request,
        code=exc.code.value,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Map payload validation failures onto 422 VALIDATION_ERROR."""
    details = [
        {
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]
    return error_response(
        request,
        code=ErrorCode.VALIDATION_ERROR.value,
        message="Request validation failed",
        status_code=422,
        details=details,
    )


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Map Starlette HTTP errors onto named error codes."""
    code = _HTTP_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
    message = str(exc.detail) if exc.detail else "HTTP error"
    return error_response(
        request,
        code=code,
        message=message,
        status_code=exc.status_code,
        details=None,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures without leaking the traceback to clients."""
    logger.exception(
        "Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__
    )
    return error_response(
        request,
        code=ErrorCode.INTERNAL_ERROR.value,
        message="Internal server error",
        status_code=500,
        details=None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every error handler to the application."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        RequestValidationError, validation_error_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        StarletteHTTPException, http_error_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, unhandled_error_handler)
