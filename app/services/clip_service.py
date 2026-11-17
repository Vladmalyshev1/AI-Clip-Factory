"""Clip generation service (M6): persistence, queueing and file bookkeeping.

Every helper performs one short unit of work and flushes instead of committing,
so the caller (API route or worker stage) owns the transaction boundary. A clip
row is created *before* its job is queued, which makes ``candidate → clip``
idempotent: the unique index on ``highlight_candidate_id`` turns a concurrent
duplicate into a conflict instead of two renders of the same moment.

Render parameters are snapshotted on the row (``options_json``), so the worker
stage needs nothing from the request except the clip id. ``start_ms``/``end_ms``
are resolved once, at creation time, and never recomputed while rendering.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.clips import ClipBounds, ClipBoundsConfig, ClipBoundsError, calculate_bounds
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.status import (
    ClipStatus,
    HighlightStatus,
    VideoStatus,
    can_transition_clip,
    parse_clip_status,
)
from app.media.cut import CropMode, EncodingMode, OutputFormat
from app.media.graph import VerticalLayout
from app.media.validate import DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_TOLERANCE
from app.models.clip import Clip
from app.models.highlight import HighlightCandidate
from app.models.job import Job
from app.models.video import Video
from app.services import highlight_service, job_service
from app.services.transcript_service import ACTIVE_JOB_STATUSES, MAX_ERROR_CHARS

logger = logging.getLogger(__name__)

#: ``Job.type`` of a job that renders one clip.
CLIP_JOB_TYPE: Final = "clip"
#: Default number of clips returned by :func:`list_clips`.
DEFAULT_LIMIT: Final = 50
#: Upper bound of candidates read in one bulk generate request.
MAX_BULK_CANDIDATES: Final = 500


def _as_int(value: object, default: int) -> int:
    """Coerce a JSON snapshot value to ``int`` (missing keeps the default)."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Expected an integer, got a boolean ({value})")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected an integer, got {type(value).__name__}")


def _as_str(value: object, default: str) -> str:
    """Coerce a JSON snapshot value to ``str`` (missing keeps the default)."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected a string, got {type(value).__name__}")


@dataclass(frozen=True)
class ClipOptions:
    """Render parameters captured on a clip row."""

    pre_padding_ms: int = 5000
    post_padding_ms: int = 10_000
    min_duration_ms: int = 5000
    max_duration_ms: int = 120_000
    duration_mode: str = "candidate"
    target_duration_ms: int | None = None
    output_format: str = OutputFormat.VERTICAL.value
    crop_mode: str = CropMode.PAD.value
    encoding_mode: str = EncodingMode.ACCURATE.value
    width: int = 1080
    height: int = 1920

    def __post_init__(self) -> None:
        # Reuse the media-layer validation: the stage must never receive a mode
        # combination ``build_clip_command`` would reject. Rejecting it here
        # turns a doomed render into a 422 before a row or a job exists.
        OutputFormat(self.output_format)
        CropMode(self.crop_mode)
        EncodingMode(self.encoding_mode)
        VerticalLayout(width=self.width, height=self.height)
        if (
            self.encoding_mode == EncodingMode.FAST.value
            and self.output_format != OutputFormat.ORIGINAL.value
        ):
            raise ValueError("fast encoding requires the original output format")
        if not (16 <= self.width <= 4096 and 16 <= self.height <= 4096):
            raise ValueError(
                f"canvas {self.width}x{self.height} is outside the 16..4096 range"
            )
        if (
            self.output_format == OutputFormat.VERTICAL.value
            and abs(self.width / self.height - DEFAULT_ASPECT_RATIO)
            > DEFAULT_ASPECT_TOLERANCE
        ):
            raise ValueError(
                f"vertical canvas {self.width}x{self.height} must keep the 9:16 aspect ratio"
            )
        self.bounds_config()

    def bounds_config(self) -> ClipBoundsConfig:
        """Return the duration limits used to resolve clip boundaries."""
        return ClipBoundsConfig(
            pre_padding_ms=self.pre_padding_ms,
            post_padding_ms=self.post_padding_ms,
            min_duration_ms=self.min_duration_ms,
            max_duration_ms=self.max_duration_ms,
            duration_mode=self.duration_mode,
            target_duration_ms=self.target_duration_ms,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise the snapshot stored in ``clips.options_json``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ClipOptions:
        """Rebuild the snapshot stored in ``options_json``.

        Missing keys fall back to the current defaults, so a clip written by an
        older version stays renderable after an upgrade.
        """
        target = data.get("target_duration_ms")
        return cls(
            pre_padding_ms=_as_int(data.get("pre_padding_ms"), cls.pre_padding_ms),
            post_padding_ms=_as_int(data.get("post_padding_ms"), cls.post_padding_ms),
            min_duration_ms=_as_int(data.get("min_duration_ms"), cls.min_duration_ms),
            max_duration_ms=_as_int(data.get("max_duration_ms"), cls.max_duration_ms),
            duration_mode=_as_str(data.get("duration_mode"), cls.duration_mode),
            target_duration_ms=_as_int(target, 0) if target is not None else None,
            output_format=_as_str(data.get("output_format"), cls.output_format),
            crop_mode=_as_str(data.get("crop_mode"), cls.crop_mode),
            encoding_mode=_as_str(data.get("encoding_mode"), cls.encoding_mode),
            width=_as_int(data.get("width"), cls.width),
            height=_as_int(data.get("height"), cls.height),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> ClipOptions:
        """Build the default options from application settings."""
        return cls(
            pre_padding_ms=settings.clip_pre_padding_ms,
            post_padding_ms=settings.clip_post_padding_ms,
            min_duration_ms=settings.clip_min_duration_ms,
            max_duration_ms=settings.clip_max_duration_ms,
            output_format=settings.clip_default_output_format,
            crop_mode=settings.clip_default_crop_mode,
            encoding_mode=settings.clip_default_encoding_mode,
            width=settings.clip_vertical_width,
            height=settings.clip_vertical_height,
        )


@dataclass(frozen=True)
class ClipStats:
    """Aggregated clip counters of one video."""

    total: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    canceled: int = 0


def _now(now_ms: int | None) -> int:
    """Return ``now_ms`` or the current epoch milliseconds."""
    return now_ms if now_ms is not None else utc_now_ms()


def _truncate(message: str) -> str:
    """Clamp an error message to :data:`MAX_ERROR_CHARS` characters."""
    return message[:MAX_ERROR_CHARS]


def _ensure_transition(current: ClipStatus | str, target: ClipStatus | str) -> None:
    """Raise ``INVALID_STATE_TRANSITION`` (409) when the move is not allowed."""
    if not can_transition_clip(current, target):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Invalid clip transition from '{parse_clip_status(current)}' "
            f"to '{parse_clip_status(target)}'",
            status_code=409,
        )


def find_clip(session: Session, clip_id: str) -> Clip | None:
    """Return a clip row by id, or ``None``."""
    return session.get(Clip, clip_id)


def get_clip(session: Session, clip_id: str) -> Clip:
    """Load a clip or raise ``CLIP_NOT_FOUND`` (404)."""
    clip = find_clip(session, clip_id)
    if clip is None:
        raise AppError(
            ErrorCode.CLIP_NOT_FOUND,
            f"Clip '{clip_id}' not found",
            status_code=404,
            details={"clip_id": clip_id},
        )
    return clip


def list_clips(
    session: Session,
    video_id: str,
    *,
    status: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Clip], int]:
    """Return a page of clips of a video, newest first."""
    conditions = [Clip.video_id == video_id]
    if status is not None:
        conditions.append(Clip.status == status)
    total = session.scalar(select(func.count()).select_from(Clip).where(*conditions))
    items = list(
        session.scalars(
            select(Clip)
            .where(*conditions)
            .order_by(Clip.created_at.desc(), Clip.id.desc())
            .limit(max(1, limit))
            .offset(max(0, offset))
        ).all()
    )
    return items, int(total or 0)


def clip_stats(session: Session, video_id: str) -> ClipStats:
    """Count the clips of a video per status."""
    rows = session.execute(
        select(Clip.status, func.count(Clip.id))
        .where(Clip.video_id == video_id)
        .group_by(Clip.status)
    ).all()
    counts = {str(status): int(count) for status, count in rows}
    return ClipStats(
        total=sum(counts.values()),
        pending=counts.get(ClipStatus.PENDING.value, 0),
        processing=counts.get(ClipStatus.PROCESSING.value, 0),
        completed=counts.get(ClipStatus.COMPLETED.value, 0),
        failed=counts.get(ClipStatus.FAILED.value, 0),
        canceled=counts.get(ClipStatus.CANCELED.value, 0),
    )


def count_active_jobs(session: Session, video_id: str) -> int:
    """Return how many clip jobs of a video are queued or running."""
    total = session.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            Job.video_id == video_id,
            Job.type == CLIP_JOB_TYPE,
            Job.status.in_(tuple(status.value for status in ACTIVE_JOB_STATUSES)),
        )
    )
    return int(total or 0)


def _require_source(session: Session, video_id: str) -> Video:
    """Load a ready source recording or raise a domain error."""
    video = session.get(Video, video_id)
    if video is None:
        raise AppError(
            ErrorCode.VIDEO_NOT_FOUND,
            f"Video '{video_id}' not found",
            status_code=404,
            details={"video_id": video_id},
        )
    if video.status != VideoStatus.READY.value or not video.duration_ms:
        raise AppError(
            ErrorCode.CLIP_SOURCE_UNAVAILABLE,
            f"Video '{video_id}' is not ready for clip generation "
            f"(status: {video.status})",
            status_code=409,
            details={"video_id": video_id, "status": video.status},
        )
    return video


def _select_candidates(
    session: Session,
    video_id: str,
    *,
    candidate_ids: Sequence[str],
    min_score: float | None,
) -> list[HighlightCandidate]:
    """Resolve the candidates of one generate request.

    An explicit id list is authoritative. An empty list only runs when
    ``min_score`` is set, so a malformed request can never render every moment
    of a recording by accident.
    """
    analysis = highlight_service.find_analysis(session, video_id)
    if analysis is None:
        raise AppError(
            ErrorCode.HIGHLIGHT_NOT_FOUND,
            f"Video '{video_id}' has no highlight analysis",
            status_code=404,
            details={"video_id": video_id},
        )
    if analysis.status != HighlightStatus.COMPLETED.value:
        raise AppError(
            ErrorCode.HIGHLIGHT_INCOMPLETE,
            f"Video '{video_id}' has no completed highlight analysis "
            f"(status: {analysis.status})",
            status_code=409,
            details={"video_id": video_id, "highlight_status": analysis.status},
        )

    if candidate_ids:
        seen: set[str] = set()
        candidates: list[HighlightCandidate] = []
        for candidate_id in candidate_ids:
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(
                highlight_service.get_candidate(session, video_id, candidate_id)
            )
        return candidates

    if min_score is None:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "candidate_ids is empty; pass min_score to select candidates explicitly",
            status_code=422,
            details={"video_id": video_id},
        )
    items, _ = highlight_service.list_candidates(
        session, analysis.id, limit=MAX_BULK_CANDIDATES
    )
    selected = [candidate for candidate in items if candidate.score >= min_score]
    if not selected:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"No highlight candidate of video '{video_id}' scores at least {min_score}",
            status_code=422,
            details={"video_id": video_id, "min_score": min_score},
        )
    return selected


def create_clips(
    session: Session,
    video_id: str,
    *,
    options: ClipOptions,
    settings: Settings,
    candidate_ids: Sequence[str] = (),
    min_score: float | None = None,
    now_ms: int | None = None,
) -> list[Clip]:
    """Create (or re-arm) one clip per selected candidate and queue its job.

    Callers must hold SQLite's write lock (``BEGIN IMMEDIATE``) around the whole
    check-then-act sequence; the unique index on the candidate is the final
    guard against a concurrent duplicate.
    """
    now = _now(now_ms)
    video = _require_source(session, video_id)
    candidates = _select_candidates(
        session, video_id, candidate_ids=candidate_ids, min_score=min_score
    )

    config = options.bounds_config()
    planned: list[tuple[HighlightCandidate, ClipBounds, Clip | None, bool]] = []
    to_queue = 0
    for candidate in candidates:
        try:
            bounds = calculate_bounds(
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                video_duration_ms=int(video.duration_ms or 0),
                config=config,
            )
        except ClipBoundsError as exc:
            raise AppError(
                ErrorCode.CLIP_BOUNDS_INVALID,
                str(exc),
                status_code=422,
                details={
                    "video_id": video_id,
                    "candidate_id": candidate.id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                },
            ) from exc

        clip = session.scalars(
            select(Clip).where(Clip.highlight_candidate_id == candidate.id)
        ).first()
        # A pending/processing/completed clip is never re-armed. Neither is a
        # failed or canceled one whose job is still winding down: a cancel only
        # *requests* termination, so its renderer may still be writing.
        in_flight = clip is not None and (
            parse_clip_status(clip.status)
            in (ClipStatus.PENDING, ClipStatus.PROCESSING, ClipStatus.COMPLETED)
            or _active_job(session, clip) is not None
        )
        if not in_flight:
            to_queue += 1
        planned.append((candidate, bounds, clip, in_flight))

    if count_active_jobs(session, video_id) + to_queue > settings.clip_max_concurrent_jobs:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Video '{video_id}' already has too many active clip jobs",
            status_code=409,
            details={
                "video_id": video_id,
                "max_concurrent_jobs": settings.clip_max_concurrent_jobs,
            },
        )

    clips: list[Clip] = []
    for candidate, bounds, clip, in_flight in planned:
        if in_flight and clip is not None:
            # Already in flight or done: never start a second render of it.
            clips.append(clip)
            continue

        if clip is None:
            clip = Clip(
                video_id=video_id,
                highlight_id=candidate.highlight_id,
                highlight_candidate_id=candidate.id,
                status=ClipStatus.PENDING.value,
                start_ms=bounds.start_ms,
                end_ms=bounds.end_ms,
                duration_ms=bounds.duration_ms,
                created_at=now,
                updated_at=now,
            )
            session.add(clip)
            session.flush()
        else:
            _ensure_transition(clip.status, ClipStatus.PENDING)

        clip.start_ms = bounds.start_ms
        clip.end_ms = bounds.end_ms
        clip.duration_ms = bounds.duration_ms
        clip.score = candidate.score
        clip.status = ClipStatus.PENDING.value
        clip.options_json = job_service.dump_json(options.to_dict())
        clip.encoding_mode = str(options.encoding_mode)
        clip.output_format = str(options.output_format)
        clip.crop_mode = str(options.crop_mode)
        clip.error_code = None
        clip.error_message = None
        clip.started_at = None
        clip.finished_at = None
        clip.updated_at = now

        job = job_service.enqueue(
            session,
            type=CLIP_JOB_TYPE,
            video_id=video_id,
            params={"clip_id": clip.id},
        )
        clip.job_id = job.id
        session.flush()
        clips.append(clip)

    logger.info(
        "Queued %d clip(s) for video %s (%d candidate(s) requested)",
        len(clips),
        video_id,
        len(candidates),
    )
    return clips


def _active_job(session: Session, clip: Clip) -> Job | None:
    """Return the clip's job when it is still queued or running."""
    if not clip.job_id:
        return None
    job = session.get(Job, clip.job_id)
    if job is None or job.status not in tuple(status.value for status in ACTIVE_JOB_STATUSES):
        return None
    return job


def retry_clip(session: Session, clip: Clip, *, now_ms: int | None = None) -> Clip:
    """Re-queue a failed or canceled clip (``CLIP_CONFLICT`` otherwise)."""
    now = _now(now_ms)
    current = parse_clip_status(clip.status)
    if current not in (ClipStatus.FAILED, ClipStatus.CANCELED):
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Clip '{clip.id}' is '{current.value}'; only failed or canceled clips "
            "can be retried",
            status_code=409,
            details={"clip_id": clip.id, "status": current.value},
        )
    active = _active_job(session, clip)
    if active is not None:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Clip '{clip.id}' is still attached to an active job",
            status_code=409,
            details={"clip_id": clip.id, "job_id": active.id},
        )

    _ensure_transition(current, ClipStatus.PENDING)
    clip.status = ClipStatus.PENDING.value
    clip.error_code = None
    clip.error_message = None
    clip.started_at = None
    clip.finished_at = None
    clip.updated_at = now

    job = job_service.enqueue(
        session,
        type=CLIP_JOB_TYPE,
        video_id=clip.video_id,
        params={"clip_id": clip.id},
    )
    clip.job_id = job.id
    session.flush()
    return clip


def cancel_clip(session: Session, clip: Clip, *, now_ms: int | None = None) -> Clip:
    """Cancel a pending or processing clip and its render job."""
    now = _now(now_ms)
    current = parse_clip_status(clip.status)
    if current is ClipStatus.COMPLETED:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Clip '{clip.id}' is already completed",
            status_code=409,
            details={"clip_id": clip.id},
        )
    if current is ClipStatus.CANCELED:
        return clip

    job = _active_job(session, clip)
    if job is not None:
        job_service.request_cancel(session, job.id)
        session.flush()

    _ensure_transition(current, ClipStatus.CANCELED)
    clip.status = ClipStatus.CANCELED.value
    clip.finished_at = now
    clip.updated_at = now
    session.flush()
    return clip


def _contained_path(settings: Settings, output_path: str | Path) -> Path:
    """Return ``output_path`` resolved under ``clips_dir`` or raise ``CLIP_FAILED``."""
    root = settings.clips_dir.resolve()
    resolved = Path(output_path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise AppError(
            ErrorCode.CLIP_FAILED,
            "Clip file is outside the configured clip directory",
            status_code=500,
            details={"path": str(resolved), "root": str(root)},
        )
    return resolved


def resolve_clip_file(settings: Settings, clip: Clip) -> Path:
    """Return the published file of a completed clip (404 when missing)."""
    if not clip.output_path:
        raise AppError(
            ErrorCode.CLIP_NOT_FOUND,
            f"Clip '{clip.id}' has no rendered file yet",
            status_code=404,
            details={"clip_id": clip.id, "status": clip.status},
        )
    path = _contained_path(settings, clip.output_path)
    if not path.is_file():
        raise AppError(
            ErrorCode.CLIP_NOT_FOUND,
            f"Clip '{clip.id}' file is missing",
            status_code=404,
            details={"clip_id": clip.id, "path": str(path)},
        )
    return path


def delete_clip(session: Session, clip: Clip, *, settings: Settings) -> None:
    """Delete a clip row and its published file (processing clips are refused)."""
    if parse_clip_status(clip.status) is ClipStatus.PROCESSING:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Clip '{clip.id}' is processing; cancel it before deleting",
            status_code=409,
            details={"clip_id": clip.id},
        )
    job = _active_job(session, clip)
    if job is not None:
        job_service.request_cancel(session, job.id)

    if clip.output_path:
        try:
            _contained_path(settings, clip.output_path).unlink(missing_ok=True)
        except (AppError, OSError):
            # A leaked absolute path must never delete anything outside clips_dir;
            # a locked file is logged and leaves the row deletable.
            logger.warning("Could not delete clip file %s", clip.output_path, exc_info=True)
    session.delete(clip)
    session.flush()


# --------------------------------------------------------------------- stage API


def start_processing(
    session: Session, clip_id: str, *, job_id: str, now_ms: int | None = None
) -> Clip:
    """Move a clip into ``processing`` (idempotent for an interrupted run)."""
    now = _now(now_ms)
    clip = get_clip(session, clip_id)
    current = parse_clip_status(clip.status)
    if current is ClipStatus.COMPLETED:
        return clip
    if current in (ClipStatus.FAILED, ClipStatus.CANCELED):
        _ensure_transition(current, ClipStatus.PENDING)
        current = ClipStatus.PENDING
    _ensure_transition(current, ClipStatus.PROCESSING)

    clip.status = ClipStatus.PROCESSING.value
    clip.job_id = job_id
    clip.started_at = clip.started_at or now
    clip.finished_at = None
    clip.error_code = None
    clip.error_message = None
    clip.updated_at = now
    session.flush()
    return clip


def complete_clip(
    session: Session,
    clip_id: str,
    *,
    output_path: Path,
    file_size: int,
    width: int,
    height: int,
    fps: float,
    now_ms: int | None = None,
) -> Clip:
    """Publish the render result on the clip row."""
    now = _now(now_ms)
    clip = get_clip(session, clip_id)
    current = parse_clip_status(clip.status)
    if current is ClipStatus.COMPLETED:
        return clip
    _ensure_transition(current, ClipStatus.COMPLETED)

    clip.status = ClipStatus.COMPLETED.value
    clip.output_path = str(output_path)
    clip.file_size = file_size
    clip.width = width
    clip.height = height
    clip.fps = fps
    clip.aspect_ratio = f"{width}:{height}"
    clip.error_code = None
    clip.error_message = None
    clip.finished_at = now
    clip.updated_at = now
    session.flush()
    return clip


def fail_clip(
    session: Session, clip_id: str, *, error_code: str, error_message: str,
    now_ms: int | None = None,
) -> Clip:
    """Mark a clip failed, keeping the (truncated) reason for the UI."""
    now = _now(now_ms)
    clip = get_clip(session, clip_id)
    current = parse_clip_status(clip.status)
    if current is ClipStatus.COMPLETED:
        return clip
    if current is not ClipStatus.FAILED:
        _ensure_transition(current, ClipStatus.FAILED)
    clip.status = ClipStatus.FAILED.value
    clip.error_code = error_code
    clip.error_message = _truncate(error_message)
    clip.finished_at = now
    clip.updated_at = now
    session.flush()
    return clip


def reconcile_clip_job(
    session: Session,
    job_id: str,
    *,
    canceled: bool,
    error_code: str = ErrorCode.WORKER_LOST.value,
    error_message: str = "",
    now_ms: int | None = None,
) -> Clip | None:
    """Align a clip row with a job that ended outside the render stage.

    The stage records its own outcome; this covers the paths that never reach
    it (a lost worker, a job canceled through the job API). ``job_id`` is not
    indexed - recovery is rare and the table stays small.
    """
    clip = session.scalars(select(Clip).where(Clip.job_id == job_id)).first()
    if clip is None:
        return None
    now = _now(now_ms)
    current = parse_clip_status(clip.status)
    if current not in (ClipStatus.PENDING, ClipStatus.PROCESSING):
        # Completed, canceled and failed rows already tell the truth.
        return clip
    if canceled:
        clip.status = ClipStatus.CANCELED.value
    else:
        clip.status = ClipStatus.FAILED.value
        clip.error_code = error_code
        clip.error_message = _truncate(error_message)
    clip.finished_at = now
    clip.updated_at = now
    session.flush()
    return clip


def interrupt_clip(
    session: Session, clip_id: str, *, now_ms: int | None = None
) -> Clip:
    """Return an interrupted clip to ``pending`` (graceful worker shutdown)."""
    now = _now(now_ms)
    clip = get_clip(session, clip_id)
    current = parse_clip_status(clip.status)
    if current in (ClipStatus.COMPLETED, ClipStatus.CANCELED):
        return clip
    if current is not ClipStatus.PENDING:
        _ensure_transition(current, ClipStatus.PENDING)
    clip.status = ClipStatus.PENDING.value
    clip.started_at = None
    clip.updated_at = now
    session.flush()
    return clip


__all__ = [
    "CLIP_JOB_TYPE",
    "DEFAULT_LIMIT",
    "MAX_BULK_CANDIDATES",
    "ClipOptions",
    "ClipStats",
    "cancel_clip",
    "clip_stats",
    "complete_clip",
    "count_active_jobs",
    "create_clips",
    "delete_clip",
    "fail_clip",
    "find_clip",
    "get_clip",
    "interrupt_clip",
    "list_clips",
    "reconcile_clip_job",
    "resolve_clip_file",
    "retry_clip",
    "start_processing",
]
