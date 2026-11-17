"""Clip exports (M7): an export is a new clip row cut from an existing one.

The export reuses the whole M6 machinery — the ``clips`` table (``kind='export'``
+ ``parent_clip_id``), the queue (``Job(type='clip')``) and the render stage —
so cancellation, retry, worker-loss recovery and file endpoints behave exactly
like generated clips. The source clip and its published file are never modified.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.status import ClipKind, ClipStatus, JobStatus, parse_clip_status
from app.models.clip import Clip
from app.models.job import Job
from app.models.video import Video
from app.services import clip_service, job_service
from app.services.clip_service import CLIP_JOB_TYPE, ClipOptions, _active_job
from app.services.editor_service import EditorDraft, get_draft
from app.services.job_service import dump_json

logger = logging.getLogger(__name__)


def list_exports(
    session: Session, clip_id: str, *, offset: int = 0, limit: int = 20
) -> tuple[list[Clip], int]:
    """Return the exports of one source clip, newest first."""
    conditions = [Clip.parent_clip_id == clip_id, Clip.kind == ClipKind.EXPORT.value]
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


def find_active_export(session: Session, clip_id: str) -> Clip | None:
    """Return the pending/processing export of a clip, if any."""
    rows = session.scalars(
        select(Clip).where(
            Clip.parent_clip_id == clip_id,
            Clip.kind == ClipKind.EXPORT.value,
            Clip.status.in_(
                (ClipStatus.PENDING.value, ClipStatus.PROCESSING.value)
            ),
        )
    ).all()
    for row in rows:
        if _active_job(session, row) is not None:
            return row
    return None


def _export_options(clip: Clip, draft: EditorDraft, settings: Settings) -> ClipOptions:
    """Build the render options of one export from the clip and the draft."""
    base = ClipOptions.from_settings(settings)
    width, height = base.width, base.height
    if draft.output_format == clip.output_format and clip.width and clip.height:
        # Re-rendering in the clip's own format keeps its original canvas.
        width, height = clip.width, clip.height
    return ClipOptions(
        pre_padding_ms=0,
        post_padding_ms=0,
        min_duration_ms=base.min_duration_ms,
        max_duration_ms=base.max_duration_ms,
        duration_mode="candidate",
        target_duration_ms=None,
        output_format=draft.output_format,
        crop_mode=draft.crop_mode,
        encoding_mode=draft.encoding_mode,
        width=width,
        height=height,
    )


def _require_source_clip(session: Session, clip_id: str) -> Clip:
    """Load the clip an export is cut from, rejecting exports of exports."""
    clip = clip_service.get_clip(session, clip_id)
    if clip.kind != ClipKind.AUTO.value:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Clip '{clip_id}' is an export; exports can only be cut from "
            "generated clips",
            status_code=409,
            details={"clip_id": clip_id, "kind": clip.kind},
        )
    return clip


def editor_service_validate(
    source: Clip, draft: EditorDraft, video_duration_ms: int | None
) -> None:
    """Re-export of the editor validation so callers cannot skip it."""
    from app.services.editor_service import validate_draft

    validate_draft(draft, source, video_duration_ms)


def create_export(
    session: Session,
    clip_id: str,
    *,
    settings: Settings,
    draft: EditorDraft | None = None,
    now_ms: int | None = None,
) -> tuple[Clip, bool]:
    """Create (or return the in-flight) export of one clip.

    Returns ``(export, created)``. Callers must hold SQLite's write lock
    (``BEGIN IMMEDIATE``) around the check-then-act sequence.
    """
    now = utc_now_ms() if now_ms is None else now_ms
    source = _require_source_clip(session, clip_id)
    resolved = draft if draft is not None else get_draft(session, clip_id)

    # Same validation as PUT /editor: an invalid draft must become a 422 here,
    # not a permanently failed export job.
    video = session.get(Video, source.video_id)
    editor_service_validate(source, resolved, video.duration_ms if video else None)

    in_flight = find_active_export(session, clip_id)
    if in_flight is not None:
        # Idempotency: an active export is returned untouched; a completed one
        # stays downloadable. A new settings version creates a new export.
        return in_flight, False

    options = _export_options(source, resolved, settings)
    duration = resolved.end_ms - resolved.start_ms
    if not source.video_id:  # pragma: no cover - NOT NULL column
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            f"Clip '{clip_id}' has no video id",
            status_code=500,
        )
    active_for_video = session.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            Job.video_id == source.video_id,
            Job.type == CLIP_JOB_TYPE,
            Job.status.in_(
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
            ),
        )
    )
    if int(active_for_video or 0) + 1 > settings.clip_max_concurrent_jobs:
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Video '{source.video_id}' already has too many active clip jobs",
            status_code=409,
            details={
                "video_id": source.video_id,
                "max_concurrent_jobs": settings.clip_max_concurrent_jobs,
            },
        )

    export = Clip(
        video_id=source.video_id,
        highlight_id=source.highlight_id,
        highlight_candidate_id=None,
        parent_clip_id=source.id,
        kind=ClipKind.EXPORT.value,
        status=ClipStatus.PENDING.value,
        title=source.title,
        start_ms=resolved.start_ms,
        end_ms=resolved.end_ms,
        duration_ms=duration,
        score=source.score,
        options_json=dump_json(options.to_dict()),
        encoding_mode=str(options.encoding_mode),
        output_format=str(options.output_format),
        crop_mode=str(options.crop_mode),
        created_at=now,
        updated_at=now,
    )
    if resolved.subtitles.enabled:
        # Snapshot the cues so the render never re-reads (possibly later-edited)
        # draft state; the export carries exactly what the user approved.
        export.edit_json = dump_json({"subtitles": resolved.subtitles.to_dict()})
    session.add(export)
    session.flush()

    job = job_service.enqueue(
        session,
        type=CLIP_JOB_TYPE,
        video_id=source.video_id,
        params={"clip_id": export.id},
    )
    export.job_id = job.id
    session.flush()
    logger.info(
        "Queued export %s of clip %s (%d..%d ms, subtitles=%s)",
        export.id,
        source.id,
        export.start_ms,
        export.end_ms,
        resolved.subtitles.enabled,
    )
    return export, True


def retry_export(session: Session, export_id: str) -> Clip:
    """Re-queue a failed or canceled export (the M6 rule, exports only)."""
    export = get_export(session, export_id)
    return clip_service.retry_clip(session, export)


def cancel_export(session: Session, export_id: str) -> Clip:
    """Cancel a pending or processing export (terminates a running FFmpeg)."""
    export = get_export(session, export_id)
    return clip_service.cancel_clip(session, export)


def get_export(session: Session, export_id: str) -> Clip:
    """Load one export row or raise ``CLIP_NOT_FOUND``."""
    export = session.get(Clip, export_id)
    if export is None or export.kind != ClipKind.EXPORT.value:
        raise AppError(
            ErrorCode.CLIP_NOT_FOUND,
            f"Export '{export_id}' not found",
            status_code=404,
            details={"export_id": export_id},
        )
    return export


def delete_export(session: Session, export_id: str, *, settings: Settings) -> None:
    """Delete an export row and its file; the source clip is never touched."""
    export = get_export(session, export_id)
    current = parse_clip_status(export.status)
    if current in (ClipStatus.PENDING, ClipStatus.PROCESSING):
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"Export '{export_id}' is still {current.value}; cancel it first",
            status_code=409,
            details={"export_id": export_id, "status": current.value},
        )
    clip_service.delete_clip(session, export, settings=settings)


def count_exports(session: Session, clip_id: str) -> int:
    """Number of exports ever created for one clip."""
    return int(
        session.scalar(
            select(func.count())
            .select_from(Clip)
            .where(
                Clip.parent_clip_id == clip_id,
                Clip.kind == ClipKind.EXPORT.value,
            )
        )
        or 0
    )


__all__ = [
    "cancel_export",
    "count_exports",
    "create_export",
    "delete_export",
    "find_active_export",
    "get_export",
    "list_exports",
    "retry_export",
]
