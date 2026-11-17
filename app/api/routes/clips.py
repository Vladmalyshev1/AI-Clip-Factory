"""Clip endpoints: generate, list, inspect, retry, cancel, delete and download.

Routes stay thin: they take SQLite's write lock for the check-then-act paths,
delegate to :mod:`app.services.clip_service` and convert rows into API schemas.
The queue owns the work, so generation answers ``202`` immediately; the clip
row (and its ``job_id``) is the progress channel.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response
from fastapi import status as http_status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError

from app.api.deps import DbSessionDep, SettingsDep
from app.core.errors import AppError, ErrorCode
from app.core.paths import safe_name
from app.core.status import ClipStatus
from app.models.clip import Clip
from app.schemas.clip import (
    ClipGenerateRequest,
    ClipGenerateResponse,
    ClipRead,
    ClipStatsRead,
    clip_to_read,
)
from app.schemas.common import Page
from app.services import clip_service, video_service

video_router = APIRouter(prefix="/videos", tags=["clips"])
clip_router = APIRouter(prefix="/clips", tags=["clips"])

LimitQuery = Annotated[int, Query(ge=1, le=200)]
OffsetQuery = Annotated[int, Query(ge=0)]


def _download_name(clip: Clip) -> str:
    """Return a safe download filename (titles may be arbitrary text)."""
    stem = safe_name(clip.title) if clip.title else ""
    return f"{stem or f'clip-{clip.id[:8]}'}.mp4"


@video_router.post(
    "/{video_id}/clips/generate",
    response_model=ClipGenerateResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def generate_clips(
    video_id: str,
    session: DbSessionDep,
    settings: SettingsDep,
    payload: ClipGenerateRequest | None = None,
) -> ClipGenerateResponse:
    """Queue one clip per selected M4 candidate.

    * the video must be ``ready`` and its highlight analysis ``completed``;
    * an empty ``candidate_ids`` requires ``min_score`` (422 otherwise);
    * a candidate that already has a pending/processing/completed clip is
      returned untouched, so repeated requests never start a second render;
    * the unique index on the candidate turns a concurrent duplicate into
      ``CLIP_CONFLICT`` instead of two renders.
    """
    request = payload or ClipGenerateRequest()
    options = request.to_options(settings)
    # Take the write lock before the "what exists / what is active" checks so
    # two concurrent requests cannot both queue the same candidate.
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    try:
        clips = clip_service.create_clips(
            session,
            video_id,
            options=options,
            settings=settings,
            candidate_ids=request.candidate_ids,
            min_score=request.min_score,
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        # Only the "one clip per candidate" index is an expected conflict; any
        # other constraint violation is a bug and must surface as a 500.
        if "highlight_candidate_id" not in str(exc.orig):
            raise
        raise AppError(
            ErrorCode.CLIP_CONFLICT,
            f"A clip of video '{video_id}' was queued concurrently",
            status_code=409,
            details={"video_id": video_id},
        ) from None
    return ClipGenerateResponse(clips=[clip_to_read(clip) for clip in clips])


@video_router.get("/{video_id}/clips", response_model=Page[ClipRead])
def list_clips(
    video_id: str,
    session: DbSessionDep,
    status: ClipStatus | None = None,
    limit: LimitQuery = clip_service.DEFAULT_LIMIT,
    offset: OffsetQuery = 0,
) -> Page[ClipRead]:
    """Page the clips of a video, newest first, optionally filtered by status."""
    video_service.get_video(session, video_id)
    items, total = clip_service.list_clips(
        session,
        video_id,
        status=status.value if status is not None else None,
        offset=offset,
        limit=limit,
    )
    return Page[ClipRead](
        items=[clip_to_read(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@video_router.get("/{video_id}/clips/stats", response_model=ClipStatsRead)
def clip_stats(video_id: str, session: DbSessionDep) -> ClipStatsRead:
    """Return per-status clip counters of a video."""
    video_service.get_video(session, video_id)
    stats = clip_service.clip_stats(session, video_id)
    return ClipStatsRead(
        total=stats.total,
        pending=stats.pending,
        processing=stats.processing,
        completed=stats.completed,
        failed=stats.failed,
        canceled=stats.canceled,
    )


@clip_router.get("/{clip_id}", response_model=ClipRead)
def get_clip(clip_id: str, session: DbSessionDep) -> ClipRead:
    """Return one clip."""
    return clip_to_read(clip_service.get_clip(session, clip_id))


@clip_router.post(
    "/{clip_id}/retry",
    response_model=ClipRead,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def retry_clip(clip_id: str, session: DbSessionDep) -> ClipRead:
    """Re-queue a failed or canceled clip (409 for any other state)."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    clip = clip_service.get_clip(session, clip_id)
    clip_service.retry_clip(session, clip)
    session.commit()
    return clip_to_read(clip)


@clip_router.post("/{clip_id}/cancel", response_model=ClipRead)
def cancel_clip(clip_id: str, session: DbSessionDep) -> ClipRead:
    """Stop a pending or processing clip; a running FFmpeg child is terminated."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    clip = clip_service.get_clip(session, clip_id)
    clip_service.cancel_clip(session, clip)
    session.commit()
    return clip_to_read(clip)


@clip_router.delete(
    "/{clip_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_clip(clip_id: str, session: DbSessionDep, settings: SettingsDep) -> Response:
    """Delete a clip row and its published file (409 while processing)."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    clip = clip_service.get_clip(session, clip_id)
    clip_service.delete_clip(session, clip, settings=settings)
    session.commit()
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@clip_router.get("/{clip_id}/file")
def download_clip(clip_id: str, session: DbSessionDep, settings: SettingsDep) -> FileResponse:
    """Serve the published file of one clip (containment-checked)."""
    clip = clip_service.get_clip(session, clip_id)
    path = clip_service.resolve_clip_file(settings, clip)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=_download_name(clip),
    )
