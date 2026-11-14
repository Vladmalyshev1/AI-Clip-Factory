"""Video endpoints: list, inspect, import, thumbnail."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi import status as http_status
from fastapi.responses import FileResponse

from app.api.deps import DbSessionDep, SessionFactoryDep, SettingsDep
from app.core.status import VideoStatus
from app.schemas.common import Page
from app.schemas.video import VideoImportRequest, VideoRead, video_to_read
from app.services import video_service

router = APIRouter(prefix="/videos", tags=["videos"])

LimitQuery = Annotated[int, Query(ge=1, le=200)]
OffsetQuery = Annotated[int, Query(ge=0)]


@router.get("", response_model=Page[VideoRead])
def list_videos(
    session: DbSessionDep,
    status: VideoStatus | None = None,
    limit: LimitQuery = 50,
    offset: OffsetQuery = 0,
) -> Page[VideoRead]:
    """List ingested videos newest first with optional filters."""
    videos, total = video_service.list_videos(
        session,
        status=status.value if status is not None else None,
        limit=limit,
        offset=offset,
    )
    return Page[VideoRead](
        items=[video_to_read(video) for video in videos],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/import", response_model=VideoRead, status_code=http_status.HTTP_201_CREATED)
def import_video(
    payload: VideoImportRequest, settings: SettingsDep, session_factory: SessionFactoryDep
) -> VideoRead:
    """Ingest an existing local recording, either by copy or in place."""
    video = video_service.import_video(
        settings,
        session_factory,
        path=payload.path,
        copy=payload.copy_source,
        title=payload.title,
    )
    return video_to_read(video)


@router.get("/{video_id}", response_model=VideoRead)
def get_video(video_id: str, session: DbSessionDep) -> VideoRead:
    """Return one ingested video."""
    return video_to_read(video_service.get_video(session, video_id))


@router.get("/{video_id}/thumbnail", response_class=FileResponse)
def get_thumbnail(
    video_id: str, session: DbSessionDep, settings: SettingsDep
) -> FileResponse:
    """Return the cached JPEG thumbnail, generating it on first use."""
    thumbnail = video_service.thumbnail_path(settings, session, video_id)
    return FileResponse(thumbnail, media_type="image/jpeg")
