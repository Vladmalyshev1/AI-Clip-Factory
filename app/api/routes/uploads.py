"""Upload endpoints: create, inspect, stream chunks, complete and cancel."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi import status as http_status

from app.api.deps import DbSessionDep, SessionFactoryDep, SettingsDep
from app.schemas.upload import UploadCreate, UploadRead, upload_to_read
from app.schemas.video import VideoRead, video_to_read
from app.services import upload_service

router = APIRouter(prefix="/uploads", tags=["uploads"])

OffsetQuery = Annotated[
    int, Query(ge=0, description="Bytes already stored; must equal received_bytes.")
]


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


@router.post("", response_model=UploadRead, status_code=http_status.HTTP_201_CREATED)
def create_upload(
    payload: UploadCreate, settings: SettingsDep, session_factory: SessionFactoryDep
) -> UploadRead:
    """Start a resumable upload and preallocate its target file."""
    upload = upload_service.create_upload(
        settings,
        session_factory,
        filename=payload.filename,
        total_bytes=payload.total_bytes,
    )
    return upload_to_read(upload, chunk_size=settings.upload_chunk_size)


@router.get("/{upload_id}", response_model=UploadRead)
def get_upload(upload_id: str, session: DbSessionDep, settings: SettingsDep) -> UploadRead:
    """Return one upload session; ``received_bytes`` is the resume offset."""
    upload = upload_service.get_upload(session, upload_id)
    return upload_to_read(upload, chunk_size=settings.upload_chunk_size)


@router.put("/{upload_id}/chunk", response_model=UploadRead)
async def upload_chunk(
    upload_id: str,
    request: Request,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    offset: OffsetQuery,
) -> UploadRead:
    """Append one raw body chunk at exactly ``received_bytes``."""
    upload = await upload_service.receive_chunk(
        settings,
        session_factory,
        upload_id,
        offset=offset,
        content_length=_content_length(request),
        stream=request.stream(),
    )
    return upload_to_read(upload, chunk_size=settings.upload_chunk_size)


@router.post(
    "/{upload_id}/complete",
    response_model=VideoRead,
    status_code=http_status.HTTP_201_CREATED,
)
def complete_upload(
    upload_id: str,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    response: Response,
) -> VideoRead:
    """Move the received file into ``originals`` and create the video row.

    Idempotent: completing an already completed upload returns the existing
    video with ``200`` instead of ``201``.
    """
    video, created = upload_service.complete_upload(settings, session_factory, upload_id)
    if not created:
        response.status_code = http_status.HTTP_200_OK
    return video_to_read(video)


@router.delete("/{upload_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def cancel_upload(
    upload_id: str, settings: SettingsDep, session_factory: SessionFactoryDep
) -> Response:
    """Cancel an in-flight upload and delete its partial file."""
    upload_service.cancel_upload(settings, session_factory, upload_id)
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)
