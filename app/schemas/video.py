"""Request/response schemas for ingested source recordings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.status import VideoStatus
from app.models.video import Video

API_PREFIX = "/api/v1"


class VideoImportRequest(BaseModel):
    """Body of ``POST /videos/import``.

    The wire field is ``copy``; it is exposed as ``copy_source`` because
    ``BaseModel.copy`` already exists and shadowing it warns.
    """

    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(min_length=1)
    copy_source: bool = Field(default=True, alias="copy")
    title: str | None = Field(default=None, max_length=300)


class VideoErrorRead(BaseModel):
    """Persisted ingest error of a video row."""

    code: str
    message: str


class VideoRead(BaseModel):
    """Full video representation returned by the API."""

    id: str
    title: str
    original_filename: str
    source_kind: str
    source_path: str
    size_bytes: int
    sha256: str | None = None
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool | None = None
    container: str | None = None
    bitrate: int | None = None
    status: str
    error: VideoErrorRead | None = None
    created_at_ms: int
    updated_at_ms: int
    thumbnail_url: str | None = None


def video_to_read(video: Video) -> VideoRead:
    """Convert a video row into its API representation."""
    error = None
    if video.error_code or video.error_message:
        error = VideoErrorRead(
            code=video.error_code or "INTERNAL_ERROR",
            message=video.error_message or "",
        )
    thumbnail_url = None
    if video.status == VideoStatus.READY.value:
        thumbnail_url = f"{API_PREFIX}/videos/{video.id}/thumbnail"
    return VideoRead(
        id=video.id,
        title=video.title,
        original_filename=video.original_filename,
        source_kind=video.source_kind,
        source_path=video.source_path,
        size_bytes=video.size_bytes,
        sha256=video.sha256,
        duration_ms=video.duration_ms,
        width=video.width,
        height=video.height,
        fps=video.fps,
        video_codec=video.video_codec,
        audio_codec=video.audio_codec,
        has_audio=None if video.has_audio is None else bool(video.has_audio),
        container=video.container,
        bitrate=video.bitrate,
        status=video.status,
        error=error,
        created_at_ms=video.created_at,
        updated_at_ms=video.updated_at,
        thumbnail_url=thumbnail_url,
    )
