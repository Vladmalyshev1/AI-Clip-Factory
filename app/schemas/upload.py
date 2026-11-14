"""Request/response schemas for resumable uploads."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.upload import Upload


class UploadCreate(BaseModel):
    """Body of ``POST /uploads``."""

    filename: str = Field(min_length=1, max_length=255)
    total_bytes: int = Field(gt=0)


class UploadRead(BaseModel):
    """One upload session; ``received_bytes`` is the resume offset."""

    id: str
    filename: str
    total_bytes: int
    received_bytes: int
    status: str
    chunk_size: int
    sha256: str | None = None
    video_id: str | None = None
    created_at_ms: int
    updated_at_ms: int
    expires_at_ms: int


def upload_to_read(upload: Upload, *, chunk_size: int) -> UploadRead:
    """Convert an upload row into its API representation."""
    return UploadRead(
        id=upload.id,
        filename=upload.filename,
        total_bytes=upload.total_bytes,
        received_bytes=upload.received_bytes,
        status=upload.status,
        chunk_size=chunk_size,
        sha256=upload.sha256,
        video_id=upload.video_id,
        created_at_ms=upload.created_at,
        updated_at_ms=upload.updated_at,
        expires_at_ms=upload.expires_at,
    )
