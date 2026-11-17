"""Request/response schemas for generated clips (M6).

The generate request mirrors the clip settings but every field is optional: an
omitted value comes from ``ACF_CLIP_*``. The contract is deliberately strict
about ambiguity — an empty ``candidate_ids`` requires an explicit ``min_score``,
so a malformed request can never render every moment of a recording.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.models.clip import Clip
from app.services.clip_service import ClipOptions

MAX_DURATION_MS = 600_000
RESOLUTION_PATTERN = re.compile(r"^(\d{2,5})x(\d{2,5})$")


class ClipErrorRead(BaseModel):
    """Persisted render error of a clip."""

    code: str
    message: str


class ClipGenerateRequest(BaseModel):
    """Body of ``POST /videos/{id}/clips/generate``."""

    candidate_ids: list[str] = Field(default_factory=list, max_length=500)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    pre_padding_ms: int | None = Field(default=None, ge=0, le=MAX_DURATION_MS)
    post_padding_ms: int | None = Field(default=None, ge=0, le=MAX_DURATION_MS)
    duration_mode: Literal["candidate", "fixed"] = "candidate"
    target_duration_ms: int | None = Field(default=None, ge=1000, le=MAX_DURATION_MS)
    min_duration_ms: int | None = Field(default=None, ge=1000, le=MAX_DURATION_MS)
    max_duration_ms: int | None = Field(default=None, ge=1000, le=MAX_DURATION_MS)
    output_format: Literal["original", "vertical"] | None = None
    crop_mode: Literal["crop", "pad"] | None = None
    encoding_mode: Literal["accurate", "fast"] | None = None
    resolution: str | None = None

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: str | None) -> str | None:
        """Accept only ``WIDTHxHEIGHT`` of even, non-zero dimensions."""
        if value is None:
            return None
        match = RESOLUTION_PATTERN.match(value.strip().lower())
        if match is None:
            raise ValueError("resolution must look like '1080x1920'")
        width, height = int(match.group(1)), int(match.group(2))
        if width % 2 or height % 2:
            raise ValueError("resolution dimensions must be even for yuv420p")
        return f"{width}x{height}"

    @model_validator(mode="after")
    def _validate_duration_mode(self) -> ClipGenerateRequest:
        """Refuse a target duration that no mode would ever use."""
        if self.target_duration_ms is not None and self.duration_mode != "fixed":
            raise ValueError("target_duration_ms requires duration_mode='fixed'")
        return self

    def to_options(self, settings: Settings) -> ClipOptions:
        """Merge the request over the configured defaults."""
        try:
            return self._merged(settings)
        except ValueError as exc:
            raise AppError(
                ErrorCode.CLIP_BOUNDS_INVALID, str(exc), status_code=422
            ) from exc

    def _merged(self, settings: Settings) -> ClipOptions:
        """Build the snapshot; ``ValueError`` becomes a 422 in the caller."""
        base = ClipOptions.from_settings(settings)
        width, height = base.width, base.height
        if self.resolution is not None:
            width, height = (int(part) for part in self.resolution.split("x"))
        values = ClipOptions(
            pre_padding_ms=(
                self.pre_padding_ms if self.pre_padding_ms is not None else base.pre_padding_ms
            ),
            post_padding_ms=(
                self.post_padding_ms
                if self.post_padding_ms is not None
                else base.post_padding_ms
            ),
            min_duration_ms=(
                self.min_duration_ms
                if self.min_duration_ms is not None
                else base.min_duration_ms
            ),
            max_duration_ms=(
                self.max_duration_ms
                if self.max_duration_ms is not None
                else base.max_duration_ms
            ),
            duration_mode=self.duration_mode,
            target_duration_ms=self.target_duration_ms,
            output_format=self.output_format or base.output_format,
            crop_mode=self.crop_mode or base.crop_mode,
            encoding_mode=self.encoding_mode or base.encoding_mode,
            width=width,
            height=height,
        )
        values.bounds_config()
        return values


class ClipRead(BaseModel):
    """Full clip representation returned by the API."""

    id: str
    video_id: str
    highlight_id: str | None = None
    highlight_candidate_id: str | None = None
    status: str
    title: str | None = None
    start_ms: int
    end_ms: int
    duration_ms: int
    score: float | None = None
    output_path: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    aspect_ratio: str | None = None
    encoding_mode: str | None = None
    output_format: str | None = None
    crop_mode: str | None = None
    error: ClipErrorRead | None = None
    job_id: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    created_at_ms: int
    updated_at_ms: int


class ClipStatsRead(BaseModel):
    """Per-status clip counters of one video."""

    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    canceled: int


class ClipGenerateResponse(BaseModel):
    """Clips touched by one generate request (created, re-armed or already live)."""

    clips: list[ClipRead] = Field(default_factory=list)


def clip_to_read(clip: Clip) -> ClipRead:
    """Convert a clip row into its API representation."""
    error = None
    if clip.error_code or clip.error_message:
        error = ClipErrorRead(
            code=clip.error_code or "INTERNAL_ERROR",
            message=clip.error_message or "",
        )
    return ClipRead(
        id=clip.id,
        video_id=clip.video_id,
        highlight_id=clip.highlight_id,
        highlight_candidate_id=clip.highlight_candidate_id,
        status=clip.status,
        title=clip.title,
        start_ms=clip.start_ms,
        end_ms=clip.end_ms,
        duration_ms=clip.duration_ms,
        score=clip.score,
        output_path=clip.output_path,
        file_size=clip.file_size,
        width=clip.width,
        height=clip.height,
        fps=clip.fps,
        aspect_ratio=clip.aspect_ratio,
        encoding_mode=clip.encoding_mode,
        output_format=clip.output_format,
        crop_mode=clip.crop_mode,
        error=error,
        job_id=clip.job_id,
        started_at_ms=clip.started_at,
        finished_at_ms=clip.finished_at,
        created_at_ms=clip.created_at,
        updated_at_ms=clip.updated_at,
    )


__all__ = [
    "ClipErrorRead",
    "ClipGenerateRequest",
    "ClipGenerateResponse",
    "ClipRead",
    "ClipStatsRead",
    "clip_to_read",
]
