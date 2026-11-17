"""Request/response schemas for the clip editor and exports (M7)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.clip import ClipRead

MAX_CUES = 500


class SubtitleCueRead(BaseModel):
    """One editable caption."""

    start_ms: int
    end_ms: int
    text: str


class SubtitleSettingsRead(BaseModel):
    """Subtitle block of an editor draft."""

    enabled: bool = False
    style: str = "classic"
    font_size: int = Field(default=64, ge=16, le=200)
    font_name: str = "Arial"
    primary_colour: str = "#FFFFFF"
    highlight_colour: str = "#00E5FF"
    position: str = "bottom"
    cues: list[SubtitleCueRead] = Field(default_factory=list, max_length=MAX_CUES)

    @field_validator("primary_colour", "highlight_colour")
    @classmethod
    def _validate_colour(cls, value: str) -> str:
        digits = value[1:] if value.startswith("#") else value
        if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
            raise ValueError("colour must be a hex value like #FFFFFF")
        return f"#{digits.upper()}"


class EditorDraftRead(BaseModel):
    """Full editor draft of one clip."""

    start_ms: int
    end_ms: int
    output_format: str
    crop_mode: str
    encoding_mode: str
    subtitles: SubtitleSettingsRead


class EditorDraftUpdate(BaseModel):
    """Body of ``PUT /clips/{id}/editor`` (all fields optional)."""

    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    output_format: str | None = None
    crop_mode: str | None = None
    encoding_mode: str | None = None
    subtitles: SubtitleSettingsRead | None = None


class ClipRenameRequest(BaseModel):
    """Body of ``PATCH /clips/{id}``."""

    title: str = Field(min_length=0, max_length=300)


class EditorDraftEnvelope(BaseModel):
    """Editor state: the draft plus the pre-filled transcript cues."""

    draft: EditorDraftRead
    transcript_available: bool
    suggested_cues: list[SubtitleCueRead] = Field(default_factory=list)


class ExportCreateRequest(BaseModel):
    """Body of ``POST /clips/{id}/exports`` (omit to use the saved draft)."""

    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    output_format: str | None = None
    crop_mode: str | None = None
    encoding_mode: str | None = None
    subtitles: SubtitleSettingsRead | None = None


class ExportListResponse(BaseModel):
    """Page of exports of one clip (always small, so a plain list envelope)."""

    items: list[ClipRead]
    total: int
