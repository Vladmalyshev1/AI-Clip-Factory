"""Editor drafts: view and save one clip's editing state (M7).

The draft lives in ``clips.edit_json`` and never changes the clip's own render
parameters. Nothing here renders anything: the draft is applied only when an
export is created from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AppError, ErrorCode
from app.core.status import TranscriptStatus
from app.media.cut import CropMode, EncodingMode, OutputFormat
from app.models.clip import Clip
from app.models.transcript import TranscriptSegment
from app.services import clip_service, transcript_service

logger = logging.getLogger(__name__)

MAX_TITLE_LENGTH = 300
DEFAULT_FONT_SIZE = 64
DEFAULT_FONT_NAME = "Arial"
#: ASS colours are ``&HAABBGGRR``; the API accepts plain ``#RRGGBB``.
DEFAULT_PRIMARY_COLOUR = "#FFFFFF"
DEFAULT_HIGHLIGHT_COLOUR = "#00E5FF"
DEFAULT_POSITION = "bottom"

STYLES: tuple[str, ...] = ("classic", "bold", "gaming", "minimal")
POSITIONS: tuple[str, ...] = ("bottom", "center", "top")
FORMATS: tuple[str, ...] = (OutputFormat.ORIGINAL.value, OutputFormat.VERTICAL.value)
CROP_MODES: tuple[str, ...] = (CropMode.PAD.value, CropMode.CROP.value)
ENCODING_MODES: tuple[str, ...] = (EncodingMode.ACCURATE.value, EncodingMode.FAST.value)

MIN_EDIT_DURATION_MS = 1000
MAX_EDIT_DURATION_MS = 600_000


@dataclass
class SubtitleCueEdit:
    """One user-edited caption of the clip."""

    start_ms: int = 0
    end_ms: int = 0
    text: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleCueEdit:
        return cls(
            start_ms=max(0, int(data.get("start_ms") or 0)),
            end_ms=max(0, int(data.get("end_ms") or 0)),
            text=str(data.get("text") or ""),
        )


@dataclass
class SubtitleSettings:
    """User-facing subtitle settings stored in the editor draft."""

    enabled: bool = False
    style: str = "classic"
    font_size: int = DEFAULT_FONT_SIZE
    font_name: str = DEFAULT_FONT_NAME
    primary_colour: str = DEFAULT_PRIMARY_COLOUR
    highlight_colour: str = DEFAULT_HIGHLIGHT_COLOUR
    position: str = DEFAULT_POSITION
    cues: list[SubtitleCueEdit] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "style": self.style,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "primary_colour": self.primary_colour,
            "highlight_colour": self.highlight_colour,
            "position": self.position,
            "cues": [cue.to_dict() for cue in self.cues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleSettings:
        raw_cues = data.get("cues")
        cues = (
            [SubtitleCueEdit.from_dict(cue) for cue in raw_cues if isinstance(cue, dict)]
            if isinstance(raw_cues, list)
            else []
        )
        return cls(
            enabled=bool(data.get("enabled") or False),
            style=str(data.get("style") or "classic").lower(),
            font_size=int(data.get("font_size") or DEFAULT_FONT_SIZE),
            font_name=str(data.get("font_name") or DEFAULT_FONT_NAME),
            primary_colour=str(data.get("primary_colour") or DEFAULT_PRIMARY_COLOUR),
            highlight_colour=str(data.get("highlight_colour") or DEFAULT_HIGHLIGHT_COLOUR),
            position=str(data.get("position") or DEFAULT_POSITION).lower(),
            cues=cues,
        )


@dataclass
class EditorDraft:
    """Everything the clip editor shows and saves for one clip."""

    start_ms: int
    end_ms: int
    output_format: str = OutputFormat.VERTICAL.value
    crop_mode: str = CropMode.PAD.value
    encoding_mode: str = EncodingMode.ACCURATE.value
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "output_format": self.output_format,
            "crop_mode": self.crop_mode,
            "encoding_mode": self.encoding_mode,
            "subtitles": self.subtitles.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorDraft:
        raw_subs = data.get("subtitles")
        return cls(
            start_ms=max(0, int(data.get("start_ms") or 0)),
            end_ms=max(0, int(data.get("end_ms") or 0)),
            output_format=str(data.get("output_format") or OutputFormat.VERTICAL.value),
            crop_mode=str(data.get("crop_mode") or CropMode.PAD.value),
            encoding_mode=str(data.get("encoding_mode") or EncodingMode.ACCURATE.value),
            subtitles=(
                SubtitleSettings.from_dict(raw_subs)
                if isinstance(raw_subs, dict)
                else SubtitleSettings()
            ),
        )


def default_draft(clip: Clip) -> EditorDraft:
    """The draft a fresh editor session starts with (the clip as rendered)."""
    options = clip_service.ClipOptions.from_dict({})
    return EditorDraft(
        start_ms=clip.start_ms,
        end_ms=clip.end_ms,
        output_format=clip.output_format or options.output_format,
        crop_mode=clip.crop_mode or options.crop_mode,
        encoding_mode=clip.encoding_mode or options.encoding_mode,
    )


def _validate_enum(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"{name} must be one of: {', '.join(allowed)}",
            status_code=422,
            details={"value": value, "allowed": list(allowed)},
        )


def _validate_colour(value: str, name: str) -> None:
    digits = value[1:] if value.startswith("#") else value
    if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"{name} must be a hex colour like #FFFFFF",
            status_code=422,
            details={"value": value},
        )


def validate_draft(draft: EditorDraft, clip: Clip, video_duration_ms: int | None) -> None:
    """Reject drafts the render pipeline could never honour (API + exports)."""
    if draft.end_ms <= draft.start_ms:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "end_ms must be greater than start_ms",
            status_code=422,
            details={"start_ms": draft.start_ms, "end_ms": draft.end_ms},
        )
    if draft.start_ms < 0:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "start_ms must not be negative",
            status_code=422,
            details={"start_ms": draft.start_ms},
        )
    if video_duration_ms and draft.end_ms > video_duration_ms:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "end_ms exceeds the source video duration",
            status_code=422,
            details={"end_ms": draft.end_ms, "video_duration_ms": video_duration_ms},
        )
    duration = draft.end_ms - draft.start_ms
    if not MIN_EDIT_DURATION_MS <= duration <= MAX_EDIT_DURATION_MS:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"clip duration must be between {MIN_EDIT_DURATION_MS} and "
            f"{MAX_EDIT_DURATION_MS} ms",
            status_code=422,
            details={"duration_ms": duration},
        )
    _validate_enum("output_format", draft.output_format, FORMATS)
    _validate_enum("crop_mode", draft.crop_mode, CROP_MODES)
    _validate_enum("encoding_mode", draft.encoding_mode, ENCODING_MODES)
    if draft.encoding_mode == EncodingMode.FAST.value and (
        draft.output_format != OutputFormat.ORIGINAL.value
    ):
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "fast encoding requires the original output format",
            status_code=422,
        )
    subs = draft.subtitles
    _validate_enum("subtitle style", subs.style, STYLES)
    _validate_enum("subtitle position", subs.position, POSITIONS)
    if not 16 <= subs.font_size <= 200:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "subtitle font_size must be between 16 and 200",
            status_code=422,
            details={"font_size": subs.font_size},
        )
    _validate_colour(subs.primary_colour, "primary_colour")
    _validate_colour(subs.highlight_colour, "highlight_colour")
    if subs.enabled and not 1 <= len(subs.cues) <= 500:
        # Disabled subtitles may keep zero cues; enabled ones need content.
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "subtitles.cues must contain between 1 and 500 cues when subtitles are enabled",
            status_code=422,
            details={"count": len(subs.cues)},
        )
    window = draft.end_ms - draft.start_ms
    for cue in subs.cues:
        if not cue.text.strip():
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "subtitle cues must not be empty",
                status_code=422,
            )
        if cue.end_ms <= cue.start_ms:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "cue end_ms must be greater than its start_ms",
                status_code=422,
                details={"cue": cue.to_dict()},
            )
        if cue.start_ms < 0 or cue.end_ms > window:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "cue times must stay inside the clip window",
                status_code=422,
                details={"cue": cue.to_dict(), "window_ms": window},
            )


def get_draft(session: Session, clip_id: str) -> EditorDraft:
    """Return the saved draft, or the clip's own settings for a fresh session."""
    clip = clip_service.get_clip(session, clip_id)
    if clip.edit_json:
        from app.services.job_service import load_json_dict

        return EditorDraft.from_dict(load_json_dict(clip.edit_json))
    return default_draft(clip)


def save_draft(
    session: Session,
    clip_id: str,
    draft: EditorDraft,
    *,
    video_duration_ms: int | None = None,
) -> EditorDraft:
    """Validate and store the draft on the clip row (never touches the render)."""
    clip = clip_service.get_clip(session, clip_id)
    validate_draft(draft, clip, video_duration_ms)
    from app.services.job_service import dump_json

    # dump_json returns a str; a draft round-trips through edit_json directly.
    clip.edit_json = dump_json(draft.to_dict())
    session.flush()
    return draft


def rename_clip(session: Session, clip_id: str, title: str) -> Clip:
    """Set a new title (empty string clears it back to the default)."""
    clip = clip_service.get_clip(session, clip_id)
    cleaned = title.strip()
    if len(cleaned) > MAX_TITLE_LENGTH:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"title must be at most {MAX_TITLE_LENGTH} characters",
            status_code=422,
        )
    clip.title = cleaned or None
    session.flush()
    return clip


def list_clip_transcript(session: Session, clip: Clip) -> list[TranscriptSegment]:
    """Return the M3 segments overlapping the clip window (may be empty)."""
    transcript = transcript_service.find_transcript(session, clip.video_id)
    if transcript is None or transcript.status != TranscriptStatus.COMPLETED.value:
        return []
    segments, _ = transcript_service.list_segments(
        session,
        transcript.id,
        start_ms=clip.start_ms,
        end_ms=clip.end_ms,
        limit=2000,
    )
    return segments


def build_default_cue_edits(
    segments: list[TranscriptSegment], *, start_ms: int, end_ms: int
) -> list[SubtitleCueEdit]:
    """Pre-fill editable cues from the M3 segments of the clip window."""
    from app.ai.subtitles import Segment as PlainSegment
    from app.ai.subtitles import build_cues

    plain = [PlainSegment(s.start_ms, s.end_ms, s.text) for s in segments]
    return [
        SubtitleCueEdit(start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text)
        for cue in build_cues(plain, start_ms=start_ms, end_ms=end_ms)
    ]


def default_cue_edits_for(session: Session, clip: Clip) -> list[SubtitleCueEdit]:
    """Convenience wrapper: the pre-filled cues of one clip."""
    segments = list_clip_transcript(session, clip)
    return build_default_cue_edits(segments, start_ms=clip.start_ms, end_ms=clip.end_ms)


def existing_draft_ids(session: Session) -> list[str]:
    """Ids of clips that carry a saved draft (diagnostics/tests only)."""
    return list(
        session.scalars(select(Clip.id).where(Clip.edit_json.is_not(None))).all()
    )


__all__ = [
    "MAX_TITLE_LENGTH",
    "EditorDraft",
    "SubtitleCueEdit",
    "SubtitleSettings",
    "build_default_cue_edits",
    "default_cue_edits_for",
    "default_draft",
    "existing_draft_ids",
    "get_draft",
    "list_clip_transcript",
    "rename_clip",
    "save_draft",
    "validate_draft",
]
