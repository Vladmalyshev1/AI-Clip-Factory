"""Clip editor, transcript cues, exports and source streaming (M7).

Editor and export routes stay thin: validation lives in the editor/export
services, rendering in the existing M6 stage. The source endpoint streams the
immutable original with HTTP Range so the editor can scrub a 5-hour VOD.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Query, Request, Response
from fastapi import status as http_status
from fastapi.responses import FileResponse

from app.api.deps import DbSessionDep, SettingsDep
from app.core.errors import AppError, ErrorCode
from app.core.status import ClipStatus
from app.schemas.clip import ClipRead, clip_to_read
from app.schemas.editor import (
    EditorDraftEnvelope,
    EditorDraftRead,
    EditorDraftUpdate,
    ExportCreateRequest,
    ExportListResponse,
    SubtitleCueRead,
)
from app.services import clip_service, editor_service, export_service, video_service

editor_router = APIRouter(prefix="/clips", tags=["editor"])
export_router = APIRouter(prefix="/exports", tags=["exports"])
source_router = APIRouter(prefix="/videos", tags=["videos"])

# Export creation and listing hang off the clip prefix for a natural URL space.
clip_router = APIRouter(prefix="/clips", tags=["exports"])

CHUNK_BYTES = 1_000_000
#: Ranges up to this size are buffered; larger ones stream from disk.
MAX_BUFFERED_RANGE_BYTES = 8 * 1024 * 1024


# --------------------------------------------------------------------- editor


def _draft_to_read(draft: editor_service.EditorDraft) -> EditorDraftRead:
    subs = draft.subtitles
    return EditorDraftRead(
        start_ms=draft.start_ms,
        end_ms=draft.end_ms,
        output_format=draft.output_format,
        crop_mode=draft.crop_mode,
        encoding_mode=draft.encoding_mode,
        subtitles={
            "enabled": subs.enabled,
            "style": subs.style,
            "font_size": subs.font_size,
            "font_name": subs.font_name,
            "primary_colour": subs.primary_colour,
            "highlight_colour": subs.highlight_colour,
            "position": subs.position,
            "cues": [cue.to_dict() for cue in subs.cues],  # type: ignore[arg-type]
        },
    )


def _cue_reads(cues: list[editor_service.SubtitleCueEdit]) -> list[SubtitleCueRead]:
    return [
        SubtitleCueRead(start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text)
        for cue in cues
    ]


@editor_router.get("/{clip_id}/editor", response_model=EditorDraftEnvelope)
def get_editor(clip_id: str, session: DbSessionDep) -> EditorDraftEnvelope:
    """Return the saved draft (or the clip defaults) plus suggested cues."""
    clip = clip_service.get_clip(session, clip_id)
    draft = editor_service.get_draft(session, clip_id)
    segments = editor_service.list_clip_transcript(session, clip)
    suggested = editor_service.build_default_cue_edits(
        segments, start_ms=draft.start_ms, end_ms=draft.end_ms
    )
    return EditorDraftEnvelope(
        draft=_draft_to_read(draft),
        transcript_available=bool(segments),
        suggested_cues=_cue_reads(suggested),
    )


@editor_router.put("/{clip_id}/editor", response_model=EditorDraftRead)
def put_editor(
    clip_id: str, payload: EditorDraftUpdate, session: DbSessionDep, settings: SettingsDep
) -> EditorDraftRead:
    """Validate and save the draft; the clip's own render never changes."""
    # Read-modify-write under the write lock so two editor tabs cannot lose
    # each other's changes.
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    clip = clip_service.get_clip(session, clip_id)
    video = video_service.get_video(session, clip.video_id)
    current = editor_service.get_draft(session, clip_id)
    subs = payload.subtitles
    if subs is not None:
        current.subtitles.enabled = subs.enabled
        current.subtitles.style = subs.style
        current.subtitles.font_size = subs.font_size
        current.subtitles.font_name = subs.font_name
        current.subtitles.primary_colour = subs.primary_colour
        current.subtitles.highlight_colour = subs.highlight_colour
        current.subtitles.position = subs.position
        if subs.cues:
            current.subtitles.cues = [
                editor_service.SubtitleCueEdit(
                    start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text
                )
                for cue in subs.cues
            ]
    if payload.start_ms is not None:
        current.start_ms = payload.start_ms
    if payload.end_ms is not None:
        current.end_ms = payload.end_ms
    if payload.output_format is not None:
        current.output_format = payload.output_format
    if payload.crop_mode is not None:
        current.crop_mode = payload.crop_mode
    if payload.encoding_mode is not None:
        current.encoding_mode = payload.encoding_mode

    saved = editor_service.save_draft(
        session, clip_id, current, video_duration_ms=video.duration_ms
    )
    session.commit()
    return _draft_to_read(saved)


@editor_router.get("/{clip_id}/transcript")
def get_clip_transcript(clip_id: str, session: DbSessionDep) -> dict[str, object]:
    """Return the M3 phrases of the clip window, clip-relative (may be empty)."""
    clip = clip_service.get_clip(session, clip_id)
    segments = editor_service.list_clip_transcript(session, clip)
    from app.ai.subtitles import Segment as PlainSegment
    from app.ai.subtitles import build_cues

    plain = [PlainSegment(s.start_ms, s.end_ms, s.text) for s in segments]
    cues = build_cues(plain, start_ms=clip.start_ms, end_ms=clip.end_ms)
    return {
        "transcript_available": bool(segments),
        "cues": [
            {"start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text}
            for cue in cues
        ],
    }


# -------------------------------------------------------------------- exports


@clip_router.post(
    "/{clip_id}/exports",
    response_model=ExportListResponse,
    status_code=http_status.HTTP_201_CREATED,
)
def create_export(
    clip_id: str,
    session: DbSessionDep,
    settings: SettingsDep,
    payload: ExportCreateRequest | None = None,
) -> ExportListResponse:
    """Start (or return the in-flight) export of one clip from the given draft."""
    request = payload or ExportCreateRequest()
    # 404 for unknown ids before any lock is taken.
    clip_service.get_clip(session, clip_id)
    draft = editor_service.get_draft(session, clip_id)
    if request.start_ms is not None:
        draft.start_ms = request.start_ms
    if request.end_ms is not None:
        draft.end_ms = request.end_ms
    if request.output_format is not None:
        draft.output_format = request.output_format
    if request.crop_mode is not None:
        draft.crop_mode = request.crop_mode
    if request.encoding_mode is not None:
        draft.encoding_mode = request.encoding_mode
    if request.subtitles is not None:
        subs = request.subtitles
        draft.subtitles.enabled = subs.enabled
        draft.subtitles.style = subs.style
        draft.subtitles.font_size = subs.font_size
        draft.subtitles.font_name = subs.font_name
        draft.subtitles.primary_colour = subs.primary_colour
        draft.subtitles.highlight_colour = subs.highlight_colour
        draft.subtitles.position = subs.position
        draft.subtitles.cues = [
            editor_service.SubtitleCueEdit(
                start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text
            )
            for cue in subs.cues
        ]

    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    export, _created = export_service.create_export(
        session,
        clip_id,
        settings=settings,
        draft=draft,
    )
    session.commit()
    return ExportListResponse(items=[clip_to_read(export)], total=1)


@clip_router.get("/{clip_id}/exports", response_model=ExportListResponse)
def list_clip_exports(
    clip_id: str,
    session: DbSessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ExportListResponse:
    """List the exports of one clip, newest first."""
    clip_service.get_clip(session, clip_id)
    items, total = export_service.list_exports(session, clip_id, offset=offset, limit=limit)
    return ExportListResponse(
        items=[clip_to_read(item) for item in items], total=total
    )


@export_router.get("/{export_id}", response_model=ClipRead)
def get_export(export_id: str, session: DbSessionDep) -> ClipRead:
    """Return one export row as a clip representation."""
    export = export_service.get_export(session, export_id)
    return clip_to_read(export)


@export_router.post("/{export_id}/cancel", response_model=ClipRead)
def cancel_export(export_id: str, session: DbSessionDep) -> ClipRead:
    """Cancel a pending or processing export (running FFmpeg is terminated)."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    export = export_service.cancel_export(session, export_id)
    session.commit()
    return clip_to_read(export)


@export_router.post("/{export_id}/retry", response_model=ClipRead)
def retry_export(export_id: str, session: DbSessionDep) -> ClipRead:
    """Re-queue a failed or canceled export."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    export = export_service.retry_export(session, export_id)
    session.commit()
    return clip_to_read(export)


@export_router.delete("/{export_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_export(export_id: str, session: DbSessionDep, settings: SettingsDep) -> Response:
    """Delete one export (row + file); the source clip is never touched."""
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    export_service.delete_export(session, export_id, settings=settings)
    session.commit()
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@export_router.get("/{export_id}/file")
def download_export(export_id: str, session: DbSessionDep, settings: SettingsDep) -> FileResponse:
    """Serve the published file of one export (containment-checked)."""
    export = export_service.get_export(session, export_id)
    if export.status != ClipStatus.COMPLETED.value:
        raise AppError(
            ErrorCode.CLIP_NOT_FOUND,
            f"Export '{export_id}' has no rendered file yet",
            status_code=404,
            details={"export_id": export_id, "status": export.status},
        )
    path = clip_service.resolve_clip_file(settings, export)
    return FileResponse(path, media_type="video/mp4", filename=f"export-{export.id[:8]}.mp4")


# --------------------------------------------------------------------- source


def _range_response(
    path: str, file_size: int, range_header: str | None
) -> Response | FileResponse:
    """Build a 200 or 206 response for one Range request (bytes only)."""
    if not range_header or not range_header.strip().lower().startswith("bytes="):
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    spec = range_header.split("=", 1)[1].split(",")[0].strip()
    first, _, last = spec.partition("-")
    try:
        start = int(first) if first else 0
        end = int(last) if last else file_size - 1
    except ValueError:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "Malformed Range header",
            status_code=416,
            details={"range": range_header},
        ) from None
    if first == "":  # suffix form: last N bytes
        length = int(last) if last else 0
        start = max(0, file_size - length)
        end = file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "Requested range not satisfiable",
            status_code=416,
            details={"file_size": file_size, "range": range_header},
        )
    end = min(end, file_size - 1)
    length = end - start + 1
    if length <= MAX_BUFFERED_RANGE_BYTES:
        with open(path, "rb") as handle:
            handle.seek(start)
            data = handle.read(length)
        return Response(
            content=data,
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(data)),
            },
        )
    # Large ranges stream from disk: buffering a 5-hour VOD would OOM the API.
    return FileResponse(
        path,
        media_type="video/mp4",
        status_code=206,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
        },
    )


@source_router.get("/{video_id}/source", response_model=None)
def stream_source(
    video_id: str, session: DbSessionDep, request: Request
) -> Response | FileResponse:
    """Stream the immutable source recording with HTTP Range support."""
    video = video_service.get_video(session, video_id)
    path = video.source_path
    if not os.path.isfile(path):
        raise AppError(
            ErrorCode.SOURCE_NOT_FOUND,
            f"Source recording is missing: {path}",
            status_code=404,
            details={"video_id": video_id},
        )
    return _range_response(path, os.path.getsize(path), request.headers.get("range"))


__all__ = ["editor_router", "export_router", "source_router"]
