"""Export service: creation, idempotency, retries and deletion (M7)."""

from __future__ import annotations

import json

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.status import JobStatus, SourceKind, VideoStatus
from app.models.clip import Clip
from app.models.job import Job, new_id
from app.models.video import Video
from app.services import export_service
from app.services.editor_service import EditorDraft, SubtitleCueEdit, SubtitleSettings
from app.services.job_service import load_json_dict
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path)


def _seed_video(session_factory: sessionmaker[Session]) -> str:
    video_id = new_id()
    now = utc_now_ms()
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title="VOD",
                original_filename="vod.mp4",
                source_path="C:/media/vod.mp4",
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=1024,
                duration_ms=600_000,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def _seed_clip(
    session_factory: sessionmaker[Session],
    video_id: str,
    *,
    kind: str = "auto",
    status: str = "completed",
) -> Clip:
    now = utc_now_ms()
    with session_factory() as session:
        clip = Clip(
            video_id=video_id,
            kind=kind,
            status=status,
            title="auto clip",
            start_ms=100_000,
            end_ms=130_000,
            duration_ms=30_000,
            output_format="vertical",
            crop_mode="pad",
            encoding_mode="accurate",
            width=1080,
            height=1920,
            output_path="C:/media/clips/x.mp4",
            created_at=now,
            updated_at=now,
        )
        session.add(clip)
        session.commit()
        return clip


def _enqueue_job(session_factory: sessionmaker[Session], video_id: str, clip_id: str) -> Job:
    now = utc_now_ms()
    with session_factory() as session:
        job = Job(
            id=new_id(),
            type="clip",
            video_id=video_id,
            status=JobStatus.QUEUED.value,
            progress=0.0,
            params_json=json.dumps({"clip_id": clip_id}),
            created_at=now,
            updated_at=now,
        )
        session.add(job)
        session.commit()
        return job


def _immediate(session_factory: sessionmaker[Session]) -> Session:
    """A session holding SQLite's write lock (BEGIN IMMEDIATE), as the API does."""
    session = session_factory()
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    return session


def test_export_requires_an_auto_source_clip(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    export = _seed_clip(session_factory, video_id, kind="export")

    session = _immediate(session_factory)
    try:
        with pytest.raises(AppError) as excinfo:
            export_service.create_export(session, export.id, settings=_settings(tmp_path))
    finally:
        session.rollback()
        session.close()
    assert excinfo.value.code is ErrorCode.CLIP_CONFLICT


def test_create_export_writes_a_pending_row_and_a_job(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    session = _immediate(session_factory)
    try:
        export, created = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
        session.commit()
    finally:
        session.close()

    assert created is True
    assert export.kind == "export"
    assert export.parent_clip_id == clip.id
    assert export.status == "pending"
    assert export.start_ms == clip.start_ms
    assert export.end_ms == clip.end_ms
    assert export.job_id is not None
    assert export.output_path is None


def test_export_of_a_draft_is_bound_by_the_draft(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    draft = EditorDraft(
        start_ms=105_000,
        end_ms=120_000,
        subtitles=SubtitleSettings(
            enabled=True,
            cues=[SubtitleCueEdit(start_ms=0, end_ms=1_000, text="hi")],
        ),
    )

    session = _immediate(session_factory)
    try:
        export, _ = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path), draft=draft
        )
        session.commit()
    finally:
        session.close()

    assert (export.end_ms - export.start_ms) == 15_000
    assert export.duration_ms == 15_000
    assert export.edit_json is not None
    subtitles = load_json_dict(export.edit_json)["subtitles"]
    assert subtitles["enabled"] is True
    assert subtitles["cues"][0]["text"] == "hi"


def test_second_create_while_active_is_idempotent(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    _enqueue_job(session_factory, video_id, "pending")

    session = _immediate(session_factory)
    try:
        first, created_first = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
        session.commit()
        second, created_second = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
    finally:
        session.close()

    assert created_first is True
    assert created_second is False
    assert first.id == second.id


def test_completed_export_blocks_until_new_settings(session_factory, tmp_path) -> None:
    """A finished export stays downloadable; new bounds create a second export."""
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    session = _immediate(session_factory)
    try:
        first, _ = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
        first.status = "completed"
        session.commit()
        second, created = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
        session.commit()
    finally:
        session.close()

    assert created is True
    assert second.id != first.id
    total = export_service.count_exports(session_factory(), clip.id)
    assert total == 2


def test_retry_and_cancel_reject_unknown_or_non_export_rows(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id, status="failed")

    session = _immediate(session_factory)
    try:
        # Unknown id → CLIP_NOT_FOUND; an auto clip is not an export → same.
        with pytest.raises(AppError) as missing_error:
            export_service.retry_export(session, "no-such-export")
        with pytest.raises(AppError) as retry_error:
            export_service.retry_export(session, clip.id)
        with pytest.raises(AppError) as cancel_error:
            export_service.cancel_export(session, clip.id)
    finally:
        session.close()
    assert missing_error.value.code is ErrorCode.CLIP_NOT_FOUND
    assert retry_error.value.code is ErrorCode.CLIP_NOT_FOUND
    assert cancel_error.value.code is ErrorCode.CLIP_NOT_FOUND


def test_delete_export_requires_a_terminal_status(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    session = _immediate(session_factory)
    try:
        export, _ = export_service.create_export(
            session, clip.id, settings=_settings(tmp_path)
        )
        session.commit()
        with pytest.raises(AppError) as excinfo:
            export_service.delete_export(session, export.id, settings=_settings(tmp_path))
    finally:
        session.close()
    assert excinfo.value.code is ErrorCode.CLIP_CONFLICT


def test_list_exports_only_returns_that_clip(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    other = _seed_clip(session_factory, video_id)

    session = _immediate(session_factory)
    try:
        export_service.create_export(session, clip.id, settings=_settings(tmp_path))
        session.commit()
    finally:
        session.close()

    items, total = export_service.list_exports(session_factory(), clip.id)
    assert total == 1
    assert items[0].parent_clip_id == clip.id
    assert export_service.list_exports(session_factory(), other.id) == ([], 0)
