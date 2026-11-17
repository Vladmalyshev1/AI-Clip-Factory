"""Regression tests for the M7 review findings."""

from __future__ import annotations

import json

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError
from app.core.status import SourceKind, VideoStatus
from app.models.clip import Clip
from app.models.job import Job, new_id
from app.models.video import Video
from app.services import clip_service, editor_service, export_service
from app.services.editor_service import EditorDraft, SubtitleSettings
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

NOW = utc_now_ms()


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path)


def _seed_video(session_factory: sessionmaker[Session]) -> str:
    video_id = new_id()
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
                created_at=NOW,
                updated_at=NOW,
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
    parent_clip_id: str | None = None,
    output_path: str | None = None,
) -> Clip:
    with session_factory() as session:
        clip = Clip(
            id=new_id(),
            video_id=video_id,
            kind=kind,
            parent_clip_id=parent_clip_id,
            status=status,
            title="clip",
            start_ms=0,
            end_ms=30_000,
            duration_ms=30_000,
            output_format="vertical",
            crop_mode="pad",
            encoding_mode="accurate",
            width=1080,
            height=1920,
            output_path=output_path,
            created_at=NOW,
            updated_at=NOW,
        )
        session.add(clip)
        session.commit()
        return clip


def _lock(session_factory: sessionmaker[Session]) -> Session:
    session = session_factory()
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    return session


def test_a_fresh_draft_without_cues_can_be_saved(session_factory, tmp_path) -> None:
    """A clip without transcript cues must still allow bounds/format edits."""
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    session = _lock(session_factory)
    try:
        draft = editor_service.get_draft(session, clip.id)
        draft.end_ms = 25_000  # no cues, subtitles disabled
        saved = editor_service.save_draft(session, clip.id, draft, video_duration_ms=600_000)
        session.commit()
    finally:
        session.close()

    assert saved.end_ms == 25_000
    assert saved.subtitles.cues == []


def test_disabled_subtitles_may_drop_the_last_cue(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    draft = EditorDraft(
        start_ms=0,
        end_ms=30_000,
        subtitles=SubtitleSettings(enabled=False, cues=[]),
    )

    session = _lock(session_factory)
    try:
        editor_service.save_draft(session, clip.id, draft, video_duration_ms=600_000)
        session.commit()
    finally:
        session.close()


def test_export_rejects_an_invalid_draft_before_queueing(session_factory, tmp_path) -> None:
    """Subtitles enabled with no cues must be a 422, not a failed render job."""
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    draft = EditorDraft(
        start_ms=0,
        end_ms=30_000,
        subtitles=SubtitleSettings(enabled=True, cues=[]),
    )

    session = _lock(session_factory)
    try:
        with pytest.raises(AppError) as excinfo:
            export_service.create_export(
                session, clip.id, settings=_settings(tmp_path), draft=draft
            )
        session.rollback()
    finally:
        session.close()

    assert excinfo.value.code is not None
    with session_factory() as session:
        assert session.query(Clip).filter(Clip.parent_clip_id == clip.id).count() == 0


def test_export_rejects_fast_vertical_and_inverted_bounds(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    fast = EditorDraft(start_ms=0, end_ms=10_000, encoding_mode="fast")
    inverted = EditorDraft(start_ms=20_000, end_ms=10_000)

    session = _lock(session_factory)
    try:
        with pytest.raises(AppError):
            export_service.create_export(
                session, clip.id, settings=_settings(tmp_path), draft=fast
            )
        with pytest.raises(AppError):
            export_service.create_export(
                session, clip.id, settings=_settings(tmp_path), draft=inverted
            )
    finally:
        session.close()


def test_deleting_a_clip_removes_its_exports_and_files(session_factory, tmp_path) -> None:
    """A parent delete must not cascade rows while leaking their MP4 files."""
    settings = _settings(tmp_path)
    video_id = _seed_video(session_factory)
    export_file = settings.clips_dir / video_id / "export.mp4"
    export_file.parent.mkdir(parents=True, exist_ok=True)
    export_file.write_bytes(b"export body")

    parent = _seed_clip(session_factory, video_id, output_path=str(export_file))
    child = _seed_clip(
        session_factory,
        video_id,
        kind="export",
        parent_clip_id=parent.id,
        output_path=str(export_file),
    )

    session = _lock(session_factory)
    try:
        stored = session.get(Clip, parent.id)
        assert stored is not None
        clip_service.delete_clip(session, stored, settings=settings)
        session.commit()
    finally:
        session.close()

    assert not export_file.exists()
    with session_factory() as session:
        assert session.get(Clip, child.id) is None
        assert session.get(Clip, parent.id) is None


def test_deleting_a_clip_refuses_when_an_export_is_processing(
    session_factory, tmp_path
) -> None:
    settings = _settings(tmp_path)
    video_id = _seed_video(session_factory)
    parent = _seed_clip(session_factory, video_id)
    _seed_clip(
        session_factory,
        video_id,
        kind="export",
        status="processing",
        parent_clip_id=parent.id,
    )

    session = _lock(session_factory)
    try:
        stored = session.get(Clip, parent.id)
        assert stored is not None
        with pytest.raises(AppError) as excinfo:
            clip_service.delete_clip(session, stored, settings=settings)
    finally:
        session.rollback()
        session.close()

    assert excinfo.value.code.value == "CLIP_CONFLICT"


def test_export_respects_the_active_job_cap(session_factory, tmp_path) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    with session_factory() as session:
        session.add(
            Job(
                id=new_id(),
                type="clip",
                video_id=video_id,
                status="running",
                params_json=json.dumps({"clip_id": clip.id}),
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()
    capped = Settings(data_dir=tmp_path, clip_max_concurrent_jobs=1)

    session = _lock(session_factory)
    try:
        with pytest.raises(AppError) as excinfo:
            export_service.create_export(session, clip.id, settings=capped)
    finally:
        session.rollback()
        session.close()

    assert excinfo.value.code.value == "CLIP_CONFLICT"
