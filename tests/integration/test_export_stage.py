"""Export rendering end to end: ASS burn-in through the real FFmpeg (M7).

The real-FFmpeg tests prove that an export row with a subtitle snapshot renders
a burned-in MP4, while M6 auto clips keep rendering without one. The remaining
tests cover the failure paths owned by the stage (bad snapshot, cleanup).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from app.core.config import Settings
from app.core.errors import AppError
from app.core.status import ClipStatus, SourceKind, VideoStatus
from app.media.binaries import find_ffmpeg
from app.media.probe import probe_media
from app.models.clip import Clip
from app.models.job import Job, new_id
from app.models.video import Video
from app.pipeline.stages.base import StageContext
from app.pipeline.stages.clips import RenderClipStage
from app.services import export_service
from app.services.editor_service import SubtitleCueEdit, SubtitleSettings
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.media

SOURCE_DURATION_MS = 2000
NOW_MS = 1_760_000_000_000


class Reporter:
    """Minimal reporter: records progress, never cancels."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def update(self, progress: float, message: str | None = None) -> None:
        self.updates.append((progress, message))

    def check_cancelled(self) -> None:
        return None


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two-second H.264/AAC MP4, 320x240 at 10 fps."""
    binary = find_ffmpeg(Settings(_env_file=None))
    if binary is None:  # pragma: no cover - environment problem
        pytest.skip("ffmpeg is not available in this environment")
    destination = tmp_path_factory.mktemp("export-stage") / "source.mp4"
    result = subprocess.run(
        [
            str(binary), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", str(SOURCE_DURATION_MS / 1000),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(destination),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:  # pragma: no cover - environment problem
        raise RuntimeError(f"ffmpeg sample generation failed: {result.stderr[-800:]}")
    return destination


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


@dataclass
class Seeded:
    """Everything one stage run needs."""

    settings: Settings
    factory: sessionmaker[Session]
    source: Path


def _seed_source(session_factory: sessionmaker[Session], source: Path) -> str:
    video_id = new_id()
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title="VOD",
                original_filename="vod.mp4",
                source_path=str(source),
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=source.stat().st_size,
                duration_ms=SOURCE_DURATION_MS,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=NOW_MS,
                updated_at=NOW_MS,
            )
        )
        session.commit()
    return video_id


def _seed_auto_clip(session_factory: sessionmaker[Session], video_id: str) -> Clip:
    """A completed auto clip covering the first 1.5 seconds."""
    with session_factory() as session:
        clip = Clip(
            video_id=video_id,
            kind="auto",
            status=ClipStatus.COMPLETED.value,
            title="auto",
            start_ms=0,
            end_ms=1_500,
            duration_ms=1_500,
            output_format="vertical",
            crop_mode="pad",
            encoding_mode="accurate",
            width=270,
            height=480,
            created_at=NOW_MS,
            updated_at=NOW_MS,
        )
        session.add(clip)
        session.commit()
        return clip


def _seed_export(
    session_factory: sessionmaker[Session], source_clip: Clip, *, subtitles: bool
) -> tuple[Clip, Job]:
    """Queue one export of ``source_clip`` with (or without) a subtitle snapshot."""
    with session_factory() as session:
        stored = session.get(Clip, source_clip.id)
        assert stored is not None
        draft = None
        if subtitles:
            draft = None
        session.rollback()

    session = session_factory()
    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    try:
        draft = None
        if subtitles:
            # Save a draft with subtitles, then export from it.
            from app.services import editor_service
            from app.services.editor_service import EditorDraft

            draft = EditorDraft(
                start_ms=source_clip.start_ms,
                end_ms=source_clip.end_ms,
                subtitles=SubtitleSettings(
                    enabled=True,
                    style="bold",
                    font_size=48,
                    primary_colour="#FFFFFF",
                    cues=[
                        SubtitleCueEdit(start_ms=0, end_ms=800, text="hello clip"),
                        SubtitleCueEdit(start_ms=800, end_ms=1_500, text="привет субтитры!"),
                    ],
                ),
            )
            editor_service.save_draft(session, source_clip.id, draft, video_duration_ms=2000)
        export, _created = export_service.create_export(
            session,
            source_clip.id,
            settings=_settings(Path("unused")),
            draft=draft,
        )
        session.commit()
    finally:
        session.close()
    job = session_factory().get(Job, export.job_id)
    assert job is not None
    return export, job


def _run(settings: Settings, factory: sessionmaker[Session], clip: Clip, job: Job) -> None:
    ctx = StageContext(
        job_id=job.id,
        video_id=clip.video_id,
        attempt=1,
        work_dir=settings.work_dir,
        params={"clip_id": clip.id},
        checkpoint=None,
        settings=settings,
        session_factory=factory,
    )
    RenderClipStage().run(ctx, Reporter())


def _stored(factory: sessionmaker[Session], clip_id: str) -> Clip:
    with factory() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        return clip


def _assert_no_partials(settings: Settings, video_id: str) -> None:
    directory = settings.clips_dir / video_id
    if directory.is_dir():
        assert list(directory.glob(".*")) == []


def test_export_with_subtitles_burns_cyrillic_cues(
    session_factory: sessionmaker[Session], source_video: Path, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    video_id = _seed_source(session_factory, source_video)
    auto = _seed_auto_clip(session_factory, video_id)
    export, job = _seed_export(session_factory, auto, subtitles=True)

    _run(settings, session_factory, export, job)

    stored = _stored(session_factory, export.id)
    final_path = settings.clips_dir / export.video_id / f"{export.id}.mp4"
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path == str(final_path)
    assert final_path.is_file()
    info = probe_media(final_path)
    assert (info.width, info.height) == (270, 480)
    assert info.has_audio is True
    assert info.duration_ms == pytest.approx(1_500, abs=750)
    # The ASS intermediate is gone, the source clip row is untouched.
    _assert_no_partials(settings, export.video_id)
    assert _stored(session_factory, auto.id).output_path is None
    assert _stored(session_factory, auto.id).status == ClipStatus.COMPLETED.value


def test_export_without_subtitles_renders_plain_video(
    session_factory: sessionmaker[Session], source_video: Path, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    video_id = _seed_source(session_factory, source_video)
    auto = _seed_auto_clip(session_factory, video_id)
    export, job = _seed_export(session_factory, auto, subtitles=False)

    _run(settings, session_factory, export, job)

    stored = _stored(session_factory, export.id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path is not None
    info = probe_media(Path(stored.output_path))
    assert (info.width, info.height) == (270, 480)
    assert info.has_audio is True
    _assert_no_partials(settings, export.video_id)


def test_auto_clip_generation_still_renders_without_subtitles(
    session_factory: sessionmaker[Session], source_video: Path, tmp_path: Path
) -> None:
    """M6 path: an auto clip with no edit_json renders exactly as before."""
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    video_id = _seed_source(session_factory, source_video)
    with session_factory() as session:
        clip = Clip(
            video_id=video_id,
            kind="auto",
            status=ClipStatus.PENDING.value,
            start_ms=0,
            end_ms=1_500,
            duration_ms=1_500,
            output_format="vertical",
            crop_mode="pad",
            encoding_mode="accurate",
            options_json='{"pre_padding_ms":0,"post_padding_ms":0,"min_duration_ms":1000,'
            '"max_duration_ms":10000,"duration_mode":"candidate","target_duration_ms":null,'
            '"output_format":"vertical","crop_mode":"pad","encoding_mode":"accurate",'
            '"width":270,"height":480}',
            width=270,
            height=480,
            created_at=NOW_MS,
            updated_at=NOW_MS,
        )
        session.add(clip)
        session.flush()
        from app.services import job_service

        job = job_service.enqueue(session, type="clip", video_id=video_id,
                                  params={"clip_id": clip.id})
        clip.job_id = job.id
        session.commit()
        clip_id, job_id = clip.id, job.id

    ctx = StageContext(
        job_id=job_id,
        video_id=video_id,
        attempt=1,
        work_dir=settings.work_dir,
        params={"clip_id": clip_id},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )
    RenderClipStage().run(ctx, Reporter())

    stored = _stored(session_factory, clip_id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path is not None


def test_corrupt_subtitle_snapshot_fails_the_export(
    session_factory: sessionmaker[Session], source_video: Path, tmp_path: Path
) -> None:
    """An enabled snapshot without cues fails clearly instead of rendering."""
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    video_id = _seed_source(session_factory, source_video)
    with session_factory() as session:
        clip = Clip(
            video_id=video_id,
            kind="export",
            status=ClipStatus.PENDING.value,
            start_ms=0,
            end_ms=1_500,
            duration_ms=1_500,
            output_format="vertical",
            crop_mode="pad",
            encoding_mode="accurate",
            options_json='{"pre_padding_ms":0,"post_padding_ms":0,"min_duration_ms":1000,'
            '"max_duration_ms":10000,"duration_mode":"candidate","target_duration_ms":null,'
            '"output_format":"vertical","crop_mode":"pad","encoding_mode":"accurate",'
            '"width":270,"height":480}',
            edit_json='{"subtitles":{"enabled":true,"cues":[]}}',
            width=270,
            height=480,
            created_at=NOW_MS,
            updated_at=NOW_MS,
        )
        session.add(clip)
        session.flush()
        from app.services import job_service

        job = job_service.enqueue(session, type="clip", video_id=video_id,
                                  params={"clip_id": clip.id})
        clip.job_id = job.id
        session.commit()
        clip_id, job_id = clip.id, job.id

    ctx = StageContext(
        job_id=job_id,
        video_id=video_id,
        attempt=1,
        work_dir=settings.work_dir,
        params={"clip_id": clip_id},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )
    with pytest.raises(AppError):
        RenderClipStage().run(ctx, Reporter())

    stored = _stored(session_factory, clip_id)
    assert stored.status == ClipStatus.FAILED.value
    assert stored.error_code == "SUBTITLE_FAILED"
    _assert_no_partials(settings, video_id)
