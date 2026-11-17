"""``RenderClipStage``: one clip row in, one published MP4 file out (M6).

The real-FFmpeg tests prove the whole path (command builder, runner, validation,
atomic publish) on a generated 2-second recording; the monkeypatched ones cover
the failure paths the stage owns - partial file cleanup, error bookkeeping and
the difference between a user cancellation and a worker shutdown.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.status import ClipStatus, SourceKind, VideoStatus
from app.media.binaries import find_ffmpeg
from app.media.cut import ClipSpec, OutputFormat
from app.media.probe import probe_media
from app.media.runner import RenderError
from app.models.clip import Clip
from app.models.highlight import HighlightCandidate
from app.models.job import Job, new_id
from app.models.video import Video
from app.pipeline.stages import clips as clips_stage
from app.pipeline.stages.base import StageContext
from app.pipeline.stages.clips import RenderClipStage
from app.services import clip_service, highlight_service
from app.services.clip_service import ClipOptions
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

SOURCE_DURATION_MS = 2000
CLIP_START_MS = 0
CLIP_END_MS = 1500


@dataclass(frozen=True)
class Candidate:
    """Minimal candidate accepted by ``highlight_service.replace_candidates``."""

    start_ms: int
    end_ms: int
    score: float
    reason: str = "excited"
    transcript_text: str = "clip this"
    metadata: dict[str, object] | None = None


class RecordingReporter:
    """Mimics the orchestrator's reporter; optionally cancels on the first check."""

    def __init__(self, cancelled: bool = False) -> None:
        self.updates: list[tuple[float, str | None]] = []
        self.cancelled = cancelled

    def update(self, progress: float, message: str | None = None) -> None:
        self.updates.append((progress, message))

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise StageCancelled()


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two-second H.264/AAC MP4, 320x240 at 10 fps."""
    return _generate_source(tmp_path_factory, audio=True)


@pytest.fixture(scope="module")
def silent_source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two-second H.264 MP4 without an audio stream."""
    return _generate_source(tmp_path_factory, audio=False)


def _generate_source(tmp_path_factory: pytest.TempPathFactory, *, audio: bool) -> Path:
    """Generate one tiny source recording with the pinned FFmpeg build."""
    binary = find_ffmpeg(Settings(_env_file=None))
    if binary is None:  # pragma: no cover - environment problem
        pytest.skip("ffmpeg is not available in this environment")
    destination = tmp_path_factory.mktemp(f"clip-stage-{audio}") / "source.mp4"
    args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=10",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    args += [
        "-t",
        str(SOURCE_DURATION_MS / 1000),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
    ]
    args += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    result = subprocess.run(
        [
            str(binary),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *args,
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - environment problem
        raise RuntimeError(f"ffmpeg sample generation failed: {result.stderr[-1000:]}")
    return destination


def _small_options(**overrides: object) -> ClipOptions:
    """Default clip options with a tiny canvas and no padding around the candidate."""
    values: dict[str, object] = {
        "pre_padding_ms": 0,
        "post_padding_ms": 0,
        "min_duration_ms": 1000,
        "max_duration_ms": 10_000,
        "width": 270,
        "height": 480,
    }
    values.update(overrides)
    return ClipOptions(**values)  # type: ignore[arg-type]


def _seed_clip(
    session_factory: sessionmaker[Session],
    source: Path,
    *,
    has_audio: bool = True,
    options: ClipOptions | None = None,
) -> tuple[Clip, Job]:
    """Insert a ready video, a completed analysis, one candidate and its clip job."""
    now = utc_now_ms()
    video_id = new_id()
    with session_factory() as session:
        session.add(
            Video(
                id=video_id,
                title="VOD",
                original_filename="vod.mp4",
                source_path=str(source),
                source_kind=SourceKind.IMPORT_IN_PLACE.value,
                size_bytes=1024,
                duration_ms=SOURCE_DURATION_MS,
                has_audio=1 if has_audio else 0,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        analysis = highlight_service.get_or_create_analysis(session, video_id)
        highlight_service.start_analysis(session, analysis)
        highlight_service.replace_candidates(
            session,
            analysis.id,
            video_id,
            [Candidate(CLIP_START_MS, CLIP_END_MS, 0.9)],
        )
        highlight_service.complete_analysis(session, analysis.id)
        session.commit()
        candidate_id = session.scalars(
            select(HighlightCandidate).where(HighlightCandidate.video_id == video_id)
        ).one().id

    with session_factory() as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        clip = clip_service.create_clips(
            session,
            video_id,
            options=options or _small_options(),
            settings=Settings(_env_file=None, data_dir=Path("unused")),
            candidate_ids=[candidate_id],
        )[0]
        session.commit()
        job = session.get(Job, clip.job_id)
        assert job is not None
        return clip, job


def _run_stage(
    settings: Settings,
    session_factory: sessionmaker[Session],
    clip: Clip,
    job: Job,
    tmp_path: Path,
    *,
    reporter: RecordingReporter | None = None,
) -> RecordingReporter:
    """Run the render stage for one clip and return the reporter."""
    ctx = StageContext(
        job_id=job.id,
        video_id=clip.video_id,
        attempt=1,
        work_dir=tmp_path,
        params={"clip_id": clip.id},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )
    reporter = reporter or RecordingReporter()
    RenderClipStage().run(ctx, reporter)
    return reporter


def _stored(session_factory: sessionmaker[Session], clip_id: str) -> Clip:
    """Reload one clip row in a fresh session."""
    with session_factory() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        return clip


def _partial_files(settings: Settings, clip: Clip) -> list[Path]:
    """Return leftovers of the atomic-publish temporary file."""
    directory = settings.clips_dir / clip.video_id
    return list(directory.glob(".*.part.mp4")) if directory.is_dir() else []


@pytest.mark.media
def test_stage_renders_and_publishes_a_vertical_clip(
    session_factory: sessionmaker[Session], settings: Settings, source_video: Path, tmp_path: Path
) -> None:
    clip, job = _seed_clip(session_factory, source_video)

    reporter = _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    final_path = settings.clips_dir / clip.video_id / f"{clip.id}.mp4"
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path == str(final_path)
    assert final_path.is_file()
    assert _partial_files(settings, clip) == []
    assert (stored.width, stored.height) == (270, 480)
    assert stored.aspect_ratio == "270:480"
    assert stored.file_size == final_path.stat().st_size > 1024
    assert stored.fps == pytest.approx(10.0, abs=0.5)
    info = probe_media(final_path)
    assert info.has_audio is True
    assert info.duration_ms == pytest.approx(CLIP_END_MS - CLIP_START_MS, abs=750)
    assert reporter.updates[-1] == (1.0, "clip ready")


@pytest.mark.media
def test_second_run_of_a_completed_clip_is_skipped(
    session_factory: sessionmaker[Session], settings: Settings, source_video: Path, tmp_path: Path
) -> None:
    clip, job = _seed_clip(session_factory, source_video)
    _run_stage(settings, session_factory, clip, job, tmp_path)
    published = _stored(session_factory, clip.id).output_path
    assert published is not None
    mtime = Path(published).stat().st_mtime_ns

    reporter = _run_stage(settings, session_factory, clip, job, tmp_path)

    assert _stored(session_factory, clip.id).output_path == published
    assert Path(published).stat().st_mtime_ns == mtime
    assert reporter.updates[-1] == (1.0, "clip already rendered")


@pytest.mark.media
def test_interrupted_clip_is_rendered_again(
    session_factory: sessionmaker[Session], settings: Settings, source_video: Path, tmp_path: Path
) -> None:
    """A clip left in ``processing`` by a crashed worker is simply resumed."""
    clip, job = _seed_clip(session_factory, source_video)
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.PROCESSING.value
        session.commit()

    _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path is not None and Path(stored.output_path).is_file()


@pytest.mark.media
def test_silent_source_renders_without_an_audio_stream(
    session_factory: sessionmaker[Session],
    settings: Settings,
    silent_source_video: Path,
    tmp_path: Path,
) -> None:
    clip, job = _seed_clip(session_factory, silent_source_video, has_audio=False)

    _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path is not None
    assert probe_media(Path(stored.output_path)).has_audio is False


@pytest.mark.media
def test_fast_original_render_stream_copies_the_source(
    session_factory: sessionmaker[Session], settings: Settings, source_video: Path, tmp_path: Path
) -> None:
    clip, job = _seed_clip(
        session_factory,
        source_video,
        options=_small_options(output_format=OutputFormat.ORIGINAL.value, encoding_mode="fast"),
    )

    _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert stored.output_path is not None
    info = probe_media(Path(stored.output_path))
    assert (info.width, info.height) == (320, 240)
    assert info.video_codec == "h264"


def test_missing_source_fails_the_clip(
    session_factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    clip, job = _seed_clip(session_factory, tmp_path / "gone.mp4")

    with pytest.raises(AppError) as excinfo:
        _run_stage(settings, session_factory, clip, job, tmp_path)

    assert excinfo.value.code is ErrorCode.SOURCE_NOT_FOUND
    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.FAILED.value
    assert stored.error_code == ErrorCode.SOURCE_NOT_FOUND.value
    assert stored.output_path is None


def _patch_render(
    monkeypatch: pytest.MonkeyPatch, behaviour: Callable[..., object]
) -> None:
    """Replace ``render_clip`` inside the stage module with ``behaviour``."""
    monkeypatch.setattr(clips_stage, "render_clip", behaviour)


def test_render_error_fails_the_clip_and_removes_the_partial_file(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip, job = _seed_clip(session_factory, source_video)

    def failing(runner: object, spec: ClipSpec, **kwargs: object) -> None:
        Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(spec.output_path).write_bytes(b"partial")
        raise RenderError("encoder exploded", code=ErrorCode.RENDER_FAILED)

    _patch_render(monkeypatch, failing)

    with pytest.raises(AppError) as excinfo:
        _run_stage(settings, session_factory, clip, job, tmp_path)

    assert excinfo.value.code is ErrorCode.RENDER_FAILED
    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.FAILED.value
    assert stored.error_code == ErrorCode.RENDER_FAILED.value
    assert "encoder exploded" in (stored.error_message or "")
    assert _partial_files(settings, clip) == []
    assert not (settings.clips_dir / clip.video_id / f"{clip.id}.mp4").exists()


def test_render_timeout_is_recorded_as_a_failure(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip, job = _seed_clip(session_factory, source_video)

    def timing_out(runner: object, spec: ClipSpec, **kwargs: object) -> None:
        assert kwargs["timeout_s"] == settings.clip_ffmpeg_timeout_s
        raise RenderError("render timed out", code=ErrorCode.RENDER_TIMEOUT)

    _patch_render(monkeypatch, timing_out)

    with pytest.raises(AppError):
        _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.FAILED.value
    assert stored.error_code == ErrorCode.RENDER_TIMEOUT.value


def test_user_cancellation_cancels_the_clip(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip, job = _seed_clip(session_factory, source_video)
    with session_factory() as session:
        stored = session.get(Job, job.id)
        assert stored is not None
        stored.cancel_requested = 1
        session.commit()

    def cancelled(runner: object, spec: ClipSpec, **kwargs: object) -> None:
        cancel_check = kwargs["cancel_check"]
        assert callable(cancel_check)
        cancel_check()

    _patch_render(monkeypatch, cancelled)

    with pytest.raises(StageCancelled):
        _run_stage(
            settings,
            session_factory,
            clip,
            job,
            tmp_path,
            reporter=RecordingReporter(cancelled=True),
        )

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.CANCELED.value
    assert _partial_files(settings, clip) == []


def test_worker_shutdown_returns_the_clip_to_pending(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip, job = _seed_clip(session_factory, source_video)

    def cancelled(runner: object, spec: ClipSpec, **kwargs: object) -> None:
        cancel_check = kwargs["cancel_check"]
        assert callable(cancel_check)
        cancel_check()

    _patch_render(monkeypatch, cancelled)

    with pytest.raises(StageCancelled):
        _run_stage(
            settings,
            session_factory,
            clip,
            job,
            tmp_path,
            reporter=RecordingReporter(cancelled=True),
        )

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.PENDING.value
    assert stored.started_at is None


def test_invalid_render_output_fails_the_clip(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip, job = _seed_clip(session_factory, source_video)

    def writes_garbage(runner: object, spec: ClipSpec, **kwargs: object) -> None:
        Path(spec.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(spec.output_path).write_text("not an mp4", encoding="utf-8")
        return None

    _patch_render(monkeypatch, writes_garbage)

    with pytest.raises(AppError) as excinfo:
        _run_stage(settings, session_factory, clip, job, tmp_path)

    assert excinfo.value.code is ErrorCode.RENDER_FAILED
    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.FAILED.value
    assert _partial_files(settings, clip) == []


def test_job_without_a_clip_id_is_rejected(
    session_factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    ctx = StageContext(
        job_id="job-1",
        video_id="video-1",
        attempt=1,
        work_dir=tmp_path,
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )

    with pytest.raises(AppError) as excinfo:
        RenderClipStage().run(ctx, RecordingReporter())

    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


@pytest.mark.media
def test_unpublished_file_is_removed_when_the_row_cannot_be_completed(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip canceled between validation and the commit must not leave a file."""
    clip, job = _seed_clip(session_factory, source_video)

    def locked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(clip_service, "complete_clip", locked)

    with pytest.raises(RuntimeError):
        _run_stage(settings, session_factory, clip, job, tmp_path)

    final_path = settings.clips_dir / clip.video_id / f"{clip.id}.mp4"
    assert not final_path.exists()
    assert _partial_files(settings, clip) == []
    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.FAILED.value
    assert stored.output_path is None


def test_completed_clip_is_skipped_even_when_the_source_is_gone(
    session_factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    """Resume after a lost worker must not need the source of a finished clip."""
    clip, job = _seed_clip(session_factory, tmp_path / "gone.mp4")
    output = settings.clips_dir / clip.video_id / f"{clip.id}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"already published")
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        stored.status = ClipStatus.COMPLETED.value
        stored.output_path = str(output)
        session.commit()

    reporter = _run_stage(settings, session_factory, clip, job, tmp_path)

    stored = _stored(session_factory, clip.id)
    assert stored.status == ClipStatus.COMPLETED.value
    assert output.read_bytes() == b"already published"
    assert reporter.updates[-1] == (1.0, "clip already rendered")
