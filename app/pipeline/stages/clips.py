"""Clip render stage (M6): turn one clip row into one published MP4 file.

The stage owns the *sequencing*, not the algorithms: boundaries were resolved at
creation time, the command builder lives in :mod:`app.media.cut`, the process
runner in :mod:`app.media.runner`, validation in :mod:`app.media.validate` and
persistence in :mod:`app.services.clip_service`. What happens here is:

1. load the clip row and its source recording (refusing a missing source);
2. move the clip to ``processing`` and render into ``.<clip>.part.mp4``;
3. probe the temporary file against the expected contract;
4. atomically move it to ``<clip>.mp4`` and persist the published metadata.

Cancellation is enforced *inside* FFmpeg: the reporter is passed as the runner's
``cancel_check``, so a cancel request terminates the child process instead of
waiting for the render to finish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.paths import safe_name
from app.core.status import ClipStatus, JobStatus, parse_clip_status
from app.media.ass import (
    SubtitleCue,
    SubtitleStyle,
    build_ass_document,
    write_ass_file,
)
from app.media.cut import ClipSpec, CropMode, EncodingMode, OutputFormat, render_clip
from app.media.graph import VerticalLayout
from app.media.probe import ProbeError
from app.media.runner import FFmpegRunner, RenderError
from app.media.validate import RenderValidationError, validate_aspect_ratio, validate_render
from app.models.job import Job
from app.models.video import Video
from app.pipeline.stages.base import ProgressReporter, StageContext, StageResult
from app.services import clip_service, job_service
from app.services.clip_service import ClipOptions

logger = logging.getLogger(__name__)

STAGE_NAME: Final = "render"
PROGRESS_START: Final = 0.05
PROGRESS_SPAN: Final = 0.85


@dataclass(frozen=True)
class _RenderPlan:
    """Values copied out of the database before the long FFmpeg call."""

    clip_id: str
    video_id: str
    source_path: Path
    has_audio: bool
    start_ms: int
    end_ms: int
    duration_ms: int
    options: ClipOptions
    already_completed: bool
    #: Subtitle snapshot of an export (``None`` = no burn-in).
    subtitles: dict[str, Any] | None = None


def _require_settings(ctx: StageContext) -> Settings:
    """Return the settings of the job or fail with a clear programming error."""
    if ctx.settings is None:
        raise RuntimeError(f"Stage '{STAGE_NAME}' requires StageContext.settings")
    return ctx.settings


def _require_session_factory(ctx: StageContext) -> sessionmaker[Session]:
    """Return the session factory of the job or fail with a clear error."""
    if ctx.session_factory is None:
        raise RuntimeError(f"Stage '{STAGE_NAME}' requires StageContext.session_factory")
    return ctx.session_factory


def _require_video_id(ctx: StageContext) -> str:
    """Return the job's video id; a clip job without one is invalid."""
    if not ctx.video_id:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"Job '{ctx.job_id}' has no video to render",
            status_code=422,
            details={"job_id": ctx.job_id},
        )
    return ctx.video_id


def _clip_id(ctx: StageContext) -> str:
    """Return ``params["clip_id"]``; it is the only input a clip job carries."""
    value = ctx.params.get("clip_id")
    if not isinstance(value, str) or not value:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"Job '{ctx.job_id}' has no clip_id parameter",
            status_code=422,
            details={"job_id": ctx.job_id},
        )
    return value


def _paths(settings: Settings, video_id: str, clip_id: str) -> tuple[Path, Path]:
    """Return ``(temporary, final)`` paths inside the configured clip directory."""
    directory = settings.clips_dir / safe_name(video_id)
    stem = safe_name(clip_id)
    return directory / f".{stem}.part.mp4", directory / f"{stem}.mp4"


#: ASS colours are ``&HAABBGGRR``; the editor stores ``#RRGGBB``.
def _ass_colour(hex_colour: str, default: str) -> str:
    digits = hex_colour.lstrip("#")
    if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        return default
    red, green, blue = digits[0:2], digits[2:4], digits[4:6]
    return f"&H00{blue}{green}{red}".upper()


#: libass alignment (numpad layout) per editor position, for a 9:16 canvas.
_POSITION_ALIGNMENT: Final[dict[str, int]] = {"top": 8, "center": 5, "bottom": 2}
_STYLE_PRESETS: Final[dict[str, dict[str, int]]] = {
    # Differences from the default SubtitleStyle per editor preset.
    "classic": {"outline": 4, "shadow": 2, "bold": -1},
    "bold": {"outline": 6, "shadow": 0, "bold": -1},
    "gaming": {"outline": 3, "shadow": 3, "bold": -1, "margin_v": 320},
    "minimal": {"outline": 1, "shadow": 0, "bold": 0},
}


def _subtitle_style_from_draft(subtitles: dict[str, Any]) -> SubtitleStyle:
    """Map the editor's subtitle settings onto the ASS style of M5."""
    preset = _STYLE_PRESETS.get(str(subtitles.get("style") or "classic"), {})
    alignment = _POSITION_ALIGNMENT.get(str(subtitles.get("position") or "bottom"), 2)
    size = subtitles.get("font_size")
    margin_v = 220 if alignment != 5 else 0
    if alignment == 5:  # centred: keep it off the exact middle for readability
        margin_v = 320
    return SubtitleStyle(
        font_size=int(size) if isinstance(size, int) and 16 <= size <= 200 else 64,
        primary_colour=_ass_colour(str(subtitles.get("primary_colour") or ""), "&H00FFFFFF"),
        secondary_colour=_ass_colour(str(subtitles.get("highlight_colour") or ""), "&H000000FF"),
        alignment=alignment,
        margin_v=int(preset.get("margin_v", margin_v)),
        outline=int(preset.get("outline", 4)),
        shadow=int(preset.get("shadow", 2)),
        bold=int(preset.get("bold", -1)),
    )


def _write_subtitles_file(
    settings: Settings, video_id: str, clip_id: str, subtitles: dict[str, Any]
) -> Path:
    """Write the burn-in ``.ass`` file for one export from its snapshot."""
    raw_cues = subtitles.get("cues")
    if not isinstance(raw_cues, list) or not raw_cues:
        raise AppError(
            ErrorCode.SUBTITLE_FAILED,
            "Subtitles are enabled but the export snapshot has no cues",
            status_code=500,
            details={"clip_id": clip_id},
        )
    cues: list[SubtitleCue] = []
    for row in raw_cues:
        if not isinstance(row, dict):
            continue
        text = " ".join(str(row.get("text") or "").split())
        if not text:
            continue
        cues.append(
            SubtitleCue(
                text=text,
                start_ms=int(row.get("start_ms") or 0),
                end_ms=int(row.get("end_ms") or 0),
            )
        )
    if not cues:  # pragma: no cover - guarded by editor validation
        raise AppError(
            ErrorCode.SUBTITLE_FAILED,
            "Subtitle snapshot contains no usable cues",
            status_code=500,
            details={"clip_id": clip_id},
        )
    options = ClipOptions.from_settings(settings)
    document = build_ass_document(
        cues,
        _subtitle_style_from_draft(subtitles),
        video_width=options.width,
        video_height=options.height,
    )
    path = settings.clips_dir / safe_name(video_id) / f".{safe_name(clip_id)}.part.ass"
    write_ass_file(path, document)
    return path


def _plan_subtitles(clip: Any) -> dict[str, Any] | None:
    """Return the subtitle snapshot of an export row, ``None`` when disabled."""
    parsed = job_service.load_json_dict(clip.edit_json)
    if not parsed:
        return None
    subtitles = parsed.get("subtitles")
    return subtitles if isinstance(subtitles, dict) and subtitles.get("enabled") else None


class RenderClipStage:
    """Render one clip row with FFmpeg and publish the validated file."""

    name: str = STAGE_NAME

    def run(self, ctx: StageContext, report: ProgressReporter) -> StageResult:
        """Render and publish the clip referenced by ``params["clip_id"]``."""
        settings = _require_settings(ctx)
        factory = _require_session_factory(ctx)
        video_id = _require_video_id(ctx)
        clip_id = _clip_id(ctx)
        temporary: Path | None = None
        subtitles_file: Path | None = None

        try:
            plan = self._prepare(factory, clip_id, video_id, job_id=ctx.job_id)
            if plan.already_completed:
                report.update(1.0, "clip already rendered")
                return StageResult(output={"clip_id": clip_id, "skipped": True})

            temporary, final_path = _paths(settings, plan.video_id, plan.clip_id)
            subtitles_file = self._subtitles_path(settings, plan)
            spec = self._build_spec(plan, temporary, subtitles_path=subtitles_file)
            runner = FFmpegRunner.from_settings(settings)

            def _on_progress(fraction: float) -> None:
                report.update(
                    PROGRESS_START + PROGRESS_SPAN * fraction, "rendering clip"
                )

            render_clip(
                runner,
                spec,
                on_progress=_on_progress,
                cancel_check=report.check_cancelled,
                timeout_s=settings.clip_ffmpeg_timeout_s,
            )
            report.check_cancelled()
            report.update(0.92, "validating clip")

            info = validate_render(
                temporary,
                expected_duration_ms=(
                    plan.duration_ms
                    if spec.encoding_mode != EncodingMode.FAST
                    else None
                ),
                expected_width=(
                    plan.options.width
                    if spec.output_format == OutputFormat.VERTICAL
                    else None
                ),
                expected_height=(
                    plan.options.height
                    if spec.output_format == OutputFormat.VERTICAL
                    else None
                ),
                expect_audio=plan.has_audio,
                settings=settings,
            )
            if spec.output_format == OutputFormat.VERTICAL:
                validate_aspect_ratio(info)

            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(final_path)
            temporary = None

            try:
                with factory() as session:
                    clip_service.complete_clip(
                        session,
                        clip_id,
                        output_path=final_path,
                        file_size=info.size_bytes,
                        width=info.width,
                        height=info.height,
                        fps=info.fps,
                    )
                    session.commit()
            except Exception:
                # The clip was canceled or re-armed while FFmpeg finished:
                # publishing the file would leave one no row points at.
                self._cleanup(final_path)
                raise

            report.update(1.0, "clip ready")
            logger.info("Clip %s rendered to %s", clip_id, final_path)
            return StageResult(
                output={
                    "clip_id": clip_id,
                    "video_id": plan.video_id,
                    "output_path": str(final_path),
                    "duration_ms": info.duration_ms,
                    "width": info.width,
                    "height": info.height,
                },
                checkpoint={"clip_id": clip_id, "output_path": str(final_path)},
            )
        except StageCancelled:
            self._cleanup(temporary)
            self._on_cancelled(factory, clip_id, ctx.job_id)
            raise
        except AppError as exc:
            self._cleanup(temporary)
            self._fail(factory, clip_id, ctx.job_id, exc.code.value, exc.message)
            raise
        except (RenderError, RenderValidationError, ProbeError) as exc:
            self._cleanup(temporary)
            self._fail(factory, clip_id, ctx.job_id, exc.code.value, exc.message)
            raise AppError(
                exc.code,
                exc.message,
                details={"clip_id": clip_id},
            ) from exc
        except Exception as exc:
            self._cleanup(temporary)
            logger.exception("Clip render failed for clip %s", clip_id)
            message = f"Clip render failed: {type(exc).__name__}: {exc}"
            self._fail(factory, clip_id, ctx.job_id, ErrorCode.CLIP_FAILED.value, message)
            raise
        finally:
            # The burn-in file is an intermediate artefact of this run only.
            if subtitles_file is not None:
                self._cleanup(subtitles_file)

    # ------------------------------------------------------------------ setup

    def _prepare(
        self,
        factory: sessionmaker[Session],
        clip_id: str,
        video_id: str,
        *,
        job_id: str,
    ) -> _RenderPlan:
        """Load the clip and its source, moving the clip into ``processing``."""
        with factory() as session:
            clip = clip_service.get_clip(session, clip_id)
            if clip.video_id != video_id:
                raise AppError(
                    ErrorCode.CLIP_CONFLICT,
                    f"Clip '{clip_id}' does not belong to video '{video_id}'",
                    status_code=409,
                    details={"clip_id": clip_id, "video_id": video_id},
                )
            video = session.get(Video, clip.video_id)
            if video is None:
                raise AppError(
                    ErrorCode.VIDEO_NOT_FOUND,
                    f"Video '{clip.video_id}' not found",
                    status_code=404,
                    details={"video_id": clip.video_id},
                )
            source = Path(video.source_path)
            already_completed = parse_clip_status(clip.status) is ClipStatus.COMPLETED
            if not already_completed and not source.is_file():
                raise AppError(
                    ErrorCode.SOURCE_NOT_FOUND,
                    f"Source recording is missing: {source}",
                    status_code=404,
                    details={"video_id": video.id, "path": str(source)},
                )

            options = ClipOptions.from_dict(job_service.load_json_dict(clip.options_json) or {})
            if not already_completed:
                clip_service.start_processing(session, clip_id, job_id=job_id)
            subtitles = _plan_subtitles(clip)
            plan = _RenderPlan(
                clip_id=clip.id,
                video_id=clip.video_id,
                source_path=source,
                has_audio=bool(video.has_audio),
                start_ms=clip.start_ms,
                end_ms=clip.end_ms,
                duration_ms=clip.duration_ms,
                options=options,
                already_completed=already_completed,
                subtitles=subtitles,
            )
            session.commit()
        return plan

    def _build_spec(
        self, plan: _RenderPlan, output_path: Path, *, subtitles_path: Path | None = None
    ) -> ClipSpec:
        """Build the FFmpeg spec from the persisted clip row."""
        return ClipSpec(
            source_path=plan.source_path,
            output_path=output_path,
            start_ms=plan.start_ms,
            end_ms=plan.end_ms,
            subtitles_path=subtitles_path,
            layout=VerticalLayout(
                width=plan.options.width, height=plan.options.height
            ),
            encoding_mode=EncodingMode(plan.options.encoding_mode),
            output_format=OutputFormat(plan.options.output_format),
            crop_mode=CropMode(plan.options.crop_mode),
        )

    def _subtitles_path(self, settings: Settings, plan: _RenderPlan) -> Path | None:
        """Return the burn-in ``.ass`` path when the export snapshot enables it."""
        if plan.subtitles is None:
            return None
        return _write_subtitles_file(settings, plan.video_id, plan.clip_id, plan.subtitles)

    # -------------------------------------------------------------- bookkeeping

    def _cleanup(self, path: Path | None) -> None:
        """Remove a partial render or an unpublished file (never a published one)."""
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - locked by a dying child
            logger.warning("Could not delete partial clip %s", path)

    def _on_cancelled(
        self, factory: sessionmaker[Session], clip_id: str, job_id: str
    ) -> None:
        """Distinguish a user cancellation from a graceful worker shutdown."""
        try:
            with factory() as session:
                clip = clip_service.get_clip(session, clip_id)
                if clip.job_id != job_id:
                    # Another job owns the clip now; do not touch its state.
                    return
                job = session.get(Job, job_id)
                user_cancelled = job is not None and (
                    bool(job.cancel_requested)
                    or job.status
                    in (JobStatus.CANCEL_REQUESTED.value, JobStatus.CANCELED.value)
                )
                if user_cancelled:
                    clip_service.cancel_clip(session, clip)
                else:
                    clip_service.interrupt_clip(session, clip_id)
                session.commit()
        except Exception:  # pragma: no cover - never mask the cancellation
            logger.warning("Could not record cancellation of clip %s", clip_id, exc_info=True)

    def _fail(
        self,
        factory: sessionmaker[Session],
        clip_id: str,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark the clip failed (best effort; never masks the real error)."""
        try:
            with factory() as session:
                clip = clip_service.get_clip(session, clip_id)
                if clip.job_id != job_id:
                    # A newer job owns the clip: its own run reports the outcome.
                    logger.warning(
                        "Clip %s no longer belongs to job %s; not recording a failure",
                        clip_id,
                        job_id,
                    )
                    return
                clip_service.fail_clip(
                    session,
                    clip_id,
                    error_code=error_code,
                    error_message=error_message,
                )
                session.commit()
        except Exception:  # pragma: no cover - e.g. already completed
            logger.warning("Could not mark clip %s as failed", clip_id, exc_info=True)


__all__ = ["STAGE_NAME", "RenderClipStage"]
