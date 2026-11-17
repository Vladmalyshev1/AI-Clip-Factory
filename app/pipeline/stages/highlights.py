"""Highlight detection stage (M4): find interesting moments in a transcript.

The stage owns the *sequencing*, not the algorithm: detection lives in
:mod:`app.ai.highlights` and persistence in
:mod:`app.services.highlight_service`. What happens here is:

1. load the video and its transcript, refusing anything but a completed one;
2. create (or re-open) the single highlight analysis of the video and start it;
3. detect the candidates over the stored segments (a few thousand rows at most,
   so the whole transcript is read once — no media file is ever touched);
4. replace the stored candidates and complete the analysis in one transaction.

Resume is trivial: an analysis recomputes the full candidate list every run, so
a crashed or canceled run simply moves back to ``pending`` and runs again.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Final

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.ai.highlights import HighlightConfig, detect_highlights
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.status import TranscriptStatus
from app.models.transcript import TranscriptSegment
from app.models.video import Video
from app.pipeline.stages.base import ProgressReporter, StageContext, StageResult
from app.services import highlight_service, transcript_service

logger = logging.getLogger(__name__)

STAGE_NAME: Final = "highlights"


def _require_settings(ctx: StageContext) -> Settings:
    """Return the settings of the job or fail with a clear programming error."""
    if ctx.settings is None:
        raise RuntimeError(f"Stage '{STAGE_NAME}' requires StageContext.settings")
    return ctx.settings


def _require_session_factory(ctx: StageContext) -> sessionmaker[Session]:
    """Return the session factory of the job or fail with a clear error."""
    if ctx.session_factory is None:
        raise RuntimeError(
            f"Stage '{STAGE_NAME}' requires StageContext.session_factory"
        )
    return ctx.session_factory


def _require_video_id(ctx: StageContext) -> str:
    """Return the job's video id; a detection job without one is invalid."""
    if not ctx.video_id:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"Job '{ctx.job_id}' has no video to analyse",
            status_code=422,
            details={"job_id": ctx.job_id},
        )
    return ctx.video_id


class HighlightStage:
    """Detect the highlight candidates of a video's completed transcript."""

    name: str = STAGE_NAME

    def run(self, ctx: StageContext, report: ProgressReporter) -> StageResult:
        """Detect and persist every highlight candidate of the job's video."""
        settings = _require_settings(ctx)
        factory = _require_session_factory(ctx)
        video_id = _require_video_id(ctx)
        config = HighlightConfig.from_settings(settings)

        highlight_id: str | None = None
        try:
            highlight_id, segments, duration_ms = self._prepare(
                factory, config, video_id
            )
            report.check_cancelled()
            report.update(0.4, "scoring transcript")
            candidates = detect_highlights(
                segments, duration_ms=duration_ms, config=config
            )
            report.check_cancelled()
            report.update(0.7, f"storing {len(candidates)} candidate(s)")
            with factory() as session:
                stored = highlight_service.replace_candidates(
                    session, highlight_id, video_id, candidates
                )
                highlight_service.complete_analysis(session, highlight_id)
                session.commit()

            report.update(1.0, f"detected {stored} highlight candidate(s)")
            logger.info(
                "Highlight analysis %s completed: %d candidate(s)",
                highlight_id,
                stored,
            )
            return StageResult(
                output={
                    "highlight_id": highlight_id,
                    "video_id": video_id,
                    "candidates": stored,
                    "duration_ms": duration_ms,
                },
                checkpoint={
                    "highlight_id": highlight_id,
                    "candidates": stored,
                },
            )
        except StageCancelled:
            self._pause(factory, highlight_id)
            raise
        except AppError as exc:
            self._fail(factory, highlight_id, exc.code.value, exc.message)
            raise
        except Exception as exc:
            logger.exception("Highlight detection failed for video %s", video_id)
            self._fail(
                factory,
                highlight_id,
                ErrorCode.HIGHLIGHT_FAILED.value,
                f"Highlight detection failed: {type(exc).__name__}: {exc}",
            )
            raise

    # ------------------------------------------------------------------ setup

    def _prepare(
        self,
        factory: sessionmaker[Session],
        config: HighlightConfig,
        video_id: str,
    ) -> tuple[str, list[TranscriptSegment], int]:
        """Validate the transcript, start the analysis and load the segments."""
        with factory() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise AppError(
                    ErrorCode.VIDEO_NOT_FOUND,
                    f"Video '{video_id}' not found",
                    status_code=404,
                    details={"video_id": video_id},
                )
            transcript = transcript_service.get_transcript(session, video_id)
            if transcript.status != TranscriptStatus.COMPLETED.value:
                raise AppError(
                    ErrorCode.TRANSCRIPT_INCOMPLETE,
                    f"Video '{video_id}' has no completed transcript "
                    f"(status: {transcript.status})",
                    status_code=409,
                    details={"video_id": video_id, "transcript_status": transcript.status},
                )
            duration_ms = int(transcript.duration_ms or 0)
            analysis = highlight_service.get_or_create_analysis(
                session, video_id, options=asdict(config)
            )
            highlight_service.start_analysis(session, analysis)
            highlight_id = analysis.id
            segments = list(
                session.scalars(
                    select(TranscriptSegment)
                    .where(TranscriptSegment.transcript_id == transcript.id)
                    .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
                ).all()
            )
            session.commit()
        return highlight_id, segments, duration_ms

    # -------------------------------------------------------------- bookkeeping

    def _fail(
        self,
        factory: sessionmaker[Session],
        highlight_id: str | None,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark the analysis failed (best effort; never masks the real error)."""
        if highlight_id is None:
            return
        try:
            with factory() as session:
                highlight_service.fail_analysis(
                    session,
                    highlight_id,
                    error_code=error_code,
                    error_message=error_message,
                )
                session.commit()
        except Exception:  # pragma: no cover - never mask the original failure
            logger.warning("Could not mark analysis %s as failed", highlight_id)

    def _pause(self, factory: sessionmaker[Session], highlight_id: str | None) -> None:
        """Return the analysis to ``pending`` so the next run retries it."""
        if highlight_id is None:
            return
        try:
            with factory() as session:
                highlight_service.pause_analysis(session, highlight_id)
                session.commit()
        except Exception:  # pragma: no cover - e.g. already completed
            logger.warning("Could not pause analysis %s", highlight_id, exc_info=True)


__all__ = ["STAGE_NAME", "HighlightStage"]
