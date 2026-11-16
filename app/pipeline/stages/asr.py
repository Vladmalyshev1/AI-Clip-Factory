"""Resumable speech-recognition stage (M3.4).

The stage owns the *sequencing* of speech recognition, not the algorithms: audio
extraction lives in :mod:`app.media.audio`, window slicing in
:mod:`app.media.audio_chunks`, decoding in :mod:`app.ai.asr` and persistence in
:mod:`app.services.transcript_service`. What happens here is:

1. create (or re-open) the single transcript of the video and start it;
2. extract a mono 16 kHz WAV **once** and reuse it while it still matches;
3. plan fixed-length chunks and create the missing chunk rows;
4. transcribe every *unfinished* chunk and commit after each one, so a crash or
   a cancellation loses at most the chunk that was in flight;
5. complete the transcript only when every chunk row is completed.

Resume is therefore chunk-granular: completed chunks keep their segments, a
``running`` chunk left by a crashed worker is claimed again, and a failed chunk
is retried up to ``ACF_ASR_CHUNK_MAX_ATTEMPTS`` times. Cancellation pauses the
transcript back to ``pending`` instead of failing it, so the next run continues.

Only one process at a time may hold the Whisper model: :class:`AsrProcessLock`
serialises workers so two 3-hour VODs cannot both allocate the model in the same
16 GB machine.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

from sqlalchemy.orm import Session, sessionmaker

from app.ai import asr as asr_api
from app.ai.asr import AsrChunkResult, AsrEngine, AsrError
from app.ai.model_manager import AsrProcessLock
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.media import audio as audio_io
from app.media import audio_chunks
from app.media.runner import FFmpegRunner, RenderCancelled, RenderError
from app.models.video import Video
from app.pipeline.stages.base import ProgressReporter, StageContext, StageResult
from app.services import transcript_service

logger = logging.getLogger(__name__)

STAGE_NAME: Final = "asr"
CHUNKS_DIRNAME: Final = "chunks"
LOCK_DIRNAME: Final = "locks"
LOCK_FILENAME: Final = "asr.lock"
PART_SUFFIX: Final = ".part"
#: Share of the stage progress spent on audio extraction (the rest is decoding).
EXTRACT_PROGRESS_SHARE: Final = 0.02


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
    """Return the job's video id; a transcription job without one is invalid."""
    if not ctx.video_id:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"Job '{ctx.job_id}' has no video to transcribe",
            status_code=422,
            details={"job_id": ctx.job_id},
        )
    return ctx.video_id


def _options(settings: Settings) -> dict[str, Any]:
    """Return the ASR configuration that produced a transcript."""
    return asr_api.transcription_options(settings)


def _audio_failure(action: str, exc: RenderError) -> AppError:
    """Map an FFmpeg failure during audio preparation onto ``ASR_FAILED``."""
    return AppError(
        ErrorCode.ASR_FAILED,
        f"Could not {action} the audio for speech recognition: {exc.message}",
        status_code=500,
        details={
            "stage": STAGE_NAME,
            "action": action,
            "render_code": exc.code.value,
            "returncode": exc.returncode,
            "stderr_tail": exc.stderr_tail or None,
        },
    )


class AsrStage:
    """Transcribe a video chunk by chunk, resuming finished work."""

    name: str = STAGE_NAME

    def __init__(self, engine: AsrEngine | None = None) -> None:
        # ``None`` means "build the process-wide engine on first use", which is
        # what the worker does; tests inject a fake engine instead.
        self._engine = engine

    def run(self, ctx: StageContext, report: ProgressReporter) -> StageResult:
        """Transcribe every unfinished chunk of the job's video."""
        settings = _require_settings(ctx)
        factory = _require_session_factory(ctx)
        video_id = _require_video_id(ctx)
        engine = self._engine if self._engine is not None else asr_api.get_engine(settings)

        transcript_id: str | None = None
        try:
            transcript_id, duration_ms, source_path, chunk_seconds = self._prepare(
                factory, settings, video_id, engine
            )
            audio_path = self._ensure_audio(settings, source_path, ctx.work_dir, duration_ms, report)
            audio_duration_ms = audio_io.wav_duration_ms(audio_path)
            if not audio_duration_ms:
                raise AppError(
                    ErrorCode.ASR_FAILED,
                    f"Extracted audio of video '{video_id}' is empty or unreadable",
                    status_code=500,
                    details={
                        "stage": STAGE_NAME,
                        "action": "extract",
                        "path": str(audio_path),
                    },
                )

            lock = AsrProcessLock(settings.work_dir / LOCK_DIRNAME / LOCK_FILENAME)
            # The model is loaded lazily by the first ``transcribe`` call, so the
            # lock has to be held from before the loop, not around a load call.
            lock.acquire(on_wait=report.check_cancelled)
            try:
                snapshot = self._transcribe_chunks(
                    factory=factory,
                    settings=settings,
                    transcript_id=transcript_id,
                    chunk_seconds=chunk_seconds,
                    audio_path=audio_path,
                    audio_duration_ms=audio_duration_ms,
                    engine=engine,
                    report=report,
                )
            finally:
                lock.release()

            with factory() as session:
                transcript_service.complete_transcript(
                    session, transcript_id, expected_ms=audio_duration_ms
                )
                segments_total = transcript_service.count_segments(session, transcript_id)
                session.commit()

            report.update(1.0, f"transcribed {snapshot.chunks_completed} chunk(s)")
            audio_io.remove_audio(audio_path)
            logger.info(
                "Transcript %s completed: %d chunk(s), %d segment(s)",
                transcript_id,
                snapshot.chunks_total,
                segments_total,
            )
            return StageResult(
                output={
                    "transcript_id": transcript_id,
                    "video_id": video_id,
                    "language": settings.asr_language,
                    "model": engine.model_name,
                    "segments": segments_total,
                    "chunks_total": snapshot.chunks_total,
                    "chunks_completed": snapshot.chunks_completed,
                    "duration_ms": duration_ms,
                },
                checkpoint={
                    "transcript_id": transcript_id,
                    "chunks_total": snapshot.chunks_total,
                    "chunks_completed": snapshot.chunks_completed,
                    "segments": segments_total,
                },
            )
        except StageCancelled:
            self._pause(factory, transcript_id)
            raise
        except AppError as exc:
            self._fail(factory, transcript_id, exc.code.value, exc.message)
            raise
        except Exception as exc:
            logger.exception("Speech recognition failed for video %s", video_id)
            self._fail(
                factory,
                transcript_id,
                ErrorCode.ASR_FAILED.value,
                f"Speech recognition failed: {type(exc).__name__}: {exc}",
            )
            raise

    # ------------------------------------------------------------------ setup

    def _prepare(
        self,
        factory: sessionmaker[Session],
        settings: Settings,
        video_id: str,
        engine: AsrEngine,
    ) -> tuple[str, int, Path, int]:
        """Create or re-open the single transcript of ``video_id`` and start it.

        Returns the transcript id, the probed duration, the source path and the
        chunk length to plan with. The chunk length is read back from the row (see
        :func:`transcript_service.adopt_chunk_grid`): a partially transcribed
        transcript keeps the grid it was started with, so a settings change can
        never re-tile finished chunks.
        """
        with factory() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise AppError(
                    ErrorCode.VIDEO_NOT_FOUND,
                    f"Video '{video_id}' not found",
                    status_code=404,
                    details={"video_id": video_id},
                )
            if not video.has_audio:
                raise AppError(
                    ErrorCode.NO_AUDIO_STREAM,
                    f"Video '{video_id}' has no audio stream to transcribe",
                    status_code=422,
                    details={"video_id": video_id},
                )
            duration_ms = int(video.duration_ms or 0)
            if duration_ms <= 0:
                raise AppError(
                    ErrorCode.PROBE_FAILED,
                    f"Video '{video_id}' has no known duration; probe it before transcribing",
                    status_code=422,
                    details={"video_id": video_id, "duration_ms": video.duration_ms},
                )
            source_path = Path(video.source_path)

            transcript = transcript_service.get_or_create_transcript(
                session,
                video_id,
                language=settings.asr_language,
                model_name=engine.model_name,
                duration_ms=duration_ms,
                chunk_seconds=settings.asr_chunk_seconds,
                options=_options(settings),
            )
            chunk_seconds = transcript_service.adopt_chunk_grid(
                session, transcript, settings.asr_chunk_seconds
            )
            transcript = transcript_service.start_transcript(session, transcript)
            transcript_id = transcript.id
            session.commit()
        return transcript_id, duration_ms, source_path, chunk_seconds

    def _ensure_audio(
        self,
        settings: Settings,
        source_path: Path,
        work_dir: Path,
        duration_ms: int,
        report: ProgressReporter,
    ) -> Path:
        """Extract the 16 kHz mono WAV once, reusing a matching previous run."""
        audio_path = work_dir / audio_io.AUDIO_FILENAME
        if audio_io.wav_matches(audio_path, duration_ms):
            logger.info("Reusing extracted audio %s", audio_path)
            return audio_path

        report.check_cancelled()
        report.update(0.0, "extracting audio")
        # A partial WAV from a crashed run must never look complete, so the
        # extraction targets a ``.part`` file that is renamed only on success.
        partial = audio_path.with_name(audio_path.name + PART_SUFFIX)

        def on_progress(fraction: float) -> None:
            report.check_cancelled()
            report.update(
                EXTRACT_PROGRESS_SHARE * max(0.0, min(1.0, fraction)),
                f"extracting audio ({fraction:.0%})",
            )

        try:
            runner = FFmpegRunner.from_settings(settings)
            audio_io.extract_audio(
                runner,
                source_path,
                partial,
                duration_ms=duration_ms,
                on_progress=on_progress,
                # A stuck FFmpeg may never emit progress again, so the runner has
                # to poll cancellation itself; otherwise the job is uncancellable.
                cancel_check=report.check_cancelled,
            )
            os.replace(partial, audio_path)
        except StageCancelled:
            audio_io.remove_audio(partial)
            raise
        except RenderCancelled as exc:
            audio_io.remove_audio(partial)
            raise StageCancelled(f"Audio extraction cancelled: {exc.message}") from exc
        except RenderError as exc:
            audio_io.remove_audio(partial)
            raise _audio_failure("extract", exc) from exc
        except OSError as exc:
            audio_io.remove_audio(partial)
            raise AppError(
                ErrorCode.ASR_FAILED,
                f"Could not store the extracted audio: {exc}",
                status_code=500,
                details={"stage": STAGE_NAME, "action": "extract", "error": str(exc)},
            ) from exc
        return audio_path

    # ------------------------------------------------------------- chunk loop

    def _transcribe_chunks(
        self,
        *,
        factory: sessionmaker[Session],
        settings: Settings,
        transcript_id: str,
        chunk_seconds: int,
        audio_path: Path,
        audio_duration_ms: int,
        engine: AsrEngine,
        report: ProgressReporter,
    ) -> transcript_service.TranscriptProgress:
        """Transcribe every claimable chunk and return the final progress."""
        plans = audio_chunks.plan_chunks(audio_duration_ms, chunk_seconds)
        with factory() as session:
            transcript = transcript_service.get_transcript_by_id(session, transcript_id)
            transcript_service.ensure_chunks(session, transcript, plans)
            session.commit()
            unfinished = {
                chunk.chunk_index
                for chunk in transcript_service.claimable_chunks(session, transcript_id)
            }
            snapshot = transcript_service.progress(session, transcript)

        todo = [plan for plan in plans if plan.index in unfinished]
        logger.info(
            "Transcript %s: %d of %d chunk(s) left to transcribe",
            transcript_id,
            len(todo),
            snapshot.chunks_total,
        )
        if not todo:
            return snapshot

        chunks_dir = audio_path.parent / CHUNKS_DIRNAME
        runner = FFmpegRunner.from_settings(settings)
        try:
            for plan in todo:
                report.check_cancelled()
                with factory() as session:
                    transcript_service.mark_chunk_running(
                        session, transcript_id, plan.index
                    )
                    session.commit()
                try:
                    result = self._transcribe_chunk(
                        runner=runner,
                        settings=settings,
                        audio_path=audio_path,
                        chunks_dir=chunks_dir,
                        plan=plan,
                        engine=engine,
                        report=report,
                    )
                except AppError as exc:
                    self._mark_chunk_failed(factory, transcript_id, plan.index, exc)
                    raise
                with factory() as session:
                    stored = transcript_service.save_chunk_result(
                        session, transcript_id, plan.index, result.segments
                    )
                    current = transcript_service.get_transcript_by_id(session, transcript_id)
                    snapshot = transcript_service.progress(session, current)
                    session.commit()
                report.update(
                    snapshot.progress,
                    f"transcribed chunk {plan.index + 1}/{snapshot.chunks_total} "
                    f"({stored} segment(s))",
                )
        finally:
            audio_chunks.cleanup_chunks(chunks_dir)
        return snapshot

    def _transcribe_chunk(
        self,
        *,
        runner: FFmpegRunner,
        settings: Settings,
        audio_path: Path,
        chunks_dir: Path,
        plan: audio_chunks.ChunkPlan,
        engine: AsrEngine,
        report: ProgressReporter,
    ) -> AsrChunkResult:
        """Slice, decode and transcribe one chunk, retrying transient failures."""
        target = audio_chunks.chunk_path(chunks_dir, plan.index)
        max_attempts = max(1, settings.asr_chunk_max_attempts)
        last_error: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            report.check_cancelled()
            try:
                audio_chunks.slice_chunk(
                    runner,
                    audio_path,
                    target,
                    plan,
                    cancel_check=report.check_cancelled,
                )
                samples = audio_io.read_audio_16k(target)
                return engine.transcribe(
                    samples,
                    offset_ms=plan.start_ms,
                    duration_ms=plan.duration_ms,
                    language=settings.asr_language,
                    should_cancel=report.check_cancelled,
                )
            except StageCancelled:
                raise
            except RenderCancelled as exc:
                raise StageCancelled(f"Chunk {plan.index} cancelled: {exc.message}") from exc
            except (AsrError, RenderError, ValueError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "Chunk %d of transcript audio failed (attempt %d/%d): %s",
                    plan.index,
                    attempt,
                    max_attempts,
                    exc,
                )
            finally:
                audio_chunks.remove_file(target)

        details: dict[str, Any] = {
            "stage": STAGE_NAME,
            "chunk_index": plan.index,
            "chunk_start_ms": plan.start_ms,
            "chunk_end_ms": plan.end_ms,
            "attempts": max_attempts,
        }
        if isinstance(last_error, RenderError):
            details.update(
                {
                    "reason": "audio",
                    "render_code": last_error.code.value,
                    "stderr_tail": last_error.stderr_tail or None,
                }
            )
        elif isinstance(last_error, AsrError):
            details.update({"reason": "decode", **last_error.details})
        else:
            details["reason"] = "audio"
        message = (
            f"Could not transcribe chunk {plan.index} "
            f"[{plan.start_ms}, {plan.end_ms}) ms after {max_attempts} attempt(s)"
        )
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise AppError(ErrorCode.ASR_FAILED, message, status_code=500, details=details)

    # -------------------------------------------------------------- bookkeeping

    def _mark_chunk_failed(
        self,
        factory: sessionmaker[Session],
        transcript_id: str,
        chunk_index: int,
        exc: AppError,
    ) -> None:
        """Record why a chunk failed; the row stays claimable for a retry."""
        try:
            with factory() as session:
                transcript_service.mark_chunk_failed(
                    session,
                    transcript_id,
                    chunk_index,
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
                session.commit()
        except Exception:  # pragma: no cover - never mask the original failure
            logger.exception("Could not record the failure of chunk %d", chunk_index)

    def _fail(
        self,
        factory: sessionmaker[Session],
        transcript_id: str | None,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark the transcript failed (best effort; never masks the real error)."""
        if transcript_id is None:
            return
        try:
            with factory() as session:
                transcript_service.fail_transcript(
                    session,
                    transcript_id,
                    error_code=error_code,
                    error_message=error_message,
                )
                session.commit()
        except Exception:  # pragma: no cover - e.g. already completed
            logger.warning(
                "Could not mark transcript %s as failed", transcript_id, exc_info=True
            )

    def _pause(self, factory: sessionmaker[Session], transcript_id: str | None) -> None:
        """Return the transcript to ``pending`` so the next run resumes it."""
        if transcript_id is None:
            return
        try:
            with factory() as session:
                transcript_service.pause_transcript(session, transcript_id)
                session.commit()
        except Exception:  # pragma: no cover - e.g. already completed
            logger.warning(
                "Could not pause transcript %s", transcript_id, exc_info=True
            )


__all__ = [
    "CHUNKS_DIRNAME",
    "EXTRACT_PROGRESS_SHARE",
    "LOCK_FILENAME",
    "PART_SUFFIX",
    "STAGE_NAME",
    "AsrStage",
]
