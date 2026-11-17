"""``HighlightStage``: sequencing of detection against a real SQLite database.

The detector itself is pure (see ``tests/ai/test_highlight_detector.py``), so
these tests exercise exactly the stage's own logic: transcript validation,
analysis bookkeeping, candidate persistence and the failure/cancel paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from app.ai.asr import AsrSegment
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, StageCancelled
from app.core.status import HighlightStatus, SourceKind, VideoStatus
from app.models.highlight import Highlight
from app.models.video import Video
from app.pipeline.stages import highlights as highlights_stage
from app.pipeline.stages.base import StageContext
from app.pipeline.stages.highlights import HighlightStage
from app.services import highlight_service, transcript_service
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

#: 120 s -> exactly one 300 s chunk.
DURATION_MS = 120_000


@dataclass(frozen=True)
class Plan:
    """Minimal chunk plan accepted by ``transcript_service.ensure_chunks``."""

    index: int
    start_ms: int
    end_ms: int


class RecordingReporter:
    """Collects progress updates and mimics the orchestrator's reporter."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def update(self, progress: float, message: str | None = None) -> None:
        self.updates.append((progress, message))

    def check_cancelled(self) -> None:
        return None


def _insert_video(session_factory: sessionmaker[Session]) -> str:
    """Insert one probed source recording and return its id."""
    from app.core.clock import utc_now_ms
    from app.models.job import new_id

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
                duration_ms=DURATION_MS,
                has_audio=1,
                status=VideoStatus.READY.value,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return video_id


def _seed_transcript(
    session_factory: sessionmaker[Session],
    video_id: str,
    *,
    texts: list[str],
    completed: bool = True,
) -> None:
    """Create a transcript with one chunk holding ``texts`` as its segments."""
    with session_factory() as session:
        transcript = transcript_service.get_or_create_transcript(
            session,
            video_id,
            language="en",
            model_name="fake-whisper",
            duration_ms=DURATION_MS,
            chunk_seconds=300,
            options={"language": "en"},
        )
        transcript_service.ensure_chunks(
            session, transcript, [Plan(index=0, start_ms=0, end_ms=DURATION_MS)]
        )
        if not completed:
            session.commit()
            return
        transcript_service.start_transcript(session, transcript)
        transcript_service.mark_chunk_running(session, transcript.id, 0)
        transcript_service.save_chunk_result(
            session,
            transcript.id,
            0,
            [
                AsrSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                )
                for start_ms, end_ms, text in texts
            ],
        )
        transcript_service.complete_transcript(
            session, transcript.id, expected_ms=DURATION_MS
        )
        session.commit()


def _run_stage(
    settings: Settings,
    session_factory: sessionmaker[Session],
    video_id: str,
    tmp_path: Path,
) -> dict[str, Any]:
    """Run the highlight stage for ``video_id`` and return its output."""
    ctx = StageContext(
        job_id="job-1",
        video_id=video_id,
        attempt=1,
        work_dir=tmp_path,
        params={},
        checkpoint=None,
        settings=settings,
        session_factory=session_factory,
    )
    result = HighlightStage().run(ctx, RecordingReporter())
    return result.output


def test_a_completed_transcript_produces_candidates(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[
            (10_000 + index * 1000, 10_000 + index * 1000 + 900, "wow no way insane!")
            for index in range(20)
        ],
    )

    output = _run_stage(settings, session_factory, video_id, tmp_path)
    assert output["candidates"] >= 1

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert analysis.status == HighlightStatus.COMPLETED.value
        items, total = highlight_service.list_candidates(session, analysis.id)
    assert total >= 1
    assert all(0.0 <= item.score <= 1.0 for item in items)


def test_a_missing_transcript_is_rejected(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    with pytest.raises(AppError) as excinfo:
        _run_stage(settings, session_factory, video_id, tmp_path)
    assert excinfo.value.code is ErrorCode.TRANSCRIPT_NOT_FOUND

    with session_factory() as session:
        assert session.scalar(select(Highlight).where(Highlight.video_id == video_id)) is None


def test_an_incomplete_transcript_is_rejected(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(session_factory, video_id, texts=[], completed=False)

    with pytest.raises(AppError) as excinfo:
        _run_stage(settings, session_factory, video_id, tmp_path)
    assert excinfo.value.code is ErrorCode.TRANSCRIPT_INCOMPLETE
    assert excinfo.value.status_code == 409

    with session_factory() as session:
        assert session.scalar(select(Highlight).where(Highlight.video_id == video_id)) is None


def test_a_silent_transcript_succeeds_with_zero_candidates(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[(index * 1000, index * 1000 + 900, "we walk to the next objective") for index in range(20)],
    )

    output = _run_stage(settings, session_factory, video_id, tmp_path)
    assert output["candidates"] == 0

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert analysis.status == HighlightStatus.COMPLETED.value
        _, total = highlight_service.list_candidates(session, analysis.id)
    assert total == 0


def test_an_analysis_failure_is_recorded(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[(10_000, 10_900, "wow no way insane!")],
    )

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(highlights_stage, "detect_highlights", explode)
    with pytest.raises(RuntimeError):
        _run_stage(settings, session_factory, video_id, tmp_path)

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert analysis.status == HighlightStatus.FAILED.value
        assert analysis.error_code == ErrorCode.HIGHLIGHT_FAILED.value


def test_a_cancellation_pauses_the_analysis(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[(10_000, 10_900, "wow no way insane!")],
    )

    def cancelled(*args: Any, **kwargs: Any) -> None:
        raise StageCancelled("cancelled mid-detection")

    monkeypatch.setattr(highlights_stage, "detect_highlights", cancelled)
    with pytest.raises(StageCancelled):
        _run_stage(settings, session_factory, video_id, tmp_path)

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        assert analysis.status == HighlightStatus.PENDING.value


def test_rerunning_an_analysis_replaces_its_candidates(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[
            (10_000 + index * 1000, 10_000 + index * 1000 + 900, "wow no way insane!")
            for index in range(20)
        ],
    )
    _run_stage(settings, session_factory, video_id, tmp_path)
    # A re-run is what ``force=true`` triggers in the API: reset, then run again.
    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        highlight_service.reset_analysis(session, analysis.id, purge=True)
        session.commit()
    first = _run_stage(settings, session_factory, video_id, tmp_path)
    assert first["candidates"] >= 1

    with session_factory() as session:
        analysis = highlight_service.get_analysis(session, video_id)
        _, total = highlight_service.list_candidates(session, analysis.id)
    assert total == first["candidates"]


def test_the_transcript_is_never_modified(
    settings: Settings,
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    from app.models.transcript import TranscriptSegment

    video_id = _insert_video(session_factory)
    _seed_transcript(
        session_factory,
        video_id,
        texts=[
            (10_000 + index * 1000, 10_000 + index * 1000 + 900, "wow no way insane!")
            for index in range(20)
        ],
    )
    _run_stage(settings, session_factory, video_id, tmp_path)

    with session_factory() as session:
        transcript = transcript_service.get_transcript(session, video_id)
        assert transcript.status == "completed"
        segments = list(
            session.scalars(
                select(TranscriptSegment).where(
                    TranscriptSegment.transcript_id == transcript.id
                )
            )
        )
    assert len(segments) == 20
