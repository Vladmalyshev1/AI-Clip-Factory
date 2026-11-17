"""Clip editor service: drafts, validation, rename and transcript prefill (M7)."""

from __future__ import annotations

import json

import pytest
from app.core.clock import utc_now_ms
from app.core.errors import AppError, ErrorCode
from app.core.status import SourceKind, TranscriptStatus, VideoStatus
from app.models.clip import Clip
from app.models.job import new_id
from app.models.transcript import Transcript
from app.models.video import Video
from app.services import editor_service, transcript_service
from app.services.editor_service import EditorDraft, SubtitleCueEdit
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.db

DURATION_MS = 600_000


def _seed_video(session_factory: sessionmaker[Session], duration_ms: int | None = DURATION_MS) -> str:
    """Insert one ready video and return its id."""
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
                duration_ms=duration_ms,
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
    """Insert one clip row with a finished render and return it."""
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


def _seed_transcript(
    session_factory: sessionmaker[Session], video_id: str, *, completed: bool = True
) -> None:
    """Create a completed transcript with two segments inside the clip window."""
    now = utc_now_ms()
    with session_factory() as session:
        transcript = Transcript(
            video_id=video_id,
            language="en",
            model_name="fake",
            status=(
                TranscriptStatus.COMPLETED.value if completed else TranscriptStatus.PENDING.value
            ),
            duration_ms=DURATION_MS,
            chunk_seconds=300,
            created_at=now,
            updated_at=now,
        )
        session.add(transcript)
        session.flush()
        if completed:
            session.add_all(
                [
                    transcript_service.TranscriptSegment(
                        transcript_id=transcript.id,
                        chunk_index=0,
                        start_ms=100_000,
                        end_ms=105_000,
                        text="first phrase here",
                        created_at=now,
                    ),
                    transcript_service.TranscriptSegment(
                        transcript_id=transcript.id,
                        chunk_index=0,
                        start_ms=106_000,
                        end_ms=110_000,
                        text="second phrase",
                        created_at=now,
                    ),
                ]
            )
        session.commit()


def test_default_draft_mirrors_the_clip(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)

    assert draft.start_ms == clip.start_ms
    assert draft.end_ms == clip.end_ms
    assert draft.output_format == "vertical"
    assert draft.subtitles.enabled is False


def test_save_and_reload_a_draft(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)
        draft.end_ms = 125_000
        draft.subtitles.enabled = True
        draft.subtitles.style = "gaming"
        draft.subtitles.cues = [SubtitleCueEdit(start_ms=0, end_ms=2_000, text="hello")]
        editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)
        session.commit()

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        assert stored.edit_json is not None
        reloaded = EditorDraft.from_dict(json.loads(stored.edit_json))
    assert reloaded.end_ms == 125_000
    assert reloaded.subtitles.enabled is True
    assert reloaded.subtitles.style == "gaming"
    assert reloaded.subtitles.cues[0].text == "hello"
    # The clip's own render is untouched.
    assert clip.end_ms == 130_000


def test_draft_bounds_are_validated(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)
        draft.end_ms = draft.start_ms  # inverted

        with pytest.raises(AppError) as excinfo:
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_draft_end_beyond_the_video_is_rejected(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory, duration_ms=120_000)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)
        draft.end_ms = 200_000

        with pytest.raises(AppError) as excinfo:
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=120_000)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_draft_rejects_bad_mode_combination_and_colours(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)
        draft.encoding_mode = "fast"  # fast requires the original format

        with pytest.raises(AppError):
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)

        draft.encoding_mode = "accurate"
        draft.subtitles.enabled = True
        draft.subtitles.primary_colour = "red"
        with pytest.raises(AppError):
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)

        draft.subtitles.primary_colour = "#FF00AA"
        draft.subtitles.style = "nope"
        with pytest.raises(AppError):
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)


def test_cue_times_must_stay_inside_the_window(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        draft = editor_service.get_draft(session, clip.id)
        draft.subtitles.enabled = True
        draft.subtitles.cues = [SubtitleCueEdit(start_ms=0, end_ms=999_999, text="late")]

        with pytest.raises(AppError) as excinfo:
            editor_service.save_draft(session, clip.id, draft, video_duration_ms=DURATION_MS)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR


def test_transcript_prefill_builds_clip_relative_cues(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)
    _seed_transcript(session_factory, video_id, completed=True)

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        cues = editor_service.default_cue_edits_for(session, stored)

    assert [cue.text for cue in cues] == ["first phrase here", "second phrase"]
    assert cues[0].start_ms == 0  # clip-relative
    assert cues[1].end_ms == 10_000  # 110_000 - 100_000
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

def test_rename_trims_and_clears(session_factory: sessionmaker[Session]) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        renamed = editor_service.rename_clip(session, clip.id, "  my best clip  ")
        assert renamed.title == "my best clip"

        cleared = editor_service.rename_clip(session, clip.id, "   ")
        assert cleared.title is None

        with pytest.raises(AppError):
            editor_service.rename_clip(session, clip.id, "x" * 301)


def test_transcript_prefill_uses_only_completed_transcripts(
    session_factory: sessionmaker[Session],
) -> None:
    video_id = _seed_video(session_factory)
    clip = _seed_clip(session_factory, video_id)

    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        # No transcript at all.
        assert editor_service.list_clip_transcript(session, stored) == []
        assert editor_service.default_cue_edits_for(session, stored) == []

    _seed_transcript(session_factory, video_id, completed=False)
    with session_factory() as session:
        stored = session.get(Clip, clip.id)
        assert stored is not None
        # A pending transcript contributes no cues either.
        assert editor_service.default_cue_edits_for(session, stored) == []
