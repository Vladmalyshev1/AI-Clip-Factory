"""Tests for animated ASS subtitle generation (no FFmpeg required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.media.ass import (
    EVENT_FORMAT,
    EVENTS_HEADER,
    SCRIPT_HEADER,
    STYLE_FORMAT,
    STYLES_HEADER,
    SubtitleCue,
    SubtitleStyle,
    build_ass_document,
    format_ass_time,
    write_ass_file,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [
        (0, "0:00:00.00"),
        (999, "0:00:00.99"),
        (1000, "0:00:01.00"),
        (61_234, "0:01:01.23"),
        (3_600_000, "1:00:00.00"),
        (3_723_456, "1:02:03.45"),
        (36_000_000, "10:00:00.00"),
    ],
)
def test_format_ass_time(milliseconds: int, expected: str) -> None:
    assert format_ass_time(milliseconds) == expected


def test_document_contains_the_required_sections() -> None:
    document = build_ass_document([SubtitleCue("Hello", 0, 1000)])
    assert document.startswith(SCRIPT_HEADER)
    assert "ScriptType: v4.00+" in document
    assert "PlayResX: 1080" in document
    assert "PlayResY: 1920" in document
    assert "WrapStyle: 2" in document
    assert "ScaledBorderAndShadow: yes" in document
    assert "YCbCr Matrix: TV.709" in document
    assert STYLES_HEADER in document
    assert STYLE_FORMAT in document
    assert EVENTS_HEADER in document
    assert EVENT_FORMAT in document
    assert document.count("Style: Default,") == 1


def test_style_line_field_order_matches_the_format() -> None:
    document = build_ass_document([])
    style_line = next(line for line in document.splitlines() if line.startswith("Style: Default,"))
    fields = style_line.removeprefix("Style: ").split(",")
    assert len(fields) == len(STYLE_FORMAT.removeprefix("Format: ").split(", "))
    assert fields[0] == "Default"
    assert fields[1] == "Arial"
    assert fields[2] == "64"
    assert fields[7] == "-1"
    assert fields[-1] == "1"


def test_custom_style_is_rendered() -> None:
    style = SubtitleStyle(font_name="Impact", font_size=72, margin_v=300)
    document = build_ass_document([], style)
    assert "Style: Default,Impact,72," in document
    assert ",300,1" in document


def test_one_dialogue_per_cue_with_the_right_times() -> None:
    document = build_ass_document(
        [SubtitleCue("Hello world", 500, 1500), SubtitleCue("Second", 1500, 2600)]
    )
    dialogues = [line for line in document.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert dialogues[0] == (
        "Dialogue: 0,0:00:00.50,0:00:01.50,Default,,0,0,0,,{\\fad(80,80)}Hello world"
    )
    assert dialogues[1].startswith("Dialogue: 0,0:00:01.50,0:00:02.60,")


def test_cues_are_sorted_and_overlaps_are_clamped() -> None:
    document = build_ass_document(
        [SubtitleCue("later", 1000, 2000), SubtitleCue("first", 0, 5000)]
    )
    dialogues = [line for line in document.splitlines() if line.startswith("Dialogue:")]
    assert dialogues[0].startswith("Dialogue: 0,0:00:00.00,0:00:01.00,")
    assert dialogues[1].startswith("Dialogue: 0,0:00:01.00,0:00:02.00,")


def test_duplicate_cues_collapse_to_a_single_dialogue() -> None:
    document = build_ass_document(
        [SubtitleCue("same", 0, 1000), SubtitleCue("same again", 0, 1000)]
    )
    dialogues = [line for line in document.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 1


def test_empty_and_whitespace_cues_are_dropped() -> None:
    document = build_ass_document([SubtitleCue("   ", 0, 1000), SubtitleCue("\n\t", 1000, 2000)])
    assert "Dialogue:" not in document
    assert document.startswith(SCRIPT_HEADER)


def test_braces_and_newlines_are_escaped() -> None:
    document = build_ass_document([SubtitleCue("  {evil}\n  injection  ", 0, 1000)])
    dialogue = next(line for line in document.splitlines() if line.startswith("Dialogue:"))
    assert dialogue.endswith("{\\fad(80,80)}\\{evil\\} injection")
    assert "evil}" not in dialogue


def test_long_text_wraps_into_at_most_max_lines() -> None:
    cue = SubtitleCue("one two three four five six seven eight", 0, 2000)
    document = build_ass_document(cue and [cue], max_chars_per_line=12, max_lines=3)
    dialogue = next(line for line in document.splitlines() if line.startswith("Dialogue:"))
    body = dialogue.split("{\\fad(80,80)}", 1)[1]
    lines = body.split("\\N")
    assert len(lines) == 3
    assert all(len(line) <= 12 for line in lines)
    assert lines[-1].endswith("...")
    assert "\n" not in dialogue


def test_short_text_stays_on_one_line() -> None:
    document = build_ass_document([SubtitleCue("short caption", 0, 1000)])
    dialogue = next(line for line in document.splitlines() if line.startswith("Dialogue:"))
    assert dialogue.endswith("{\\fad(80,80)}short caption")
    assert "\\N" not in dialogue


def test_zero_cues_still_produce_a_valid_document() -> None:
    document = build_ass_document([])
    assert document.startswith(SCRIPT_HEADER + "\n")
    assert document.endswith("\n")
    assert EVENTS_HEADER in document
    assert document.count("[Events]") == 1


def test_invalid_cue_times_are_rejected() -> None:
    with pytest.raises(ValueError):
        SubtitleCue("bad", 1000, 1000)
    with pytest.raises(ValueError):
        SubtitleCue("bad", 2000, 1000)
    with pytest.raises(ValueError):
        SubtitleCue("bad", -1, 1000)


def test_invalid_document_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_ass_document([], video_width=0)
    with pytest.raises(ValueError):
        build_ass_document([], max_chars_per_line=0)
    with pytest.raises(ValueError):
        build_ass_document([], max_lines=0)


def test_write_ass_file_is_utf8_without_bom_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "subs" / "clip.ass"
    document = build_ass_document([SubtitleCue("héllo", 0, 1000)])

    write_ass_file(destination, document)

    payload = destination.read_bytes()
    assert payload.startswith(b"[Script Info]")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in payload
    assert payload.decode("utf-8") == document
    assert list(destination.parent.glob("*.tmp")) == []
