"""Tests for the pure vertical filtergraph builder."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.media.graph import (
    VerticalLayout,
    build_ass_filter,
    build_clip_filtergraph,
    build_vertical_filter,
    escape_filter_path,
)

pytestmark = pytest.mark.unit

EXPECTED_1080X1920 = (
    "[0:v]split=2[bg][fg];"
    "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
    "boxblur=luma_radius=20:luma_power=2[bgb];"
    "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgs];"
    "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
)


def test_default_vertical_filter_is_exact() -> None:
    assert build_vertical_filter() == EXPECTED_1080X1920
    assert build_vertical_filter(VerticalLayout()) == EXPECTED_1080X1920


def test_vertical_filter_uses_the_layout_numbers() -> None:
    graph = build_vertical_filter(VerticalLayout(width=270, height=480, blur_sigma=8, blur_power=1))
    assert "scale=270:480:force_original_aspect_ratio=increase" in graph
    assert "crop=270:480" in graph
    assert "boxblur=luma_radius=8:luma_power=1" in graph
    assert "scale=270:480:force_original_aspect_ratio=decrease" in graph
    assert graph.endswith("format=yuv420p[vout]")


def test_vertical_filter_has_no_zoompan_newlines_or_trailing_semicolon() -> None:
    graph = build_vertical_filter()
    assert "zoompan" not in graph
    assert "\n" not in graph
    assert "\r" not in graph
    assert not graph.endswith(";")
    assert graph.count("[vout]") == 1


def test_clip_filtergraph_without_subtitles_is_the_vertical_graph() -> None:
    graph = build_clip_filtergraph()
    assert graph == EXPECTED_1080X1920
    assert "[vfinal]" not in graph


def test_clip_filtergraph_burns_subtitles_after_the_composition() -> None:
    graph = build_clip_filtergraph(ass_path=Path("C:/tmp/subs.ass"))
    assert graph.startswith(EXPECTED_1080X1920)
    assert graph.endswith("[vout]ass=C\\\\:/tmp/subs.ass[vfinal]")
    assert "[vfinal]" in graph
    assert graph.count("[vfinal]") == 1


def test_clip_filtergraph_respects_a_custom_layout() -> None:
    graph = build_clip_filtergraph(
        ass_path=Path("C:/tmp/subs.ass"), layout=VerticalLayout(width=270, height=480)
    )
    assert "crop=270:480" in graph
    assert "[vout]ass=C\\\\:/tmp/subs.ass[vfinal]" in graph


def test_escape_filter_path_uses_forward_slashes_and_escapes_colons() -> None:
    assert escape_filter_path("C:/a/b.ass") == "C\\\\:/a/b.ass"
    assert escape_filter_path(r"C:\a\b.ass") == "C\\\\:/a/b.ass"
    assert escape_filter_path(Path("C:/a/b.ass")) == "C\\\\:/a/b.ass"


def test_escape_filter_path_escapes_quotes_for_both_parser_levels() -> None:
    # One level escapes the quote, the second level escapes the backslash that
    # the first one introduced: three backslashes before the apostrophe.
    assert escape_filter_path("C:/it's/a.ass") == "C\\\\:/it\\\\\\'s/a.ass"


def test_escape_filter_path_escapes_filtergraph_syntax() -> None:
    assert escape_filter_path("/tmp/a,b;c[d].ass") == "/tmp/a\\,b\\;c\\[d\\].ass"
    assert escape_filter_path("/tmp/x=y.ass") == "/tmp/x=y.ass"


def test_escape_filter_path_keeps_cyrillic_and_spaces() -> None:
    assert escape_filter_path("/tmp/субтитры it's.ass") == "/tmp/субтитры it\\\\\\'s.ass"


def test_build_ass_filter_is_unquoted() -> None:
    # FFmpeg cannot represent a literal quote inside a quoted option value, so
    # the path is escaped instead of quoted.
    assert build_ass_filter("C:/a/b.ass") == "ass=C\\\\:/a/b.ass"
    assert build_ass_filter(Path("C:/a/b.ass")) == "ass=C\\\\:/a/b.ass"
    quoted_path = build_ass_filter("C:/it's/a.ass")
    assert "ass='" not in quoted_path
    assert quoted_path == "ass=" + escape_filter_path("C:/it's/a.ass")


@pytest.mark.parametrize(("width", "height"), [(1079, 1920), (1080, 1919), (0, 1920), (-2, 480)])
def test_layout_rejects_odd_or_non_positive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        VerticalLayout(width=width, height=height)


def test_layout_rejects_negative_blur_settings() -> None:
    with pytest.raises(ValueError):
        VerticalLayout(blur_sigma=-1)
    with pytest.raises(ValueError):
        VerticalLayout(blur_power=-1)
