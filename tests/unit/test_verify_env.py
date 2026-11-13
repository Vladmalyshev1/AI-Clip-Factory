"""Tests for the environment verifier.

These are the first guards of the project: if the environment checks themselves
are broken, every later "tests passed" claim is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import verify_env as ve

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_python_check_passes_inside_supported_interpreter() -> None:
    check = ve.check_python()
    assert check.status == ve.OK, check.detail
    assert check.code == 0


@pytest.mark.unit
def test_failed_check_carries_a_distinct_exit_code() -> None:
    report = ve.Report(root=ROOT)
    report.add(ve.Check("ffmpeg", ve.FAIL, "not found", code=11))
    report.add(ve.Check("disk", ve.OK, "ok"))
    assert [c.code for c in report.failed] == [11]


@pytest.mark.media
def test_bundled_ffmpeg_is_resolved() -> None:
    ffmpeg = ve.resolve_ffmpeg(ROOT)
    assert ffmpeg is not None, "run tools\\setup.ps1 first"
    assert ffmpeg.is_file()


@pytest.mark.media
def test_ffmpeg_build_supports_subtitles_and_encoding() -> None:
    ffmpeg = ve.resolve_ffmpeg(ROOT)
    assert ffmpeg is not None
    version_check, version = ve.check_ffmpeg(ffmpeg)
    assert version_check.status == ve.OK, version_check.detail
    assert version is not None and version >= ve.MIN_FFMPEG

    assert ve.check_libass(ffmpeg).status == ve.OK
    assert ve.check_encoders(ffmpeg).status == ve.OK


@pytest.mark.media
def test_ffprobe_sits_next_to_ffmpeg() -> None:
    ffmpeg = ve.resolve_ffmpeg(ROOT)
    assert ffmpeg is not None
    ffprobe = ve.resolve_ffprobe(ffmpeg, ROOT)
    assert ffprobe is not None
    assert ffprobe.parent == ffmpeg.parent
