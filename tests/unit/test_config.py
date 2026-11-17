"""Settings resolution guards: absolute paths and the derived database URL."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import DEFAULT_DB_FILENAME, Settings


@pytest.mark.unit
def test_data_dir_is_resolved_to_an_absolute_path(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=Path("var"))
    assert settings.data_dir.is_absolute()
    assert settings.media_root.is_absolute()
    assert settings.work_dir.is_absolute()
    assert settings.logs_dir.is_absolute()


@pytest.mark.unit
def test_database_url_is_derived_under_data_dir(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.database_url is not None
    assert settings.database_url.startswith("sqlite+pysqlite:///")
    assert settings.database_url.endswith((tmp_path / DEFAULT_DB_FILENAME).as_posix())


@pytest.mark.unit
def test_explicit_database_url_is_preserved(tmp_path: Path) -> None:
    explicit = "sqlite+pysqlite:///C:/custom/other.db"
    settings = Settings(_env_file=None, data_dir=tmp_path, database_url=explicit)
    assert settings.database_url == explicit


@pytest.mark.unit
def test_clip_canvas_defaults_are_accepted(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert (settings.clip_vertical_width, settings.clip_vertical_height) == (1080, 1920)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"clip_vertical_width": 1080, "clip_vertical_height": 1080},
        {"clip_vertical_width": 1081, "clip_vertical_height": 1920},
        {"clip_vertical_width": 4, "clip_vertical_height": 8},
        {"clip_default_encoding_mode": "fast"},
        {"clip_min_duration_ms": 10_000, "clip_max_duration_ms": 5000},
    ],
)
def test_impossible_clip_settings_fail_at_startup(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=tmp_path, **overrides)  # type: ignore[arg-type]
