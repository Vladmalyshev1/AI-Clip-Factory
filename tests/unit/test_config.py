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
