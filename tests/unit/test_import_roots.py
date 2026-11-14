"""Resolution of the server-side import allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import Settings
from app.services.video_service import allowed_import_roots


@pytest.mark.unit
def test_import_roots_are_expanded_and_resolved(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, import_roots=[tmp_path, "~"])

    roots = allowed_import_roots(settings)

    assert tmp_path.resolve() in roots
    assert Path.home().resolve() in roots


@pytest.mark.unit
def test_empty_import_roots_deny_every_server_side_import() -> None:
    settings = Settings(_env_file=None, import_roots=[])

    assert allowed_import_roots(settings) == []
