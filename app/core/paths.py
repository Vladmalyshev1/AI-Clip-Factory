"""Filesystem layout and safe path component helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings

_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
MAX_NAME_LENGTH = 80


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved directories used by the application."""

    data_dir: Path
    media_root: Path
    incoming_dir: Path
    originals_dir: Path
    renders_dir: Path
    thumbs_dir: Path
    work_dir: Path
    logs_dir: Path
    database_url: str


def workspace_layout(settings: Settings) -> WorkspaceLayout:
    """Build the workspace layout for the given settings."""
    return WorkspaceLayout(
        data_dir=settings.data_dir,
        media_root=settings.media_root,
        incoming_dir=settings.incoming_dir,
        originals_dir=settings.originals_dir,
        renders_dir=settings.renders_dir,
        thumbs_dir=settings.thumbs_dir,
        work_dir=settings.work_dir,
        logs_dir=settings.logs_dir,
        database_url=settings.database_url or "",
    )


def ensure_dirs(layout: WorkspaceLayout) -> None:
    """Create every directory of the layout."""
    for path in (
        layout.data_dir,
        layout.media_root,
        layout.incoming_dir,
        layout.originals_dir,
        layout.renders_dir,
        layout.thumbs_dir,
        layout.work_dir,
        layout.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str, *, fallback: str = "unnamed", max_length: int = MAX_NAME_LENGTH) -> str:
    """Return an ASCII-only, filesystem-safe path component.

    Job work directories are built from these components, so the result never
    contains separators, leading dots or non-ASCII characters.
    """
    cleaned = _UNSAFE_PATTERN.sub("_", value.strip()).strip("._-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._-")
    return cleaned or fallback
