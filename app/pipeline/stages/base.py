"""Stage plugin interface shared by the orchestrator and concrete stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


class ProgressReporter(Protocol):
    """Callbacks handed to a stage while it runs."""

    def update(self, progress: float, message: str | None = None) -> None:
        """Report stage progress in ``[0, 1]`` (persisted at a throttled rate)."""

    def check_cancelled(self) -> None:
        """Raise :class:`~app.core.errors.StageCancelled` when cancellation was requested."""


@dataclass(frozen=True)
class StageContext:
    """Everything a stage needs; carries no *open* database session on purpose.

    A stage that has to persist progress (chunked transcription, for example)
    receives the application :class:`~app.core.config.Settings` and the
    :class:`~sqlalchemy.orm.sessionmaker` instead, so it owns its short
    transactions and never holds a connection across a long media operation.
    Both are optional because a purely computational stage does not need them.
    """

    job_id: str
    video_id: str | None
    attempt: int
    work_dir: Path
    params: dict[str, Any]
    checkpoint: dict[str, Any] | None
    settings: Settings | None = None
    session_factory: sessionmaker[Session] | None = None


@dataclass(frozen=True)
class StageResult:
    """Outcome of a finished stage, persisted atomically by the orchestrator."""

    output: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] | None = None


class Stage(Protocol):
    """A resumable unit of pipeline work."""

    name: str

    def run(self, ctx: StageContext, report: ProgressReporter) -> StageResult:
        """Execute the stage and return its output."""
