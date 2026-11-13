"""Stage registry: ordered list of stages the orchestrator executes."""

from __future__ import annotations

from app.pipeline.stages.base import ProgressReporter, Stage, StageContext, StageResult
from app.pipeline.stages.noop import NoopStage

REGISTRY: tuple[Stage, ...] = (NoopStage("noop"), NoopStage("noop2"))

__all__ = [
    "REGISTRY",
    "NoopStage",
    "ProgressReporter",
    "Stage",
    "StageContext",
    "StageResult",
]
