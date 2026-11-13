"""Placeholder stages that validate queue semantics (claim, resume, cancel, retry).

They touch no media: they tick, report progress, honour cancellation and can be
told to fail on purpose through job params.
"""

from __future__ import annotations

import time
from typing import Any

from app.pipeline.stages.base import ProgressReporter, StageContext, StageResult

DEFAULT_DURATION_S = 1.0
TICK_SECONDS = 0.05


def stage_duration(params: dict[str, Any], name: str) -> float:
    """Resolve the duration for a stage: ``<name>_duration_s`` then ``noop_duration_s``."""
    raw = params.get(f"{name}_duration_s", params.get("noop_duration_s", DEFAULT_DURATION_S))
    try:
        duration = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_DURATION_S
    return max(0.0, duration)


class NoopStage:
    """Sleeps for a configurable duration while reporting progress."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx: StageContext, report: ProgressReporter) -> StageResult:
        params = ctx.params
        if params.get("fail_at") == self.name:
            raise RuntimeError(f"stage '{self.name}' is configured to fail")

        duration = stage_duration(params, self.name)
        report.check_cancelled()
        started = time.monotonic()
        elapsed = 0.0
        while True:
            elapsed = time.monotonic() - started
            progress = 1.0 if duration <= 0 else min(1.0, elapsed / duration)
            report.update(progress, f"{self.name}: {elapsed:.2f}s of {duration:.2f}s")
            if progress >= 1.0:
                break
            time.sleep(min(TICK_SECONDS, max(0.0, duration - elapsed)))
            report.check_cancelled()

        report.update(1.0, f"{self.name}: done")
        return StageResult(
            output={"stage": self.name, "duration_s": duration},
            checkpoint={"completed": True, "duration_s": duration},
        )
