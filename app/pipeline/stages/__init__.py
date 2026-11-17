"""Stage registries: which stages run for which ``Job.type``.

A single ordered registry cannot serve two job kinds at once: a ``pipeline`` job
validates the queue with ``noop`` stages, while a ``transcribe`` job runs speech
recognition. :data:`PIPELINES` maps a job type onto its ordered stage tuple, and
:func:`stages_for` resolves it with a safe fallback, so an unknown type keeps
behaving like the original placeholder pipeline instead of crashing a worker.
"""

from __future__ import annotations

from typing import Final

from app.pipeline.stages.asr import AsrStage
from app.pipeline.stages.base import ProgressReporter, Stage, StageContext, StageResult
from app.pipeline.stages.highlights import HighlightStage
from app.pipeline.stages.noop import NoopStage
from app.services.highlight_service import HIGHLIGHT_JOB_TYPE
from app.services.transcript_service import ASR_JOB_TYPE as TRANSCRIBE_JOB_TYPE

#: Job type of the placeholder pipeline that exercises queue semantics.
PLACEHOLDER_JOB_TYPE: Final = "pipeline"

PIPELINES: Final[dict[str, tuple[Stage, ...]]] = {
    PLACEHOLDER_JOB_TYPE: (NoopStage("noop"), NoopStage("noop2")),
    TRANSCRIBE_JOB_TYPE: (AsrStage(),),
    HIGHLIGHT_JOB_TYPE: (HighlightStage(),),
}

#: Pipeline used for an unknown or missing job type (backwards compatible).
DEFAULT_PIPELINE: Final = PLACEHOLDER_JOB_TYPE


def stages_for(job_type: str | None) -> tuple[Stage, ...]:
    """Return the ordered stages of ``job_type``, falling back to the default."""
    key = (job_type or DEFAULT_PIPELINE).strip().lower()
    return PIPELINES.get(key, PIPELINES[DEFAULT_PIPELINE])


__all__ = [
    "DEFAULT_PIPELINE",
    "HIGHLIGHT_JOB_TYPE",
    "PIPELINES",
    "PLACEHOLDER_JOB_TYPE",
    "TRANSCRIBE_JOB_TYPE",
    "NoopStage",
    "ProgressReporter",
    "Stage",
    "StageContext",
    "StageResult",
    "stages_for",
]
