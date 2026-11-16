"""Stage registry: one pipeline per ``Job.type`` (M3.4).

The registry is what makes unrelated job kinds coexist: ``pipeline`` keeps the
placeholder ``noop``/``noop2`` stages that the queue tests rely on, while
``transcribe`` runs speech recognition. Unknown types must fall back instead of
leaving a claimed job without any stage.
"""

from __future__ import annotations

import pytest
from app.pipeline.stages import (
    DEFAULT_PIPELINE,
    PIPELINES,
    PLACEHOLDER_JOB_TYPE,
    TRANSCRIBE_JOB_TYPE,
    stages_for,
)
from app.services.job_service import _registry_stage_names

pytestmark = pytest.mark.unit


def test_every_pipeline_has_named_stages() -> None:
    assert PIPELINES
    for job_type, stages in PIPELINES.items():
        assert stages, f"pipeline '{job_type}' has no stages"
        assert all(stage.name for stage in stages)


def test_placeholder_pipeline_is_unchanged() -> None:
    assert [stage.name for stage in stages_for(PLACEHOLDER_JOB_TYPE)] == ["noop", "noop2"]


def test_transcribe_pipeline_runs_the_asr_stage() -> None:
    assert [stage.name for stage in stages_for(TRANSCRIBE_JOB_TYPE)] == ["asr"]


@pytest.mark.parametrize("job_type", [None, "", "unknown", "  TranscriBE  "])
def test_unknown_types_fall_back_to_the_default_pipeline(job_type: str | None) -> None:
    expected = ["noop", "noop2"]
    if job_type is not None and job_type.strip().lower() == TRANSCRIBE_JOB_TYPE:
        expected = ["asr"]
    assert [stage.name for stage in stages_for(job_type)] == expected
    assert DEFAULT_PIPELINE == PLACEHOLDER_JOB_TYPE


def test_stage_names_follow_the_job_type() -> None:
    assert _registry_stage_names(TRANSCRIBE_JOB_TYPE) == ["asr"]
    assert _registry_stage_names(PLACEHOLDER_JOB_TYPE) == ["noop", "noop2"]
    assert _registry_stage_names(None) == ["noop", "noop2"]
