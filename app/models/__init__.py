"""Importing this package registers every ORM model on ``Base.metadata``."""

from __future__ import annotations

from app.models.base import Base
from app.models.clip import Clip
from app.models.highlight import Highlight, HighlightCandidate
from app.models.job import Job, JobEvent, JobStage, new_id
from app.models.transcript import Transcript, TranscriptChunk, TranscriptSegment
from app.models.upload import Upload
from app.models.video import Video

__all__ = [
    "Base",
    "Clip",
    "Highlight",
    "HighlightCandidate",
    "Job",
    "JobEvent",
    "JobStage",
    "Transcript",
    "TranscriptChunk",
    "TranscriptSegment",
    "Upload",
    "Video",
    "new_id",
]
