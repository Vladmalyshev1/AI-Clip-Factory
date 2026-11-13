"""Importing this package registers every ORM model on ``Base.metadata``."""

from __future__ import annotations

from app.models.base import Base
from app.models.job import Job, JobEvent, JobStage, new_id

__all__ = ["Base", "Job", "JobEvent", "JobStage", "new_id"]
