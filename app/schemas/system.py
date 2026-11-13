"""Response schema for ``GET /system/status``."""

from __future__ import annotations

from pydantic import BaseModel


class FfmpegStatus(BaseModel):
    """FFmpeg binary and feature probe."""

    found: bool
    version: str | None = None
    path: str | None = None
    libass: bool = False
    libx264: bool = False
    nvenc: bool = False


class FfprobeStatus(BaseModel):
    """FFprobe binary probe."""

    found: bool


class WorkerStatus(BaseModel):
    """Worker liveness derived from running job heartbeats."""

    alive: bool
    last_heartbeat_ms: int | None = None


class DiskStatus(BaseModel):
    """Free space on the data volume."""

    free_gb: float
    min_free_gb: float


class OllamaStatus(BaseModel):
    """Optional local LLM reachability."""

    reachable: bool


class SystemStatus(BaseModel):
    """Aggregated readiness report."""

    ready: bool
    ffmpeg: FfmpegStatus
    ffprobe: FfprobeStatus
    worker: WorkerStatus
    disk: DiskStatus
    ollama: OllamaStatus
    data_dir: str
