"""System readiness probes: ffmpeg/ffprobe, worker liveness, disk and Ollama."""

from __future__ import annotations

import logging
import shutil
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import JobStatus
from app.media.binaries import DEFAULT_TIMEOUT_SECONDS, find_ffmpeg, find_ffprobe, run_command
from app.models.job import Job
from app.schemas.system import (
    DiskStatus,
    FfmpegStatus,
    FfprobeStatus,
    OllamaStatus,
    SystemStatus,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
PROBE_CACHE_TTL_SECONDS = 30.0
FFMPEG_VERSION_PATTERN = r"ffmpeg version (n?[\w.\-]+)"
BYTES_PER_GB = 1024**3

_probe_cache: dict[str, tuple[float, FfmpegStatus]] = {}
_probe_lock = threading.Lock()



def _parse_version(output: str) -> str | None:
    first_line = output.splitlines()[0] if output else ""
    marker = "ffmpeg version "
    if marker in first_line:
        return first_line.split(marker, 1)[1].split()[0]
    return None


def _probe_ffmpeg(path: Path) -> FfmpegStatus:
    version_result = run_command([str(path), "-hide_banner", "-version"])
    build_result = run_command([str(path), "-hide_banner", "-buildconf"])
    encoder_result = run_command([str(path), "-hide_banner", "-encoders"])
    if version_result.returncode != 0:
        return FfmpegStatus(found=False, path=str(path))
    build_info = build_result.output
    encoders = encoder_result.output
    return FfmpegStatus(
        found=True,
        version=_parse_version(version_result.output),
        path=str(path),
        libass="--enable-libass" in build_info,
        libx264="libx264" in encoders,
        nvenc="h264_nvenc" in encoders,
    )


def probe_ffmpeg(path: Path) -> FfmpegStatus:
    """Probe ffmpeg and cache the result briefly to keep status cheap."""
    key = str(path)
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(key)
        if cached is not None and now - cached[0] < PROBE_CACHE_TTL_SECONDS:
            return cached[1]
    probe = _probe_ffmpeg(path)
    with _probe_lock:
        _probe_cache[key] = (now, probe)
    return probe


def probe_ffprobe(path: Path | None) -> FfprobeStatus:
    """Check that ffprobe exists and executes."""
    if path is None:
        return FfprobeStatus(found=False)
    result = run_command([str(path), "-hide_banner", "-version"])
    return FfprobeStatus(found=result.returncode == 0)


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_status(settings: Settings) -> DiskStatus:
    """Report free space on the volume that holds ``data_dir``."""
    usage = shutil.disk_usage(_existing_ancestor(settings.data_dir))
    return DiskStatus(
        free_gb=round(usage.free / BYTES_PER_GB, 2),
        min_free_gb=settings.min_free_disk_gb,
    )


def worker_status(
    session_factory: sessionmaker[Session], settings: Settings, *, now_ms: int | None = None
) -> WorkerStatus:
    """Derive worker liveness from the newest heartbeat of running jobs."""
    now = now_ms if now_ms is not None else utc_now_ms()
    with session_factory() as session:
        last_heartbeat = session.scalar(
            select(func.max(Job.heartbeat_at)).where(
                Job.status.in_((JobStatus.RUNNING.value, JobStatus.CANCEL_REQUESTED.value))
            )
        )
    if last_heartbeat is None:
        return WorkerStatus(alive=False, last_heartbeat_ms=None)
    stale_ms = int(max(0.0, settings.job_stale_seconds) * 1000)
    return WorkerStatus(alive=now - int(last_heartbeat) <= stale_ms, last_heartbeat_ms=int(last_heartbeat))


def ollama_status(settings: Settings) -> OllamaStatus:
    """Check whether Ollama answers on its port (optional dependency)."""
    parsed = urlparse(settings.ollama_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    timeout = max(0.05, settings.ollama_timeout_seconds)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return OllamaStatus(reachable=True)
    except OSError:
        return OllamaStatus(reachable=False)


def get_system_status(settings: Settings, session_factory: sessionmaker[Session]) -> SystemStatus:
    """Collect the full readiness report (always succeeds, never raises)."""
    ffmpeg_path = find_ffmpeg(settings)
    ffmpeg = (
        probe_ffmpeg(ffmpeg_path) if ffmpeg_path is not None else FfmpegStatus(found=False)
    )
    ffprobe = probe_ffprobe(find_ffprobe(settings, ffmpeg_path=ffmpeg_path))
    ollama = ollama_status(settings)
    ready = ffmpeg.found and ffmpeg.libass and ffprobe.found
    return SystemStatus(
        ready=ready,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        worker=worker_status(session_factory, settings),
        disk=disk_status(settings),
        ollama=ollama,
        data_dir=str(settings.data_dir),
    )
