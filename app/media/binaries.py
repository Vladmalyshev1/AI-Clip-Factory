"""Resolution of the FFmpeg toolchain and a safe short-command runner.

Both the API process (readiness probes) and the worker (media stages) must use
the same binaries. Resolution order is identical everywhere:

1. an explicit ``ACF_FFMPEG_PATH`` setting,
2. the bundled ``tools/ffmpeg/bin`` directory (pinned build, see ADR 0001),
3. whatever ``ffmpeg`` is on ``PATH``.

``ffprobe`` is resolved next to the resolved ``ffmpeg`` first, because a mixed
toolchain (different builds on PATH) produces surprising results.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BUNDLED_BIN_DIR: Final = REPO_ROOT / "tools" / "ffmpeg" / "bin"
DEFAULT_TIMEOUT_SECONDS: Final = 15.0


def executable_name(base: str) -> str:
    """Return the platform-specific executable file name."""
    return f"{base}.exe" if sys.platform == "win32" else base


def creation_flags() -> int:
    """Hide console windows for child processes on Windows."""
    if sys.platform != "win32":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a short probe command; never raises on failure."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Return whether the command exited successfully."""
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Combined output, as the readiness probes want it."""
        return f"{self.stdout}{self.stderr}"


def run_command(
    command: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CommandResult:
    """Run a short command and capture both streams without raising."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creation_flags(),
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(returncode=127, stdout="", stderr="executable not found")
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124, stdout="", stderr=f"timed out after {timeout}s")
    except OSError as exc:  # pragma: no cover - platform specific
        return CommandResult(returncode=126, stdout="", stderr=str(exc))
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _resolve(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def find_ffmpeg(settings: Settings | None = None) -> Path | None:
    """Locate the ffmpeg executable, or return ``None`` when absent."""
    resolved = _resolve(settings)
    if resolved.ffmpeg_path:
        candidate = Path(resolved.ffmpeg_path).expanduser()
        if candidate.is_file():
            return candidate
    bundled = BUNDLED_BIN_DIR / executable_name("ffmpeg")
    if bundled.is_file():
        return bundled
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def find_ffprobe(settings: Settings | None = None, *, ffmpeg_path: Path | None = None) -> Path | None:
    """Locate ffprobe, preferring the directory of the resolved ffmpeg."""
    resolved = _resolve(settings)
    exe = executable_name("ffprobe")
    candidates: list[Path] = []
    if ffmpeg_path is not None:
        candidates.append(ffmpeg_path.with_name(exe))
    if resolved.ffmpeg_path:
        candidates.append(Path(resolved.ffmpeg_path).expanduser().with_name(exe))
    candidates.append(BUNDLED_BIN_DIR / exe)
    found = shutil.which("ffprobe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def require_ffmpeg(settings: Settings | None = None) -> Path:
    """Return the ffmpeg path or raise ``FFMPEG_NOT_FOUND``."""
    path = find_ffmpeg(settings)
    if path is None:
        raise AppError(
            ErrorCode.FFMPEG_NOT_FOUND,
            "FFmpeg executable not found; run tools/setup.ps1 or set ACF_FFMPEG_PATH",
            status_code=500,
        )
    return path


def require_ffprobe(settings: Settings | None = None, *, ffmpeg_path: Path | None = None) -> Path:
    """Return the ffprobe path or raise ``FFPROBE_NOT_FOUND``."""
    path = find_ffprobe(settings, ffmpeg_path=ffmpeg_path)
    if path is None:
        raise AppError(
            ErrorCode.FFPROBE_NOT_FOUND,
            "ffprobe executable not found; it must ship next to ffmpeg",
            status_code=500,
        )
    return path


__all__ = [
    "BUNDLED_BIN_DIR",
    "DEFAULT_TIMEOUT_SECONDS",
    "REPO_ROOT",
    "CommandResult",
    "creation_flags",
    "executable_name",
    "find_ffmpeg",
    "find_ffprobe",
    "require_ffmpeg",
    "require_ffprobe",
    "run_command",
]
