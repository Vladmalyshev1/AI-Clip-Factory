"""Cancellable FFmpeg process runner with progress reporting.

Only :mod:`app.media.binaries` knows where FFmpeg lives; this module knows how
to drive a long-running encode:

* the command is always an argument **list** (never a shell, never a
  concatenated string), so paths containing ``&``, ``;``, quotes or spaces are
  passed through verbatim;
* ``stderr`` goes to a temporary file instead of a pipe, because an unread
  stderr pipe fills up and deadlocks FFmpeg on long renders;
* ``-progress pipe:1`` emits machine-readable ``key=value`` lines on stdout,
  which are parsed into fractions for the job progress column;
* cancellation and timeouts are enforced from a ~50 ms polling loop that
  terminates, then kills the child and always reaps it.

FFmpeg's ``out_time_us`` and ``out_time_ms`` progress keys both carry
**microseconds** (the ``_ms`` spelling is a historical FFmpeg quirk).
:class:`ProgressUpdate.out_time_ms` keeps FFmpeg's name but stores the
microsecond value.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.media.binaries import creation_flags, require_ffmpeg

logger = logging.getLogger(__name__)

GLOBAL_FLAGS: Final[tuple[str, ...]] = (
    "-hide_banner",
    "-nostdin",
    "-loglevel",
    "error",
    "-progress",
    "pipe:1",
    "-nostats",
)
STDERR_TAIL_BYTES: Final = 8 * 1024
POLL_INTERVAL_S: Final = 0.05
PROGRESS_THROTTLE_S: Final = 0.1
TERMINATE_GRACE_S: Final = 3.0
READER_JOIN_S: Final = 5.0
_TIME_KEYS: Final = frozenset({"out_time_us", "out_time_ms"})


class RenderError(Exception):
    """A render failed, timed out or could not be started.

    Carries the canonical :class:`ErrorCode` plus everything the job record
    needs to explain the failure to the user.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.RENDER_FAILED,
        command: Sequence[str] | None = None,
        returncode: int | None = None,
        stderr_tail: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.command = list(command or [])
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.details = details or {}


class RenderCancelled(Exception):  # noqa: N818 - name follows the pipeline vocabulary
    """Raised when the caller's cancel event fired during a render."""

    def __init__(self, message: str = "Render cancelled", *, command: Sequence[str] | None = None):
        super().__init__(message)
        self.message = message
        self.command = list(command or [])


@dataclass(frozen=True)
class RenderResult:
    """Outcome of a successful FFmpeg run."""

    returncode: int
    elapsed_s: float
    stderr_tail: str
    command: list[str]


@dataclass(frozen=True)
class ProgressUpdate:
    """One parsed ``-progress`` line.

    ``out_time_ms`` is FFmpeg's key name; the stored value is in microseconds.
    ``fraction`` is only meaningful for ``progress=end`` (``1.0``), because a
    pure parser has no access to the expected duration.
    """

    fraction: float
    out_time_ms: int | None
    speed: str | None


def parse_progress_line(line: str) -> ProgressUpdate | None:
    """Parse a single FFmpeg progress line, or return ``None`` for noise."""
    text = line.strip()
    if not text or "=" not in text:
        return None
    key, _, raw_value = text.partition("=")
    key = key.strip()
    value = raw_value.strip()
    if key in _TIME_KEYS:
        try:
            microseconds = int(value)
        except ValueError:
            return None
        if microseconds < 0:
            return None
        return ProgressUpdate(fraction=0.0, out_time_ms=microseconds, speed=None)
    if key == "speed":
        speed = None if value in {"", "N/A"} else value
        return ProgressUpdate(fraction=0.0, out_time_ms=None, speed=speed)
    if key == "progress" and value == "end":
        return ProgressUpdate(fraction=1.0, out_time_ms=None, speed=None)
    return None


def _pump_stdout(process: subprocess.Popen[str], sink: queue.Queue[str]) -> None:
    """Forward FFmpeg's stdout lines into ``sink`` without blocking the poll loop."""
    stream = process.stdout
    if stream is None:  # pragma: no cover - stdout is always piped
        return
    try:
        for line in stream:
            sink.put(line)
    except (OSError, ValueError):  # pragma: no cover - stream closed during shutdown
        return


def _read_stderr_tail(handle: IO[bytes]) -> str:
    """Return the last ``STDERR_TAIL_BYTES`` of the stderr capture file."""
    try:
        handle.flush()
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - STDERR_TAIL_BYTES))
        raw = handle.read()
    except (OSError, ValueError):  # pragma: no cover - defensive
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").rstrip()
    return str(raw).rstrip()  # pragma: no cover - the file is opened in binary mode


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate, then kill and always reap the child process."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=TERMINATE_GRACE_S)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=TERMINATE_GRACE_S)
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - unkillable child
        logger.error("FFmpeg process %s could not be stopped", process.pid)


class FFmpegRunner:
    """Runs one FFmpeg command at a time with progress, cancel and timeout support."""

    def __init__(self, ffmpeg_path: Path, *, log_dir: Path | None = None) -> None:
        self.ffmpeg_path = Path(ffmpeg_path)
        self.log_dir = Path(log_dir) if log_dir is not None else None

    @classmethod
    def from_settings(cls, settings: Settings) -> FFmpegRunner:
        """Build a runner from application settings (pinned/bundled FFmpeg)."""
        return cls(require_ffmpeg(settings), log_dir=settings.logs_dir)

    def build_command(self, args: Sequence[str]) -> list[str]:
        """Return the full argv: binary, global flags, then caller arguments."""
        return [str(self.ffmpeg_path), *GLOBAL_FLAGS, *args]

    def run(
        self,
        args: Sequence[str],
        *,
        duration_ms: int | None = None,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        cancel_check: Callable[[], None] | None = None,
        timeout_s: float | None = None,
        cwd: Path | None = None,
    ) -> RenderResult:
        """Run FFmpeg to completion.

        Raises :class:`RenderCancelled` when ``cancel_event`` fires,
        :class:`RenderError` on timeout or non-zero exit.

        ``cancel_check`` is polled on every loop iteration and may raise any
        exception to abort the run (the pipeline passes a callback that raises
        ``StageCancelled``). It exists because progress-driven cancellation only
        works while FFmpeg keeps emitting progress: a stuck child process would
        otherwise make a job uncancellable. The child is always stopped.
        """
        command = self.build_command(args)
        started = time.monotonic()
        stderr_fd, stderr_path = self._create_stderr_file()
        stderr_handle = os.fdopen(stderr_fd, "w+b")
        progress_queue: queue.Queue[str] = queue.Queue()
        process: subprocess.Popen[str] | None = None
        reader: threading.Thread | None = None
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr_handle,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=str(cwd) if cwd is not None else None,
                    creationflags=creation_flags(),
                )
            except OSError as exc:
                raise RenderError(
                    f"Could not start FFmpeg: {exc}",
                    code=ErrorCode.FFMPEG_NOT_FOUND,
                    command=command,
                    details={"error": str(exc)},
                ) from exc

            reader = threading.Thread(
                target=_pump_stdout,
                args=(process, progress_queue),
                name="ffmpeg-progress",
                daemon=True,
            )
            reader.start()

            emitted_fraction = -1.0
            last_emit = 0.0
            status = "exited"
            while True:
                if cancel_check is not None:
                    cancel_check()
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    break
                if timeout_s is not None and time.monotonic() - started > timeout_s:
                    status = "timeout"
                    break
                try:
                    line: str | None = progress_queue.get_nowait()
                except queue.Empty:
                    line = None
                if line is not None:
                    update = parse_progress_line(line)
                    if (
                        update is not None
                        and update.out_time_ms is not None
                        and duration_ms is not None
                        and duration_ms > 0
                    ):
                        fraction = min(1.0, update.out_time_ms / 1000.0 / duration_ms)
                        if fraction > emitted_fraction and on_progress is not None:
                            now = time.monotonic()
                            if now - last_emit >= PROGRESS_THROTTLE_S or fraction >= 1.0:
                                on_progress(fraction)
                                emitted_fraction = fraction
                                last_emit = now
                    continue
                if process.poll() is not None:
                    break
                time.sleep(POLL_INTERVAL_S)

            elapsed = time.monotonic() - started
            if status == "cancelled":
                _stop_process(process)
                logger.info("FFmpeg render cancelled after %.2fs", elapsed)
                raise RenderCancelled(
                    f"FFmpeg render cancelled after {elapsed:.2f}s", command=command
                )
            if status == "timeout":
                _stop_process(process)
                raise RenderError(
                    f"FFmpeg render timed out after {timeout_s}s",
                    code=ErrorCode.RENDER_TIMEOUT,
                    command=command,
                    returncode=None,
                    details={"timeout_s": timeout_s, "elapsed_s": elapsed},
                )

            returncode = process.wait()
            if reader is not None:
                reader.join(timeout=READER_JOIN_S)
            stderr_tail = _read_stderr_tail(stderr_handle)
            if returncode != 0:
                raise RenderError(
                    f"FFmpeg failed with exit code {returncode}",
                    code=ErrorCode.RENDER_FAILED,
                    command=command,
                    returncode=returncode,
                    stderr_tail=stderr_tail,
                    details={"elapsed_s": elapsed},
                )
            if on_progress is not None and emitted_fraction < 1.0:
                on_progress(1.0)
            logger.debug("FFmpeg finished in %.2fs: %s", elapsed, command)
            return RenderResult(
                returncode=returncode,
                elapsed_s=elapsed,
                stderr_tail=stderr_tail,
                command=command,
            )
        finally:
            if process is not None:
                _stop_process(process)
                if reader is not None:
                    reader.join(timeout=READER_JOIN_S)
                if process.stdout is not None:
                    with contextlib.suppress(OSError):  # pragma: no cover - already closed
                        process.stdout.close()
            with contextlib.suppress(OSError):  # pragma: no cover - already closed
                stderr_handle.close()
            try:
                stderr_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - file locked by a stuck child
                logger.warning("Could not delete FFmpeg stderr capture %s", stderr_path)

    def _create_stderr_file(self) -> tuple[int, Path]:
        """Create the temporary stderr capture file (in ``log_dir`` when usable)."""
        directory: str | None = None
        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                directory = str(self.log_dir)
            except OSError:  # pragma: no cover - read-only log directory
                directory = None
        fd, name = tempfile.mkstemp(prefix="ffmpeg-stderr-", suffix=".log", dir=directory)
        return fd, Path(name)


__all__ = [
    "GLOBAL_FLAGS",
    "FFmpegRunner",
    "ProgressUpdate",
    "RenderCancelled",
    "RenderError",
    "RenderResult",
    "parse_progress_line",
]
