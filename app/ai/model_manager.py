"""Lazy Whisper model management and the cross-process ASR lock (M3.2).

Two rules shape this module:

* **Nothing heavy is imported at module scope.** ``faster_whisper`` and
  ``ctranslate2`` are imported inside :func:`resolve_model_dir` and
  :func:`_default_model_factory`, so importing ``app.ai.model_manager`` (or any
  module that imports it) never loads CUDA or allocates a model.
* **Models are never downloaded implicitly.** :func:`resolve_model_dir`
  defaults to ``local_files_only=True``; the only path that can download is the
  explicit ``python -m app.ai.model_manager fetch`` CLI. A missing model becomes
  an ``ASR_MODEL_UNAVAILABLE`` error (503) that tells the user exactly which
  command to run.

A single model instance is created per :class:`AsrModelManager` (guarded by a
lock, so concurrent first calls still load once). :class:`AsrProcessLock` adds a
cross-process guard because two worker processes loading Whisper at the same
time would exhaust 16 GB of RAM.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "Systran/faster-whisper-small"
FETCH_INSTRUCTION: Final = "python -m app.ai.model_manager fetch"
LOCK_PID_FILENAME: Final = "pid"
STALE_LOCK_SECONDS: Final = 12 * 3600
MISSING_PID_GRACE_SECONDS: Final = 5.0
DEFAULT_LOCK_POLL_SECONDS: Final = 0.5
MIN_LOCK_POLL_SECONDS: Final = 0.01


def _model_unavailable(
    model: str,
    *,
    local_files_only: bool,
    download_root: Path | None,
    exc: BaseException,
) -> AppError:
    """Build the canonical "model is not on this machine" error."""
    message = (
        f"ASR model '{model}' is not available locally. "
        f"Run '{FETCH_INSTRUCTION} --model {model}' to download it once, "
        "then retry the transcript."
    )
    details: dict[str, Any] = {
        "model": model,
        "local_files_only": local_files_only,
        "download_root": str(download_root) if download_root is not None else None,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    return AppError(
        ErrorCode.ASR_MODEL_UNAVAILABLE,
        message,
        status_code=503,
        details=details,
    )


def _download_model_dir(
    model: str,
    *,
    local_files_only: bool,
    cache_dir: str | None,
) -> str:
    """Call ``faster_whisper.utils.download_model`` (heavy import is local)."""
    from faster_whisper.utils import download_model

    return str(
        download_model(
            model,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    )


def resolve_model_dir(
    model: str,
    *,
    local_files_only: bool = True,
    download_root: Path | None = None,
) -> Path:
    """Return the local directory holding ``model``.

    An existing directory (a converted CTranslate2 model) is returned as-is and
    never touches the network. Otherwise the model id is resolved through the
    Hugging Face cache; with the default ``local_files_only=True`` this is an
    offline lookup. Every failure is mapped to
    ``AppError(ErrorCode.ASR_MODEL_UNAVAILABLE, status_code=503)`` whose message
    contains the explicit fetch command.
    """
    if model.strip():
        candidate = Path(model).expanduser()
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            pass

    cache_dir = str(Path(download_root).expanduser()) if download_root is not None else None
    try:
        resolved = Path(_download_model_dir(
            model,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"Resolved model path is not a directory: {resolved}")
        return resolved.resolve()
    except Exception as exc:  # LocalEntryNotFoundError/OSError/ValueError/offline mode
        raise _model_unavailable(
            model,
            local_files_only=local_files_only,
            download_root=download_root,
            exc=exc,
        ) from exc


def model_available(model: str, *, download_root: Path | None = None) -> bool:
    """Return whether ``model`` resolves locally, without downloading or raising."""
    try:
        resolve_model_dir(model, local_files_only=True, download_root=download_root)
    except AppError:
        return False
    return True


def _default_model_factory(
    model_path: str,
    *,
    device: str,
    compute_type: str,
    cpu_threads: int,
) -> WhisperModel:
    """Construct a ``WhisperModel``; the heavy import happens here."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


class AsrModelManager:
    """Owns at most one loaded Whisper model and resolves it exactly once."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        compute_type: str,
        cpu_threads: int,
        local_files_only: bool = True,
        download_root: Path | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.local_files_only = local_files_only
        self.download_root = Path(download_root) if download_root is not None else None
        self._model_factory = model_factory
        self._model: Any | None = None
        self._resolved_dir: Path | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Configured model id (also what :class:`app.ai.asr` reports)."""
        return self.model

    @property
    def loaded(self) -> bool:
        """Whether a model instance is currently held in memory."""
        return self._model is not None

    def get_model(self) -> Any:
        """Return the single model instance, loading it on first use.

        The lock makes the "first use" race safe: even with eight worker
        threads calling this simultaneously the factory runs exactly once.
        """
        with self._lock:
            if self._model is not None:
                return self._model
            if self._resolved_dir is None:
                self._resolved_dir = resolve_model_dir(
                    self.model,
                    local_files_only=self.local_files_only,
                    download_root=self.download_root,
                )
            factory = self._model_factory or _default_model_factory
            logger.info(
                "Loading ASR model %s from %s (%s/%s)",
                self.model,
                self._resolved_dir,
                self.device,
                self.compute_type,
            )
            self._model = factory(
                str(self._resolved_dir),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            return self._model

    def unload(self) -> None:
        """Drop the model reference so the next :meth:`get_model` reloads it."""
        with self._lock:
            self._model = None

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot for status endpoints and logs."""
        return {
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "local_files_only": self.local_files_only,
            "download_root": str(self.download_root) if self.download_root is not None else None,
            "model_dir": str(self._resolved_dir) if self._resolved_dir is not None else None,
            "loaded": self.loaded,
            "status": "loaded" if self.loaded else "idle",
        }


def build_manager(settings: Settings) -> AsrModelManager:
    """Build a manager from the existing ``asr_*`` application settings."""
    return AsrModelManager(
        model=settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        cpu_threads=settings.asr_cpu_threads,
        local_files_only=settings.asr_local_files_only,
    )


_MANAGER_LOCK: Final = threading.Lock()
_manager: AsrModelManager | None = None


def get_manager(settings: Settings | None = None) -> AsrModelManager:
    """Return the process-wide model manager (created once, thread-safe)."""
    global _manager
    if _manager is not None:
        return _manager
    with _MANAGER_LOCK:
        if _manager is None:
            _manager = build_manager(settings if settings is not None else get_settings())
        return _manager


def reset_manager() -> None:
    """Forget the process-wide manager (test isolation)."""
    global _manager
    with _MANAGER_LOCK:
        _manager = None


class AsrLockTimeout(Exception):  # noqa: N818 - name follows the stage vocabulary
    """Raised when the cross-process ASR lock is still held after a timeout."""

    def __init__(
        self,
        message: str,
        *,
        lock_dir: Path | None = None,
        timeout_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.lock_dir = lock_dir
        self.timeout_s = timeout_s


def _pid_exists(pid: int) -> bool:
    """Return whether ``pid`` is alive; conservative if psutil is missing."""
    if pid <= 0:
        return False
    try:
        psutil = importlib.import_module("psutil")
    except ImportError:  # pragma: no cover - psutil is a runtime dependency
        return True
    try:
        return bool(psutil.pid_exists(pid))
    except Exception:  # pragma: no cover - defensive: never break a live lock
        return True


class AsrProcessLock:
    """Cross-process guard: only one process may own an ASR model at a time.

    The lock is a directory created with atomic ``os.mkdir`` plus a ``pid``
    file. A lock whose pid is gone (or which is older than
    :data:`STALE_LOCK_SECONDS`) is quarantined and removed once, so a crashed
    worker cannot wedge the queue forever.
    """

    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = Path(lock_dir)
        self.pid_path = self.lock_dir / LOCK_PID_FILENAME
        self._acquired = False

    @property
    def acquired(self) -> bool:
        """Whether this instance currently owns the lock."""
        return self._acquired

    def __enter__(self) -> AsrProcessLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(
        self,
        *,
        timeout: float | None = None,
        poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """Take the lock, waiting up to ``timeout`` seconds.

        ``on_wait`` runs once per poll while another process holds the lock; the
        pipeline stage passes its cancellation check there. Raises
        :class:`AsrLockTimeout` when the wait expires.
        """
        if self._acquired:
            return
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        poll = max(MIN_LOCK_POLL_SECONDS, poll_seconds)
        timeout_message = (
            f"Timed out after {timeout}s waiting for the ASR process lock {self.lock_dir}"
        )
        while True:
            if self._try_create():
                self._acquired = True
                logger.debug("Acquired ASR process lock %s", self.lock_dir)
                return
            if self._break_stale() and self._try_create():
                self._acquired = True
                logger.info(
                    "Acquired ASR process lock %s after breaking a stale lock",
                    self.lock_dir,
                )
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise AsrLockTimeout(
                    timeout_message,
                    lock_dir=self.lock_dir,
                    timeout_s=timeout,
                )
            if on_wait is not None:
                on_wait()
            sleep_for = poll
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AsrLockTimeout(
                        timeout_message,
                        lock_dir=self.lock_dir,
                        timeout_s=timeout,
                    )
                sleep_for = min(poll, remaining)
            time.sleep(sleep_for)

    def release(self) -> None:
        """Release the lock; safe to call repeatedly or when never acquired."""
        if not self._acquired:
            return
        self._acquired = False
        try:
            self.pid_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove ASR lock pid file %s: %s", self.pid_path, exc)
        try:
            self.lock_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove ASR lock directory %s: %s", self.lock_dir, exc)
        else:
            logger.debug("Released ASR process lock %s", self.lock_dir)

    def _try_create(self) -> bool:
        """Atomically create the lock directory, then record our pid."""
        try:
            self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(self.lock_dir)
        except FileExistsError:
            return False
        except OSError as exc:
            raise RuntimeError(
                f"Could not create ASR lock directory {self.lock_dir}: {exc}"
            ) from exc
        try:
            self.pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        except OSError:
            with contextlib.suppress(OSError):
                self.lock_dir.rmdir()
            raise
        return True

    def _read_pid(self) -> int | None:
        """Return the recorded holder pid, or ``None`` when missing/corrupt."""
        try:
            first_line = self.pid_path.read_text(encoding="utf-8").splitlines()[0]
            return int(first_line.strip())
        except (OSError, IndexError, ValueError):
            return None

    def _lock_age_s(self) -> float | None:
        try:
            return max(0.0, time.time() - self.lock_dir.stat().st_mtime)
        except OSError:
            return None

    def _break_stale(self) -> bool:
        """Remove a dead or ancient lock; return whether removal was attempted."""
        age = self._lock_age_s()
        if age is None:
            return False
        pid = self._read_pid()
        if age > STALE_LOCK_SECONDS:
            reason = f"older than {STALE_LOCK_SECONDS}s"
        elif pid is None:
            if age < MISSING_PID_GRACE_SECONDS:
                return False
            reason = "has no readable pid file"
        elif not _pid_exists(pid):
            reason = f"holder pid {pid} is gone"
        else:
            return False

        # Re-read right before removing: a fresh owner may have replaced the
        # directory between the stat above and now.
        if self._read_pid() != pid:
            return False
        logger.warning("Breaking stale ASR process lock %s (%s)", self.lock_dir, reason)
        quarantine = self.lock_dir.with_name(
            f"{self.lock_dir.name}.stale-{os.getpid()}-{time.monotonic_ns()}"
        )
        try:
            os.rename(self.lock_dir, quarantine)
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.warning("Could not quarantine stale ASR lock %s: %s", self.lock_dir, exc)
            return False
        shutil.rmtree(quarantine, ignore_errors=True)
        return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ai.model_manager",
        description="Manage the local faster-whisper model cache.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser(
        "fetch",
        help="download a model from the Hugging Face Hub into the local cache",
    )
    fetch.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model id or existing directory (default: {DEFAULT_MODEL})",
    )
    fetch.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help="cache directory (default: the Hugging Face cache)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; ``fetch`` is the only operation that may download."""
    args = _build_parser().parse_args(argv)
    if args.command != "fetch":  # pragma: no cover - argparse enforces choices
        return 2
    try:
        directory = resolve_model_dir(
            args.model,
            local_files_only=False,
            download_root=args.download_root,
        )
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MODEL",
    "FETCH_INSTRUCTION",
    "STALE_LOCK_SECONDS",
    "AsrLockTimeout",
    "AsrModelManager",
    "AsrProcessLock",
    "build_manager",
    "get_manager",
    "main",
    "model_available",
    "reset_manager",
    "resolve_model_dir",
]
