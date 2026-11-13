"""Background worker: ``python -m app.worker``.

Loop: recover stale jobs, claim the next queued job, run it, repeat. State lives
in SQLite, so the worker can be killed and restarted without losing progress.
A graceful stop (Ctrl+C, SIGTERM) returns the running job to the queue with its
stage checkpoints intact; only an explicit API cancellation cancels a job.

Usage::

    python -m app.worker [--once] [--poll-seconds N] [--worker-id ID] [--log-level LEVEL]
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
from collections.abc import Sequence
from types import FrameType
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.clock import utc_now_ms
from app.core.config import Settings, get_settings
from app.core.db import init_schema, make_engine, make_session_factory
from app.core.errors import ErrorCode
from app.core.logging import configure_logging
from app.core.status import JobStatus
from app.pipeline.orchestrator import run_job
from app.services import job_service

logger = logging.getLogger(__name__)

MIN_POLL_SECONDS = 0.01
MIN_HEARTBEAT_SECONDS = 0.1
HEARTBEAT_JOIN_TIMEOUT_S = 5.0


class WorkerHeartbeat:
    """Background heartbeat for the job currently owned by this worker."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        *,
        worker_id: str,
        interval_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        self._factory = factory
        self._worker_id = worker_id
        self._interval = max(MIN_HEARTBEAT_SECONDS, interval_seconds)
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._job_id: str | None = None
        self._thread = threading.Thread(target=self._run, name="acf-heartbeat", daemon=True)

    def start(self) -> None:
        """Start the heartbeat thread."""
        self._thread.start()

    def set_job(self, job_id: str | None) -> None:
        """Point the heartbeat at a job (or clear it)."""
        with self._lock:
            self._job_id = job_id

    def stop(self) -> None:
        """Stop the heartbeat thread and wait for it briefly."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=HEARTBEAT_JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            with self._lock:
                job_id = self._job_id
            if job_id is None:
                continue
            try:
                with self._factory() as session:
                    job_service.heartbeat(session, job_id, self._worker_id)
            except Exception:  # pragma: no cover - transient SQLite contention
                logger.exception("Heartbeat failed for job %s", job_id)


def _install_signal_handlers(stop_event: threading.Event) -> dict[int, Any]:
    """Convert SIGINT/SIGTERM into a clean shutdown request."""
    previous: dict[int, Any] = {}

    def _handle(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %s; stopping worker", signum)
        stop_event.set()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:  # pragma: no cover - not available on this platform
            continue
        try:
            previous[sig] = signal.signal(sig, _handle)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            continue
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # pragma: no cover - defensive
            continue


def default_worker_id() -> str:
    """Return a stable-ish identifier for this worker process."""
    host = socket.gethostname().split(".")[0] or "host"
    return f"worker-{host}-{os.getpid()}"


def _recover_stale(engine: Engine, settings: Settings) -> list[str]:
    try:
        return job_service.recover_stale(engine, settings.job_stale_seconds, utc_now_ms())
    except Exception:  # pragma: no cover - keep the loop alive
        logger.exception("Stale job recovery failed")
        return []


def _fail_job_after_error(
    factory: sessionmaker[Session], job_id: str, error_message: str
) -> None:
    try:
        with factory() as session:
            job_service.mark_job_failed(
                session,
                job_id,
                error_code=ErrorCode.WORKER_ERROR.value,
                error_message=error_message,
            )
            session.commit()
    except Exception:  # pragma: no cover - best effort
        logger.exception("Could not mark job %s as failed", job_id)


def run_worker(
    settings: Settings,
    *,
    worker_id: str,
    once: bool = False,
    stop_event: threading.Event | None = None,
) -> int:
    """Run the claim/process loop until stopped (or once when ``once`` is set)."""
    engine = make_engine(settings)
    init_schema(engine)
    factory = make_session_factory(engine)
    stopping = stop_event if stop_event is not None else threading.Event()

    heartbeat = WorkerHeartbeat(
        factory,
        worker_id=worker_id,
        interval_seconds=settings.worker_heartbeat_seconds,
        stop_event=threading.Event(),
    )
    heartbeat.start()
    previous_handlers = _install_signal_handlers(stopping)
    poll_seconds = max(MIN_POLL_SECONDS, settings.worker_poll_seconds)

    try:
        _recover_stale(engine, settings)
        while not stopping.is_set():
            claimed = job_service.claim_next(engine, worker_id)
            if claimed is None:
                if once:
                    break
                stopping.wait(poll_seconds)
                _recover_stale(engine, settings)
                continue

            heartbeat.set_job(claimed.job_id)
            print(
                f"claimed job {claimed.job_id} "
                f"(attempt {claimed.attempt}/{claimed.max_attempts})",
                flush=True,
            )
            try:
                final_status = run_job(
                    engine, settings, claimed, worker_id, stop_event=stopping
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Unhandled error while running job %s", claimed.job_id)
                _fail_job_after_error(
                    factory,
                    claimed.job_id,
                    f"Worker error: {type(exc).__name__}: {exc}",
                )
                final_status_value = "failed"
            else:
                final_status_value = final_status.value
            finally:
                heartbeat.set_job(None)
            if final_status_value == JobStatus.QUEUED.value:
                print(
                    f"interrupted job {claimed.job_id}: returned to the queue",
                    flush=True,
                )
            else:
                print(f"finished job {claimed.job_id}: {final_status_value}", flush=True)
            if once:
                break
    finally:
        heartbeat.stop()
        _restore_signal_handlers(previous_handlers)
        engine.dispose()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.worker",
        description="Process queued AI Clip Factory jobs.",
    )
    parser.add_argument(
        "--once", action="store_true", help="process at most one job, then exit"
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="override ACF_WORKER_POLL_SECONDS",
    )
    parser.add_argument("--worker-id", default=None, help="worker identifier for claims")
    parser.add_argument("--log-level", default=None, help="override ACF_LOG_LEVEL")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides on top of environment settings."""
    updates: dict[str, Any] = {}
    if args.poll_seconds is not None:
        updates["worker_poll_seconds"] = max(MIN_POLL_SECONDS, float(args.poll_seconds))
    if args.log_level:
        updates["log_level"] = str(args.log_level)
    settings = get_settings()
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m app.worker``."""
    args = build_parser().parse_args(argv)
    settings = build_settings(args)
    configure_logging(settings.log_level, settings.logs_dir)
    settings.ensure_dirs()
    worker_id = args.worker_id or default_worker_id()
    logger.info("Worker %s starting (poll=%ss)", worker_id, settings.worker_poll_seconds)
    exit_code = run_worker(settings, worker_id=worker_id, once=bool(args.once))
    logger.info("Worker %s stopped", worker_id)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
