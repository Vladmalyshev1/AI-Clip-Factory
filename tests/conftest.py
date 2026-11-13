"""Shared pytest configuration and fixtures.

The project is intentionally not installed as a distribution yet, so the
repository root and the tools directory are added to sys.path explicitly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "tools"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from app.core.config import Settings  # noqa: E402
from app.core.db import init_schema, make_engine, make_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402

WaitUntil = Callable[..., Any]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing at a temporary data directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "var",
        log_level="WARNING",
        worker_poll_seconds=0.05,
        worker_heartbeat_seconds=0.2,
        job_stale_seconds=5.0,
    )


@pytest.fixture()
def engine(settings: Settings) -> Iterator[Engine]:
    """Engine with schema created on the isolated database file."""
    settings.ensure_dirs()
    engine = make_engine(settings)
    init_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the test engine."""
    return make_session_factory(engine)


@pytest.fixture()
def app(settings: Settings) -> FastAPI:
    """Application instance using the isolated settings."""
    return create_app(settings)


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    """Test client with lifespan (schema creation) executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def wait_until() -> Callable[..., Any]:
    """Poll ``predicate`` until it returns truthy or the timeout expires."""

    def _wait(predicate: Callable[[], Any], *, timeout: float = 5.0, interval: float = 0.02) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(interval)
        return predicate()

    return _wait


def _worker_env(settings: Settings, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["ACF_DATA_DIR"] = str(settings.data_dir)
    env["ACF_LOG_LEVEL"] = "WARNING"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


@pytest.fixture()
def run_worker(
    settings: Settings,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run ``python -m app.worker`` to completion in a subprocess."""

    def _run(
        *args: str,
        timeout: float = 60.0,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "app.worker", *args],
            cwd=ROOT,
            env=_worker_env(settings, env_extra),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    return _run


@pytest.fixture()
def spawn_worker(
    settings: Settings,
) -> Iterator[Callable[..., subprocess.Popen[str]]]:
    """Start ``python -m app.worker`` in the background; clean it up afterwards."""
    processes: list[subprocess.Popen[str]] = []

    def _spawn(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [sys.executable, "-m", "app.worker", *args],
            cwd=ROOT,
            env=_worker_env(settings, env_extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        processes.append(process)
        return process

    yield _spawn

    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn process
                process.kill()
                process.wait(timeout=10)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
