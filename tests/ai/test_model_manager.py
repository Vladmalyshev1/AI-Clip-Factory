"""Tests for the lazy model manager, cross-process lock and fetch CLI (M3.2)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from app.ai.model_manager import (
    DEFAULT_MODEL,
    AsrLockTimeout,
    AsrModelManager,
    AsrProcessLock,
    build_manager,
    get_manager,
    model_available,
    reset_manager,
    resolve_model_dir,
)
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode

ROOT = Path(__file__).resolve().parents[2]
MISSING_MODEL = "acf/definitely-not-a-cached-model-xyz"


class _FakeModel:
    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs


def _factory(delay: float = 0.0) -> tuple[Callable[..., _FakeModel], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(path: str, **kwargs: Any) -> _FakeModel:
        calls.append({"path": path, **kwargs})
        if delay:
            time.sleep(delay)
        return _FakeModel(path, **kwargs)

    return create, calls


def _manager(
    model_dir: Path,
    factory: Callable[..., Any],
    **overrides: Any,
) -> AsrModelManager:
    params: dict[str, Any] = {
        "model": str(model_dir),
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 6,
        **overrides,
    }
    return AsrModelManager(model_factory=factory, **params)


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=30)
    return process.pid


@pytest.fixture(autouse=True)
def _reset_singleton() -> Iterator[None]:
    reset_manager()
    yield
    reset_manager()


@pytest.mark.unit
def test_get_model_calls_the_factory_once_for_many_calls(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    factory, calls = _factory()
    manager = _manager(model_dir, factory)

    models = [manager.get_model() for _ in range(20)]

    assert len(calls) == 1
    assert len({id(model) for model in models}) == 1
    assert manager.loaded is True
    assert calls[0] == {
        "path": str(model_dir.resolve()),
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 6,
    }


@pytest.mark.unit
def test_get_model_loads_once_under_concurrent_calls(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    factory, calls = _factory(delay=0.05)
    manager = _manager(model_dir, factory)

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: manager.get_model(), range(8)))

    assert len(calls) == 1
    assert len({id(model) for model in models}) == 1


@pytest.mark.unit
def test_unload_then_get_model_reloads(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    factory, calls = _factory()
    manager = _manager(model_dir, factory)

    first = manager.get_model()
    manager.unload()
    assert manager.loaded is False
    second = manager.get_model()

    assert len(calls) == 2
    assert second is not first
    assert manager.loaded is True


@pytest.mark.unit
def test_describe_reports_configuration_and_state(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    factory, _calls = _factory()
    manager = _manager(model_dir, factory, device="cuda", compute_type="float16")

    idle = manager.describe()
    assert idle["model"] == str(model_dir)
    assert idle["device"] == "cuda"
    assert idle["compute_type"] == "float16"
    assert idle["cpu_threads"] == 6
    assert idle["local_files_only"] is True
    assert idle["loaded"] is False
    assert idle["status"] == "idle"
    assert idle["model_dir"] is None

    manager.get_model()
    loaded = manager.describe()
    assert loaded["loaded"] is True
    assert loaded["status"] == "loaded"
    assert loaded["model_dir"] == str(model_dir.resolve())


@pytest.mark.unit
def test_resolve_model_dir_returns_an_existing_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "converted-model"
    model_dir.mkdir()

    assert resolve_model_dir(str(model_dir)) == model_dir.resolve()


@pytest.mark.unit
def test_missing_model_maps_to_asr_model_unavailable(tmp_path: Path) -> None:
    with pytest.raises(AppError) as caught:
        resolve_model_dir(MISSING_MODEL, local_files_only=True)

    error = caught.value
    assert error.code is ErrorCode.ASR_MODEL_UNAVAILABLE
    assert error.status_code == 503
    assert "python -m app.ai.model_manager fetch" in error.message
    assert error.details["model"] == MISSING_MODEL
    assert error.details["local_files_only"] is True

    assert model_available(MISSING_MODEL) is False

    # A local directory is "available" without consulting the Hugging Face
    # cache, which keeps this test hermetic on a machine that never fetched a
    # model (the real model is exercised by the opt-in smoke test only).
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    assert model_available(str(local_model)) is True
    assert DEFAULT_MODEL, "the default model id must not be empty"


@pytest.mark.unit
def test_build_manager_uses_the_asr_settings() -> None:
    settings = Settings(
        _env_file=None,
        asr_model="small.en",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_cpu_threads=4,
        asr_local_files_only=False,
    )

    manager = build_manager(settings)

    assert manager.model == "small.en"
    assert manager.device == "cuda"
    assert manager.compute_type == "float16"
    assert manager.cpu_threads == 4
    assert manager.local_files_only is False


@pytest.mark.unit
def test_get_manager_is_a_process_wide_singleton(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, asr_model="small.en")
    first = get_manager(settings)

    assert get_manager(settings) is first

    reset_manager()
    assert get_manager(settings) is not first


@pytest.mark.unit
def test_lock_breaks_a_stale_dead_pid_and_releases_idempotently(tmp_path: Path) -> None:
    lock_dir = tmp_path / "asr.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{_dead_pid()}\n", encoding="utf-8")
    lock = AsrProcessLock(lock_dir)

    lock.acquire(timeout=2.0, poll_seconds=0.01)

    assert lock.acquired is True
    assert int((lock_dir / "pid").read_text(encoding="utf-8")) == os.getpid()

    lock.release()
    lock.release()
    assert not lock_dir.exists()


@pytest.mark.unit
def test_lock_breaks_a_lock_older_than_the_stale_threshold(tmp_path: Path) -> None:
    lock_dir = tmp_path / "asr.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    old = time.time() - 13 * 3600
    os.utime(lock_dir, (old, old))

    with AsrProcessLock(lock_dir) as lock:
        assert lock.acquired is True

    assert not lock_dir.exists()


@pytest.mark.unit
def test_lock_times_out_on_a_live_lock_and_polls(tmp_path: Path) -> None:
    lock_dir = tmp_path / "asr.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    waits: list[int] = []

    with pytest.raises(AsrLockTimeout) as caught:
        AsrProcessLock(lock_dir).acquire(
            timeout=0.15,
            poll_seconds=0.01,
            on_wait=lambda: waits.append(1),
        )

    assert caught.value.lock_dir == lock_dir
    assert caught.value.timeout_s == 0.15
    assert len(waits) >= 2
    assert lock_dir.exists()


@pytest.mark.unit
def test_lock_waiter_acquires_after_the_holder_releases(tmp_path: Path) -> None:
    lock_dir = tmp_path / "asr.lock"
    holder = AsrProcessLock(lock_dir)
    holder.acquire()

    def release_soon() -> None:
        time.sleep(0.05)
        holder.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(release_soon)
        AsrProcessLock(lock_dir).acquire(timeout=2.0, poll_seconds=0.01)
        future.result(timeout=5)

    assert lock_dir.exists()


@pytest.mark.unit
def test_cli_help_lists_the_fetch_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "app.ai.model_manager", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fetch" in result.stdout
