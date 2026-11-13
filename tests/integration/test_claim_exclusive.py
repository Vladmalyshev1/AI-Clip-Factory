"""Two workers must never claim the same queued job."""

from __future__ import annotations

import threading
import time

import pytest
from app.core.config import Settings
from app.core.db import make_engine
from app.services import job_service
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


@pytest.mark.windows_only
def test_two_workers_claim_distinct_jobs(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        job_ids = [
            job_service.enqueue(session, params={"noop_duration_s": 0.0}).id for _ in range(6)
        ]
        session.commit()

    claimed_ids: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def drain(worker_id: str) -> None:
        engine: Engine = make_engine(settings)
        try:
            barrier.wait(timeout=10)
            while True:
                claimed = job_service.claim_next(engine, worker_id)
                if claimed is None:
                    return
                time.sleep(0.005)
                with lock:
                    claimed_ids.append(claimed.job_id)
        except BaseException as exc:  # pragma: no cover - surfaced below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    threads = [threading.Thread(target=drain, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(claimed_ids) == sorted(job_ids)
    assert len(claimed_ids) == len(set(claimed_ids))


def _race_for_claim(
    settings: Settings,
    worker_id: str,
    barrier: threading.Barrier,
    lock: threading.Lock,
    results: list[str | None],
) -> None:
    engine = make_engine(settings)
    try:
        barrier.wait(timeout=10)
        claimed = job_service.claim_next(engine, worker_id)
    finally:
        engine.dispose()
    with lock:
        results.append(claimed.job_id if claimed is not None else None)


@pytest.mark.windows_only
def test_single_job_is_claimed_exactly_once(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    for _ in range(5):
        with session_factory() as session:
            job = job_service.enqueue(session, params={"noop_duration_s": 0.0})
            job_id = job.id
            session.commit()

        results: list[str | None] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        threads = [
            threading.Thread(
                target=_race_for_claim,
                args=(settings, f"racer-{index}", barrier, lock, results),
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        winners = [job_id_result for job_id_result in results if job_id_result is not None]
        assert winners == [job_id]
