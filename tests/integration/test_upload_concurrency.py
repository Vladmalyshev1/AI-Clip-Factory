"""Concurrency: duplicate chunk offsets and duplicate upload completion.

These are service-level tests on purpose: ``TestClient`` serialises requests
through a portal, so it cannot reproduce two threads inside the same request
handler. Real database rows and real files are used; nothing is mocked.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from functools import partial
from pathlib import Path

import pytest
from app.core.config import Settings
from app.core.errors import AppError
from app.models.upload import Upload
from app.models.video import Video
from app.services import upload_service
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

PAYLOAD_BYTES = 1024


async def _stream(payload: bytes) -> AsyncIterator[bytes]:
    yield payload


def _receive(
    settings: Settings,
    factory: sessionmaker[Session],
    upload_id: str,
    payload: bytes,
    *,
    offset: int = 0,
) -> None:
    """Drive ``receive_chunk`` once from synchronous test code."""

    async def run() -> None:
        await upload_service.receive_chunk(
            settings,
            factory,
            upload_id,
            offset=offset,
            content_length=len(payload),
            stream=_stream(payload),
        )

    asyncio.run(run())


def _run_threads(targets: list[Callable[[], object]], *, timeout: float = 30.0) -> None:
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "worker thread did not finish in time"


@pytest.mark.api
def test_parallel_chunks_at_the_same_offset_have_a_single_winner(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    first_payload = b"A" * PAYLOAD_BYTES
    second_payload = b"B" * PAYLOAD_BYTES
    upload = upload_service.create_upload(
        settings, session_factory, filename="race.mp4", total_bytes=PAYLOAD_BYTES
    )
    part_path = Path(upload.temp_path)
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []

    def worker(payload: bytes) -> None:
        barrier.wait()
        try:
            _receive(settings, session_factory, upload.id, payload)
        except AppError as exc:
            outcomes.append(("error", exc.code.value))
        else:
            outcomes.append(("ok", payload))

    _run_threads(
        [partial(worker, payload) for payload in (first_payload, second_payload)]
    )

    winners = [value for kind, value in outcomes if kind == "ok"]
    errors = [value for kind, value in outcomes if kind == "error"]
    assert len(winners) == 1, outcomes
    assert errors == ["UPLOAD_CONFLICT"], outcomes
    # The stored range is exactly one payload: the loser never wrote to ``.part``.
    assert part_path.read_bytes() == winners[0]

    with session_factory() as session:
        row = session.get(Upload, upload.id)
        assert row is not None
        assert row.received_bytes == PAYLOAD_BYTES
        assert row.status == "uploading"


@pytest.mark.api
def test_parallel_complete_returns_one_video_and_stays_idempotent(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    data = b"this is not a real video, only bytes"
    upload = upload_service.create_upload(
        settings, session_factory, filename="double.mp4", total_bytes=len(data)
    )
    _receive(settings, session_factory, upload.id, data)
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []

    def worker() -> None:
        barrier.wait()
        try:
            video, created = upload_service.complete_upload(
                settings, session_factory, upload.id
            )
        except Exception as exc:
            outcomes.append(("error", f"{type(exc).__name__}: {exc}"))
        else:
            outcomes.append(("ok", (video.id, created)))

    _run_threads([worker, worker])

    assert [kind for kind, _ in outcomes] == ["ok", "ok"], outcomes
    values = [value for _, value in outcomes]
    video_ids = {video_id for video_id, _ in values}
    assert len(video_ids) == 1, outcomes
    assert sorted(created for _, created in values) == [False, True], outcomes

    with session_factory() as session:
        videos = list(session.scalars(select(Video)).all())
        assert len(videos) == 1
        assert videos[0].id == video_ids.pop()
        upload_row = session.get(Upload, upload.id)
        assert upload_row is not None
        assert upload_row.status == "completed"
        assert upload_row.video_id == videos[0].id

    assert not Path(upload.temp_path).exists()
    stored = list(settings.originals_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].read_bytes() == data
