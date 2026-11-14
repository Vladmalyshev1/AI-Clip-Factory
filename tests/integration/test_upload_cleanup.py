"""The worker loop owns periodic expiry of abandoned uploads.

Cleanup runs inside ``run_worker`` (never as a separate process), so this test
starts the real worker loop once and checks the database and the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.status import UploadStatus
from app.models.job import new_id
from app.models.upload import Upload
from app.worker import run_worker
from sqlalchemy.orm import Session, sessionmaker

MS_PER_HOUR = 3_600_000


def _insert_stale_upload(
    factory: sessionmaker[Session],
    *,
    temp_path: Path,
    expires_at: int,
    now: int,
) -> str:
    upload_id = new_id()
    with factory() as session:
        session.add(
            Upload(
                id=upload_id,
                filename="abandoned.mp4",
                total_bytes=1024,
                received_bytes=64,
                status=UploadStatus.UPLOADING.value,
                temp_path=str(temp_path),
                sha256=None,
                video_id=None,
                created_at=now - 10_000,
                updated_at=now - 10_000,
                expires_at=expires_at,
            )
        )
        session.commit()
    return upload_id


@pytest.mark.api
def test_worker_expires_stale_uploads_and_sweeps_orphaned_chunks(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    now = utc_now_ms()
    settings.ensure_dirs()
    stale_id = _insert_stale_upload(
        session_factory,
        temp_path=settings.incoming_dir / "abandoned.part",
        expires_at=now - 1000,
        now=now,
    )
    stale_path = settings.incoming_dir / "abandoned.part"
    stale_path.write_bytes(b"partial data")

    orphan = settings.incoming_dir / "crashed-request.chunk"
    orphan.write_bytes(b"scratch data")
    old = (now - 2 * (settings.upload_ttl_hours + 1) * MS_PER_HOUR) / 1000
    os.utime(orphan, (old, old))

    fresh_chunk = settings.incoming_dir / "in-flight.chunk"
    fresh_chunk.write_bytes(b"current request")

    assert run_worker(settings, worker_id="cleanup-test", once=True) == 0

    with session_factory() as session:
        row = session.get(Upload, stale_id)
        assert row is not None
        assert row.status == UploadStatus.EXPIRED.value
    assert not stale_path.exists()
    assert not orphan.exists()
    assert fresh_chunk.is_file(), "a live chunk scratch file must not be swept"
