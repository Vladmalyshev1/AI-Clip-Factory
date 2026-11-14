"""Resumable chunked uploads: session lifecycle, streaming writes, completion.

The body of a chunk request is never read into memory: it is streamed into a
private scratch file, ``received_bytes`` is advanced with a compare-and-set
``UPDATE``, and only the request that won the claim splices its bytes into the
preallocated ``.part`` file. Two writers can therefore never interleave.
A database session is never held open while the body is being received.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, Any, Final, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, storage_error
from app.core.status import SourceKind, UploadStatus
from app.models.job import new_id
from app.models.upload import Upload
from app.models.video import Video
from app.services import video_service

logger = logging.getLogger(__name__)

STREAM_SUBCHUNK_BYTES: Final[int] = 1024 * 1024
MS_PER_HOUR: Final[int] = 3_600_000
MAX_ERROR_CHARS: Final[int] = 2000
CHUNK_SUFFIX: Final = ".chunk"

#: Serialises completion inside this process so a duplicate ``POST /complete``
#: waits for the winner and returns its video (HTTP 200) instead of colliding
#: with the ``.part`` file. Cross-process safety comes from the compare-and-set
#: on the upload row. Completions are rare and already I/O bound, so one lock
#: for the process is enough and cannot leak per-upload state.
_COMPLETION_LOCK: Final = threading.Lock()

#: Guards the short ``.part`` operations (fsync and the atomic move) against a
#: concurrent chunk splice. On Windows an open handle blocks ``os.replace`` with
#: ``WinError 32``; serialising them inside the process removes that collision.
_PARTOPS_LOCK: Final = threading.Lock()


def get_upload(session: Session, upload_id: str) -> Upload:
    """Load an upload or raise ``UPLOAD_NOT_FOUND`` (404)."""
    upload = session.get(Upload, upload_id)
    if upload is None:
        raise AppError(
            ErrorCode.UPLOAD_NOT_FOUND, f"Upload '{upload_id}' not found", status_code=404
        )
    return upload


def _expiry_ms(settings: Settings, now_ms: int) -> int:
    return now_ms + int(max(0.0, settings.upload_ttl_hours) * MS_PER_HOUR)


def _remove_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - locked file on Windows
        logger.warning("Could not remove upload temp file %s: %s", path, exc)


def _sweep_orphan_chunks(settings: Settings, now_ms: int) -> int:
    """Delete chunk scratch files abandoned by a crashed request.

    They are per-request temporaries under ``incoming``, never received data:
    a live request owns one for at most a few seconds, so the TTL is a safe
    margin. Files belonging to an in-flight chunk keep their recent mtime.
    """
    cutoff_ms = now_ms - int(max(0.0, settings.upload_ttl_hours) * MS_PER_HOUR)
    try:
        candidates = list(settings.incoming_dir.glob(f"*{CHUNK_SUFFIX}"))
    except OSError:  # pragma: no cover - directory missing or unreadable
        return 0
    removed = 0
    for path in candidates:
        try:
            if path.stat().st_mtime * 1000 <= cutoff_ms:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:  # pragma: no cover - locked by a live request
            logger.warning("Could not remove orphaned chunk file %s", path)
    return removed


def _chunk_temp_file(settings: Settings, upload_id: str) -> tuple[IO[bytes], Path]:
    """Open a private scratch file for one received chunk.

    The shared ``.part`` file is never written while a request body is being
    received: two requests racing for the same offset would interleave their
    bytes. Each request streams into its own file and only the one that wins the
    compare-and-set splices it into ``.part``.
    """
    settings.ensure_dirs()
    try:
        fd, name = tempfile.mkstemp(
            prefix=f"{upload_id}-", suffix=CHUNK_SUFFIX, dir=settings.incoming_dir
        )
        return os.fdopen(fd, "wb"), Path(name)
    except OSError as exc:
        raise storage_error(exc, path=settings.incoming_dir) from exc


def _splice_chunk(part_path: Path, offset: int, chunk_path: Path) -> None:
    """Copy a received chunk into the partial file at ``offset``."""
    with _PARTOPS_LOCK, open(chunk_path, "rb") as source, open(part_path, "r+b") as destination:
        destination.seek(offset)
        while block := source.read(STREAM_SUBCHUNK_BYTES):
            destination.write(block)
        destination.flush()


def _release_chunk_claim(
    session_factory: sessionmaker[Session], upload_id: str, claimed: int, expected: int
) -> None:
    """Undo a chunk claim whose bytes could not be stored."""
    try:
        with session_factory() as session:
            cast(
                CursorResult[Any],
                session.execute(
                    update(Upload)
                    .where(
                        Upload.id == upload_id,
                        Upload.status == UploadStatus.UPLOADING.value,
                        Upload.received_bytes == claimed,
                    )
                    .values(received_bytes=expected, updated_at=utc_now_ms())
                ),
            )
            session.commit()
    except Exception:  # pragma: no cover - best effort during shutdown
        logger.exception("Could not roll back the chunk claim for upload %s", upload_id)


def expire_stale_uploads(settings: Settings, session: Session, now_ms: int) -> int:
    """Mark abandoned sessions ``expired`` and delete their partial files.

    Only ``.part`` files under ``incoming`` are removed: they are incomplete
    temporary data, never a user recording. The caller commits.
    """
    stale = list(
        session.scalars(
            select(Upload).where(
                Upload.status == UploadStatus.UPLOADING.value,
                Upload.expires_at < now_ms,
            )
        ).all()
    )
    for upload in stale:
        upload.status = UploadStatus.EXPIRED.value
        upload.updated_at = now_ms
        _remove_temp_file(Path(upload.temp_path))
    if stale:
        session.flush()
        logger.info("Expired %d stale upload session(s)", len(stale))
    orphans = _sweep_orphan_chunks(settings, now_ms)
    if orphans:
        logger.info("Removed %d orphaned chunk file(s)", orphans)
    return len(stale)


def create_upload(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    filename: str,
    total_bytes: int,
) -> Upload:
    """Create an upload session and preallocate its target file."""
    display_name = video_service.media_basename(filename)
    video_service.ensure_supported_extension(display_name)
    if total_bytes > settings.max_upload_bytes:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "file exceeds the configured upload limit",
            status_code=413,
            details={
                "max_upload_bytes": settings.max_upload_bytes,
                "total_bytes": total_bytes,
            },
        )
    video_service.ensure_disk_headroom(settings, total_bytes)

    settings.ensure_dirs()
    now = utc_now_ms()
    upload_id = new_id()
    temp_path = settings.incoming_dir / f"{upload_id}.part"

    with session_factory() as session:
        expire_stale_uploads(settings, session, now)
        # Reserve the space up front so every offset is a valid file position.
        try:
            with open(temp_path, "wb") as handle:
                handle.truncate(total_bytes)
        except OSError as exc:
            raise storage_error(exc, path=temp_path) from exc
        upload = Upload(
            id=upload_id,
            filename=display_name,
            total_bytes=total_bytes,
            received_bytes=0,
            status=UploadStatus.UPLOADING.value,
            temp_path=str(temp_path),
            sha256=None,
            video_id=None,
            created_at=now,
            updated_at=now,
            expires_at=_expiry_ms(settings, now),
        )
        session.add(upload)
        session.commit()
        logger.info(
            "Created upload %s for %s (%d bytes)", upload_id, display_name, total_bytes
        )
        return upload


def cancel_upload(
    settings: Settings, session_factory: sessionmaker[Session], upload_id: str
) -> None:
    """Cancel an in-flight upload and delete its partial file."""
    with session_factory() as session:
        upload = get_upload(session, upload_id)
        status = UploadStatus(upload.status)
        if status is UploadStatus.COMPLETED:
            raise AppError(
                ErrorCode.UPLOAD_CONFLICT,
                f"Upload '{upload_id}' is already completed",
                status_code=409,
                details={"status": status.value, "video_id": upload.video_id},
            )
        if status is UploadStatus.UPLOADING:
            upload.status = UploadStatus.CANCELED.value
            upload.updated_at = utc_now_ms()
            session.commit()
        temp_path = Path(upload.temp_path)
    _remove_temp_file(temp_path)


async def receive_chunk(
    settings: Settings,
    session_factory: sessionmaker[Session],
    upload_id: str,
    *,
    offset: int,
    content_length: int | None,
    stream: AsyncIterator[bytes],
) -> Upload:
    """Append one chunk to the partial file and advance ``received_bytes``.

    Four phases keep memory and lock time small: a short read, a session-free
    stream into a private scratch file, a guarded compare-and-set that claims the
    byte range, then a splice of that range into ``.part``. Only the request that
    wins the compare-and-set writes to the shared file, so concurrent retries of
    the same offset can never interleave their bytes.
    """
    now = utc_now_ms()
    with session_factory() as session:
        upload = get_upload(session, upload_id)
        status = UploadStatus(upload.status)
        if status in (UploadStatus.CANCELED, UploadStatus.EXPIRED):
            raise AppError(
                ErrorCode.UPLOAD_EXPIRED,
                f"Upload '{upload_id}' is {status.value} and can no longer receive data",
                status_code=410,
                details={"status": status.value},
            )
        if status is UploadStatus.COMPLETED:
            raise AppError(
                ErrorCode.UPLOAD_CONFLICT,
                f"Upload '{upload_id}' is already completed",
                status_code=409,
                details={"status": status.value, "video_id": upload.video_id},
            )
        if upload.expires_at < now:
            upload.status = UploadStatus.EXPIRED.value
            upload.updated_at = now
            session.commit()
            _remove_temp_file(Path(upload.temp_path))
            raise AppError(
                ErrorCode.UPLOAD_EXPIRED,
                f"Upload '{upload_id}' has expired",
                status_code=410,
                details={"expires_at_ms": upload.expires_at},
            )
        expected = upload.received_bytes
        total_bytes = upload.total_bytes
        temp_path = Path(upload.temp_path)

    if offset != expected:
        raise AppError(
            ErrorCode.UPLOAD_CONFLICT,
            "offset does not match the number of bytes already received",
            status_code=409,
            details={"expected_offset": expected, "received_bytes": expected},
        )

    chunk_size = settings.upload_chunk_size
    if content_length is not None and content_length > chunk_size:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            f"chunk exceeds the maximum size of {chunk_size} bytes",
            status_code=413,
            details={"chunk_size": chunk_size, "content_length": content_length},
        )
    if content_length is not None and expected + content_length > total_bytes:
        raise AppError(
            ErrorCode.UPLOAD_CONFLICT,
            "chunk would exceed the declared upload size",
            status_code=409,
            details={
                "received_bytes": expected + content_length,
                "total_bytes": total_bytes,
            },
        )

    if not temp_path.is_file():
        raise AppError(
            ErrorCode.UPLOAD_EXPIRED,
            "upload data is no longer available; start the upload again",
            status_code=410,
            details={"upload_id": upload_id},
        )

    handle, chunk_path = _chunk_temp_file(settings, upload_id)
    written = 0
    try:
        try:
            async for block in stream:
                if not block:
                    continue
                if written + len(block) > chunk_size:
                    raise AppError(
                        ErrorCode.VALIDATION_ERROR,
                        f"chunk exceeds the maximum size of {chunk_size} bytes",
                        status_code=413,
                        details={"chunk_size": chunk_size, "received": written},
                    )
                if expected + written + len(block) > total_bytes:
                    raise AppError(
                        ErrorCode.UPLOAD_CONFLICT,
                        "chunk would exceed the declared upload size",
                        status_code=409,
                        details={
                            "received_bytes": expected + written + len(block),
                            "total_bytes": total_bytes,
                        },
                    )
                view = memoryview(block)
                for start in range(0, len(view), STREAM_SUBCHUNK_BYTES):
                    handle.write(view[start : start + STREAM_SUBCHUNK_BYTES])
                written += len(block)
            handle.flush()
        except OSError as exc:
            raise storage_error(exc, path=chunk_path) from exc
        finally:
            handle.close()

        if written == 0:
            raise AppError(
                ErrorCode.VALIDATION_ERROR,
                "chunk body must not be empty",
                status_code=422,
                details={"chunk_size": chunk_size},
            )

        # Claim the byte range before touching the shared file: at most one
        # writer (in this process or another) can advance ``received_bytes``.
        with session_factory() as session:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(Upload)
                    .where(
                        Upload.id == upload_id,
                        Upload.received_bytes == expected,
                        Upload.status == UploadStatus.UPLOADING.value,
                    )
                    .values(received_bytes=expected + written, updated_at=utc_now_ms())
                ),
            )
            if result.rowcount == 0:
                session.rollback()
                raise AppError(
                    ErrorCode.UPLOAD_CONFLICT,
                    "upload changed while the chunk was being received",
                    status_code=409,
                    details={"expected_offset": expected, "received_bytes": written},
                )
            session.commit()

        try:
            _splice_chunk(temp_path, expected, chunk_path)
        except FileNotFoundError as exc:
            _release_chunk_claim(session_factory, upload_id, expected + written, expected)
            raise AppError(
                ErrorCode.UPLOAD_EXPIRED,
                "upload data is no longer available; start the upload again",
                status_code=410,
                details={"upload_id": upload_id},
            ) from exc
        except OSError as exc:
            _release_chunk_claim(session_factory, upload_id, expected + written, expected)
            raise storage_error(exc, path=temp_path) from exc
    finally:
        _remove_temp_file(chunk_path)

    with session_factory() as session:
        return get_upload(session, upload_id)


def complete_upload(
    settings: Settings, session_factory: sessionmaker[Session], upload_id: str
) -> tuple[Video, bool]:
    """Finalise an upload and return ``(video, created)``.

    ``created`` is ``False`` when the upload was already completed: the call is
    idempotent and returns the existing video instead of failing. Completions in
    this process are serialised by ``_COMPLETION_LOCK``, so a duplicate request
    waits for the winner and then takes the idempotent branch; a duplicate from
    another process is detected by the compare-and-set on the upload row (and by
    the ``.part`` file disappearing).
    """
    with _COMPLETION_LOCK:
        return _complete_upload_locked(settings, session_factory, upload_id)


def _completed_video(
    session_factory: sessionmaker[Session], upload_id: str
) -> tuple[Video, bool] | None:
    """Return the existing video when another caller completed the upload."""
    with session_factory() as session:
        upload = session.get(Upload, upload_id)
        if (
            upload is not None
            and upload.status == UploadStatus.COMPLETED.value
            and upload.video_id
        ):
            return video_service.get_video(session, upload.video_id), False
    return None


def _restore_part_file(destination: Path, temp_path: Path) -> None:
    """Move a stored original back to ``.part`` so completion can be retried.

    Used when the move succeeded but persisting the result failed: the upload
    row is still ``uploading`` with all bytes received, so keeping the data
    avoids forcing the user to upload the recording again.
    """
    try:
        if destination.is_file():
            os.replace(destination, temp_path)
    except OSError as exc:  # pragma: no cover - locked file on Windows
        logger.warning("Could not restore %s after a failed completion: %s", temp_path, exc)


def _complete_upload_locked(
    settings: Settings, session_factory: sessionmaker[Session], upload_id: str
) -> tuple[Video, bool]:
    with session_factory() as session:
        upload = get_upload(session, upload_id)
        status = UploadStatus(upload.status)
        if status is UploadStatus.COMPLETED:
            if upload.video_id:
                return video_service.get_video(session, upload.video_id), False
            raise AppError(
                ErrorCode.UPLOAD_CONFLICT,
                f"Upload '{upload_id}' is completed but has no video",
                status_code=409,
                details={"status": status.value},
            )
        if status in (UploadStatus.CANCELED, UploadStatus.EXPIRED):
            raise AppError(
                ErrorCode.UPLOAD_EXPIRED,
                f"Upload '{upload_id}' is {status.value}",
                status_code=410,
                details={"status": status.value},
            )
        if upload.received_bytes != upload.total_bytes:
            raise AppError(
                ErrorCode.UPLOAD_INCOMPLETE,
                "upload is not fully received",
                status_code=409,
                details={
                    "received_bytes": upload.received_bytes,
                    "total_bytes": upload.total_bytes,
                },
            )
        expected = upload.received_bytes
        total_bytes = upload.total_bytes
        filename = upload.filename
        temp_path = Path(upload.temp_path)

    if not temp_path.is_file():
        completed = _completed_video(session_factory, upload_id)
        if completed is not None:
            return completed
        raise AppError(
            ErrorCode.UPLOAD_INCOMPLETE,
            "upload temp file is missing",
            status_code=409,
            details={"upload_id": upload_id},
        )

    extension = video_service.ensure_supported_extension(filename)
    video_id = new_id()
    destination = settings.originals_dir / f"{video_id}{extension}"

    try:
        settings.ensure_dirs()
    except OSError as exc:
        raise storage_error(exc, path=settings.originals_dir) from exc

    try:
        digest = video_service.sha256_file(temp_path)
        with _PARTOPS_LOCK:
            # The handle must be closed before the move: on Windows an open
            # handle blocks ``os.replace`` with ``WinError 32``.
            with open(temp_path, "rb+") as handle:
                os.fsync(handle.fileno())
            # Same volume: the move is atomic and the original is never rewritten.
            os.replace(temp_path, destination)
    except FileNotFoundError as exc:
        # Another process may have completed the upload between our read and
        # this move: the ``.part`` file is gone because that caller moved it.
        completed = _completed_video(session_factory, upload_id)
        if completed is not None:
            return completed
        raise AppError(
            ErrorCode.UPLOAD_INCOMPLETE,
            "upload data is no longer available; start the upload again",
            status_code=409,
            details={"upload_id": upload_id},
        ) from exc
    except OSError as exc:
        raise storage_error(exc, path=temp_path) from exc

    with session_factory() as session:
        result = cast(
            CursorResult[Any],
            session.execute(
                update(Upload)
                .where(
                    Upload.id == upload_id,
                    Upload.status == UploadStatus.UPLOADING.value,
                    Upload.received_bytes == expected,
                )
                .values(
                    status=UploadStatus.COMPLETED.value,
                    sha256=digest,
                    video_id=video_id,
                    updated_at=utc_now_ms(),
                )
            ),
        )
        if result.rowcount == 0:
            session.rollback()
            # We hold the only copy of the data, so a lost claim means the row
            # was expired or canceled concurrently: drop the fresh file.
            _remove_temp_file(destination)
            existing = session.get(Upload, upload_id)
            if existing is not None and existing.video_id:
                logger.warning(
                    "Upload %s was completed concurrently; keeping %s",
                    upload_id,
                    existing.video_id,
                )
                return video_service.get_video(session, existing.video_id), False
            raise AppError(
                ErrorCode.UPLOAD_CONFLICT,
                "upload changed while it was being completed",
                status_code=409,
                details={"upload_id": upload_id},
            )
        video = video_service.create_video(
            session,
            title=video_service.default_title(filename),
            original_filename=filename,
            source_path=destination,
            source_kind=SourceKind.UPLOAD,
            size_bytes=total_bytes,
            sha256=digest,
            video_id=video_id,
        )
        try:
            session.commit()
        except Exception:
            session.rollback()
            _restore_part_file(destination, temp_path)
            raise
        stored_id = video.id

    logger.info(
        "Upload %s completed as video %s (%d bytes)", upload_id, stored_id, total_bytes
    )
    return video_service.finalize_video(settings, session_factory, stored_id), True


__all__ = [
    "cancel_upload",
    "complete_upload",
    "create_upload",
    "expire_stale_uploads",
    "get_upload",
    "receive_chunk",
]
