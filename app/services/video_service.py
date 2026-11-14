"""Ingest of source recordings: import, metadata probe and thumbnails.

A video row is created first (``pending``), then probed. A probe failure keeps
the file and the row (``failed`` plus the error), because a recording is user
data and must never be deleted by ingest. All timestamps are epoch milliseconds.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path, PurePosixPath
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.clock import utc_now_ms
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, storage_error
from app.core.paths import safe_name
from app.core.status import SourceKind, VideoStatus
from app.media.binaries import require_ffmpeg, run_command
from app.media.probe import ProbeError, probe_media
from app.models.job import new_id
from app.models.video import Video

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".flv", ".ts", ".avi"}
)
DISK_HEADROOM_BYTES: Final[int] = 512 * 1024**2
HASH_BLOCK_BYTES: Final[int] = 8 * 1024**2
THUMBNAIL_WIDTH: Final[int] = 480
THUMBNAIL_MAX_SEEK_SECONDS: Final[float] = 30.0
THUMBNAIL_TIMEOUT_SECONDS: Final[float] = 60.0
MAX_ERROR_CHARS: Final[int] = 2000
DEFAULT_TITLE: Final[str] = "Untitled recording"


def media_basename(filename: str) -> str:
    """Return the last path component of a client-supplied filename.

    The value is only ever used as a display name and to derive an extension;
    every real path component is an opaque id, so a hostile filename can never
    escape a directory.
    """
    normalized = filename.replace("\\", "/").strip()
    name = PurePosixPath(normalized).name.strip()
    return name or "upload"


def media_extension(filename: str) -> str:
    """Return the lower-cased extension of a filename (``""`` when absent)."""
    return PurePosixPath(media_basename(filename)).suffix.lower()


def ensure_supported_extension(filename: str) -> str:
    """Return the extension or raise ``UNSUPPORTED_MEDIA`` (415)."""
    extension = media_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise AppError(
            ErrorCode.UNSUPPORTED_MEDIA,
            f"Unsupported media extension '{extension or 'none'}'; "
            f"allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            status_code=415,
            details={
                "filename": media_basename(filename),
                "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            },
        )
    return extension


def default_title(filename: str) -> str:
    """Derive a readable title from a filename."""
    basename = media_basename(filename)
    stem = PurePosixPath(basename).stem.strip()
    if stem:
        return stem
    return safe_name(basename, fallback=DEFAULT_TITLE)


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the volume holding ``path`` (nearest existing ancestor)."""
    return int(shutil.disk_usage(_existing_ancestor(path)).free)


def ensure_disk_headroom(settings: Settings, required_bytes: int) -> None:
    """Raise ``INSUFFICIENT_DISK`` (507) when the volume is too full."""
    required = max(0, int(required_bytes)) + DISK_HEADROOM_BYTES
    free = free_disk_bytes(settings.data_dir)
    if free < required:
        raise AppError(
            ErrorCode.INSUFFICIENT_DISK,
            "Not enough free disk space for this recording",
            status_code=507,
            details={"free_bytes": free, "required_bytes": required},
        )


def sha256_file(path: Path, *, block_size: int = HASH_BLOCK_BYTES) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def get_video(session: Session, video_id: str) -> Video:
    """Load a video or raise ``VIDEO_NOT_FOUND`` (404)."""
    video = session.get(Video, video_id)
    if video is None:
        raise AppError(
            ErrorCode.VIDEO_NOT_FOUND, f"Video '{video_id}' not found", status_code=404
        )
    return video


def list_videos(
    session: Session,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Video], int]:
    """Return a page of videos (newest first) and the total matching count."""
    filters = []
    if status is not None:
        filters.append(Video.status == status)
    count_stmt = select(func.count()).select_from(Video)
    page_stmt = select(Video).order_by(Video.created_at.desc(), Video.id)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
        page_stmt = page_stmt.where(condition)
    total = session.scalar(count_stmt) or 0
    items = list(session.scalars(page_stmt.limit(limit).offset(offset)).all())
    return items, int(total)


def create_video(
    session: Session,
    *,
    title: str,
    original_filename: str,
    source_path: str | Path,
    source_kind: SourceKind | str,
    size_bytes: int,
    sha256: str | None = None,
    video_id: str | None = None,
    now_ms: int | None = None,
) -> Video:
    """Insert a ``pending`` video row and flush it (caller commits).

    ``video_id`` lets upload completion reuse the identifier that already names
    the stored original, so ``uploads.video_id`` always resolves to this row.
    """
    now = now_ms if now_ms is not None else utc_now_ms()
    video = Video(
        id=video_id if video_id is not None else new_id(),
        title=title,
        original_filename=original_filename,
        source_path=str(source_path),
        source_kind=SourceKind(source_kind).value,
        size_bytes=int(size_bytes),
        sha256=sha256,
        status=VideoStatus.PENDING.value,
        created_at=now,
        updated_at=now,
    )
    session.add(video)
    session.flush()
    return video


def finalize_video(
    settings: Settings,
    session_factory: sessionmaker[Session],
    video_id: str,
) -> Video:
    """Probe the stored file and persist the outcome on the video row.

    The probe runs without a DB session; only the short update holds one. A
    probe failure never deletes the file: the row is marked ``failed`` with the
    error so the UI can explain what is wrong.
    """
    with session_factory() as session:
        source_path = get_video(session, video_id).source_path

    try:
        info = probe_media(source_path, settings=settings)
    except ProbeError as exc:
        logger.warning("Probe failed for video %s: %s", video_id, exc.message)
        with session_factory() as session:
            video = get_video(session, video_id)
            video.status = VideoStatus.FAILED.value
            video.error_code = exc.code.value
            video.error_message = exc.message[:MAX_ERROR_CHARS]
            video.updated_at = utc_now_ms()
            session.commit()
            return video

    with session_factory() as session:
        video = get_video(session, video_id)
        video.duration_ms = info.duration_ms
        video.width = info.width
        video.height = info.height
        video.fps = info.fps
        video.video_codec = info.video_codec or None
        video.audio_codec = info.audio_codec
        video.has_audio = 1 if info.has_audio else 0
        video.container = info.container or None
        video.bitrate = info.bitrate
        video.size_bytes = info.size_bytes
        video.status = VideoStatus.READY.value
        video.error_code = None
        video.error_message = None
        video.updated_at = utc_now_ms()
        session.commit()
        return video


def _is_inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:  # pragma: no cover - unresolvable path on Windows
        return False


def allowed_import_roots(settings: Settings) -> list[Path]:
    """Return the directories ``POST /videos/import`` may read from.

    Configured through ``ACF_IMPORT_ROOTS`` (default: the user's home). Paths
    are expanded and resolved once per call, so a relative entry is interpreted
    against the process working directory. An empty list denies every
    server-side import; uploads are unaffected.
    """
    roots: list[Path] = []
    for raw in settings.import_roots:
        try:
            roots.append(Path(raw).expanduser().resolve())
        except OSError:  # pragma: no cover - unresolvable path
            logger.warning("Ignoring unresolvable import root %s", raw)
    return roots


def ensure_import_allowed(settings: Settings, resolved: Path) -> None:
    """Reject a server-side import path outside every configured root."""
    roots = allowed_import_roots(settings)
    if any(_is_inside(resolved, root) for root in roots):
        return
    raise AppError(
        ErrorCode.VALIDATION_ERROR,
        "source path is outside the allowed import directories",
        status_code=403,
        details={
            "path": str(resolved),
            "allowed_roots": [str(root) for root in roots],
        },
    )


def import_video(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    path: str,
    copy: bool = True,
    title: str | None = None,
) -> Video:
    """Ingest an existing recording from the local filesystem.

    ``copy=True`` copies the file into ``media/originals`` (``import``) and may
    store a SHA-256 when the file fits the upload limit; ``copy=False`` only
    references the absolute path (``import_in_place``) and leaves ``sha256``
    NULL, because hashing multi-gigabyte in-place files is not worth it at
    ingest time. In-place files are never opened for writing afterwards.
    """
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise AppError(
            ErrorCode.SOURCE_NOT_FOUND,
            "Source path must be absolute",
            status_code=404,
            details={"path": path},
        )
    resolved = source.resolve()
    if not resolved.is_file():
        raise AppError(
            ErrorCode.SOURCE_NOT_FOUND,
            f"Source file not found: {resolved}",
            status_code=404,
            details={"path": str(resolved)},
        )
    if _is_inside(resolved, settings.data_dir):
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            "source must be outside the application data directory",
            status_code=422,
            details={"path": str(resolved), "data_dir": str(settings.data_dir)},
        )
    ensure_import_allowed(settings, resolved)
    extension = ensure_supported_extension(resolved.name)
    resolved_title = (title or "").strip() or default_title(resolved.name)
    size_bytes = resolved.stat().st_size

    if copy:
        ensure_disk_headroom(settings, size_bytes)
        settings.ensure_dirs()
        destination = settings.originals_dir / f"{new_id()}{extension}"
        try:
            shutil.copyfile(resolved, destination)
        except OSError as exc:
            # A failed copy (including a full disk) may leave a partial file.
            # ``destination`` is a fresh UUID path, never the user's original.
            try:
                destination.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - locked file on Windows
                logger.warning("Could not remove partial copy %s", destination)
            raise storage_error(exc, path=destination) from exc
        source_path: Path = destination
        source_kind = SourceKind.IMPORT
        digest = sha256_file(destination) if size_bytes <= settings.max_upload_bytes else None
    else:
        source_path = resolved
        source_kind = SourceKind.IMPORT_IN_PLACE
        digest = None

    with session_factory() as session:
        video = create_video(
            session,
            title=resolved_title,
            original_filename=resolved.name,
            source_path=source_path,
            source_kind=source_kind,
            size_bytes=size_bytes,
            sha256=digest,
        )
        video_id = video.id
        session.commit()

    return finalize_video(settings, session_factory, video_id)


def thumbnail_path(settings: Settings, session: Session, video_id: str) -> Path:
    """Return a cached JPEG thumbnail, generating it on first use."""
    video = get_video(session, video_id)
    if video.status != VideoStatus.READY.value:
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Video '{video_id}' is {video.status}; thumbnails require a ready video",
            status_code=409,
            details={"status": video.status},
        )

    thumbnail = settings.thumbs_dir / f"{video.id}.jpg"
    if thumbnail.is_file() and thumbnail.stat().st_size > 0:
        return thumbnail

    settings.ensure_dirs()
    ffmpeg = require_ffmpeg(settings)
    duration_s = (video.duration_ms or 0) / 1000.0
    seek_seconds = min(duration_s * 0.1, THUMBNAIL_MAX_SEEK_SECONDS)
    result = run_command(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{seek_seconds:.3f}",
            "-i",
            video.source_path,
            "-frames:v",
            "1",
            "-vf",
            f"scale={THUMBNAIL_WIDTH}:-2",
            "-q:v",
            "4",
            str(thumbnail),
        ],
        timeout=THUMBNAIL_TIMEOUT_SECONDS,
    )
    if not result.ok or not thumbnail.is_file() or thumbnail.stat().st_size == 0:
        raise AppError(
            ErrorCode.RENDER_FAILED,
            "FFmpeg could not generate a thumbnail",
            status_code=500,
            details={"stderr": result.stderr[-MAX_ERROR_CHARS:]},
        )
    return thumbnail


__all__ = [
    "ALLOWED_EXTENSIONS",
    "DISK_HEADROOM_BYTES",
    "allowed_import_roots",
    "create_video",
    "default_title",
    "ensure_disk_headroom",
    "ensure_import_allowed",
    "ensure_supported_extension",
    "finalize_video",
    "free_disk_bytes",
    "get_video",
    "import_video",
    "list_videos",
    "media_basename",
    "media_extension",
    "sha256_file",
    "thumbnail_path",
]
