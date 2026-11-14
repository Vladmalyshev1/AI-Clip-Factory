"""ffprobe-backed media metadata extraction.

The ingest service stores the result of :func:`probe_media` on the video row so
that the pipeline never has to inspect the same file twice. All times are
integer milliseconds, matching the rest of the schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.media.binaries import find_ffprobe, run_command

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 60.0
MAX_STDERR_CHARS = 2000


class ProbeError(Exception):
    """Raised when a media file cannot be inspected.

    Carries an :class:`ErrorCode` so API callers can map it onto the canonical
    error envelope without guessing.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.PROBE_FAILED,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class MediaInfo:
    """Everything the pipeline needs to know about a source recording."""

    path: str
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    pix_fmt: str | None
    audio_codec: str | None
    has_audio: bool
    container: str
    bitrate: int | None
    size_bytes: int
    rotation: int = 0
    sample_rate: int | None = None
    channels: int | None = None

    @property
    def duration_s(self) -> float:
        """Duration in seconds, for FFmpeg command lines."""
        return self.duration_ms / 1000.0

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height (0.0 when height is unknown)."""
        return self.width / self.height if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        """Whether the source is already taller than it is wide."""
        return self.height > self.width

    @property
    def is_landscape(self) -> bool:
        """Whether the source is wider than it is tall."""
        return self.width > self.height

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON column on the video row."""
        return {
            "path": self.path,
            "duration_ms": self.duration_ms,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "pix_fmt": self.pix_fmt,
            "audio_codec": self.audio_codec,
            "has_audio": self.has_audio,
            "container": self.container,
            "bitrate": self.bitrate,
            "size_bytes": self.size_bytes,
            "rotation": self.rotation,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None


def _parse_rate(value: object) -> float:
    """Parse an ffprobe frame rate such as ``30000/1001`` into a float."""
    if isinstance(value, str) and "/" in value:
        numerator, _, denominator = value.partition("/")
        top = _as_float(numerator)
        bottom = _as_float(denominator)
        if top is not None and bottom:
            return round(top / bottom, 3)
        return 0.0
    parsed = _as_float(value)
    return round(parsed, 3) if parsed is not None else 0.0


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags")
    if isinstance(tags, dict):
        parsed = _as_int(tags.get("rotate"))
        if parsed is not None:
            return parsed % 360
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict):
                parsed = _as_int(item.get("rotation"))
                if parsed is not None:
                    return parsed % 360
    return 0


def _select_video_stream(streams: list[Any]) -> dict[str, Any] | None:
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition")
        if isinstance(disposition, dict) and disposition.get("attached_pic") == 1:
            continue  # cover art, not real video
        return stream
    return None


def _select_audio_stream(streams: list[Any]) -> dict[str, Any] | None:
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def probe_media(
    path: str | Path,
    *,
    ffprobe_path: Path | None = None,
    settings: Settings | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> MediaInfo:
    """Inspect a media file and return its metadata.

    Raises :class:`ProbeError` when the file is missing, unreadable or contains
    no video stream. ``ffprobe_path`` wins over settings-based resolution.
    """
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_file():
        raise ProbeError(
            f"Media file not found: {resolved_path}",
            code=ErrorCode.SOURCE_NOT_FOUND,
            details={"path": str(resolved_path)},
        )

    binary = ffprobe_path if ffprobe_path is not None else find_ffprobe(settings)
    if binary is None:
        raise ProbeError("ffprobe executable not found", code=ErrorCode.FFPROBE_NOT_FOUND)

    result = run_command(
        [
            str(binary),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(resolved_path),
        ],
        timeout=timeout,
    )
    if not result.ok:
        raise ProbeError(
            f"ffprobe failed with exit code {result.returncode}",
            details={"stderr": result.stderr[-MAX_STDERR_CHARS:]},
        )

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(
            "ffprobe returned invalid JSON",
            details={"error": str(exc), "stdout": result.stdout[:MAX_STDERR_CHARS]},
        ) from exc
    if not isinstance(payload, dict):
        raise ProbeError("ffprobe returned an unexpected payload")

    streams = payload.get("streams")
    stream_list = streams if isinstance(streams, list) else []
    video = _select_video_stream(stream_list)
    if video is None:
        raise ProbeError(
            "No video stream found in the file",
            code=ErrorCode.UNSUPPORTED_MEDIA,
            details={"path": str(resolved_path)},
        )
    audio = _select_audio_stream(stream_list)

    raw_format = payload.get("format")
    container_info: dict[str, Any] = raw_format if isinstance(raw_format, dict) else {}

    duration_s = (
        _as_float(container_info.get("duration"))
        or _as_float(video.get("duration"))
        or 0.0
    )
    width = _as_int(video.get("width")) or 0
    height = _as_int(video.get("height")) or 0
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    size_bytes = _as_int(container_info.get("size")) or resolved_path.stat().st_size

    return MediaInfo(
        path=str(resolved_path),
        duration_ms=max(0, round(duration_s * 1000)),
        width=width,
        height=height,
        fps=fps,
        video_codec=str(video.get("codec_name") or ""),
        pix_fmt=str(video["pix_fmt"]) if video.get("pix_fmt") else None,
        audio_codec=str(audio["codec_name"]) if audio and audio.get("codec_name") else None,
        has_audio=audio is not None,
        container=str(container_info.get("format_name") or ""),
        bitrate=_as_int(container_info.get("bit_rate")),
        size_bytes=size_bytes,
        rotation=_rotation(video),
        sample_rate=_as_int(audio.get("sample_rate")) if audio else None,
        channels=_as_int(audio.get("channels")) if audio else None,
    )


__all__ = ["MediaInfo", "ProbeError", "probe_media"]
