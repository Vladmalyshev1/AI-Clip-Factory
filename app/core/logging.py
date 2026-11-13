"""UTF-8 safe logging configuration.

A cp866 console must never raise on non-ASCII log messages, therefore the
stdout stream is reconfigured (when possible) before handlers are installed.
"""

from __future__ import annotations

import logging
import sys
from contextlib import suppress
from pathlib import Path
from typing import TextIO

_HANDLER_MARKER = "_acf_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_FILENAME = "app.log"


class RequestIDLogFilter(logging.Filter):
    """Inject the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _current_request_id()  # type: ignore[attr-defined]
        return True


def _current_request_id() -> str:
    from app.core.request_id import current_request_id

    return current_request_id() or "-"


def _utf8_stream() -> TextIO:
    """Return stdout with UTF-8 encoding, tolerating wrappers without reconfigure."""
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        with suppress(ValueError, OSError, AttributeError):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    return stream


def _tag_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARKER, True)  # marker for our own handlers
    return handler


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    *,
    filename: str = _DEFAULT_FILENAME,
) -> None:
    """Install stdout (and optionally file) logging with UTF-8 output."""
    resolved_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    root = logging.getLogger()

    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream_handler = _tag_handler(logging.StreamHandler(_utf8_stream()))
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RequestIDLogFilter())
    root.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = _tag_handler(
            logging.FileHandler(log_dir / filename, encoding="utf-8", delay=True)
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIDLogFilter())
        root.addHandler(file_handler)

    root.setLevel(resolved_level)
