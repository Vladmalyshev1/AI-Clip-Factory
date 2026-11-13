"""Time helpers.

Every persisted timestamp is an integer count of milliseconds since the Unix
epoch in UTC. Naive ``datetime`` objects are never stored.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

MS_PER_SECOND = 1000


def utc_now_ms() -> int:
    """Return the current UTC time as integer epoch milliseconds."""
    return time.time_ns() // 1_000_000


def seconds_to_ms(seconds: float) -> int:
    """Convert a duration in seconds to whole milliseconds."""
    return int(seconds * MS_PER_SECOND)


def ms_to_iso(value: int | None) -> str | None:
    """Render epoch milliseconds as an ISO-8601 UTC string (logging only)."""
    if value is None:
        return None
    return datetime.fromtimestamp(value / MS_PER_SECOND, tz=UTC).isoformat()
