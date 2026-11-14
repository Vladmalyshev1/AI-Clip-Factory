"""SQLAlchemy engine/session factories and SQLite pragma setup.

The application uses the synchronous SQLAlchemy ORM. Sessions are short-lived:
a session is never held open while a pipeline stage runs.
"""

from __future__ import annotations

from typing import Any, Final

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models.base import Base

SQLITE_DRIVER_PREFIX: Final = "sqlite"
SQLITE_PRAGMAS: Final[tuple[tuple[str, str], ...]] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("busy_timeout", "5000"),
    ("foreign_keys", "ON"),
)


def is_sqlite_url(url: str) -> bool:
    """Return whether the URL targets SQLite."""
    return url.split(":", 1)[0].startswith(SQLITE_DRIVER_PREFIX)


def _apply_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        for name, value in SQLITE_PRAGMAS:
            cursor.execute(f"PRAGMA {name}={value}")
    finally:
        cursor.close()


def make_engine(settings: Settings) -> Engine:
    """Create an engine with SQLite pragmas configured for concurrent access."""
    url = settings.database_url or ""
    if not url:
        raise ValueError("Settings.database_url is empty; cannot create an engine")
    options: dict[str, Any] = {}
    if is_sqlite_url(url):
        options["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **options)
    if is_sqlite_url(url):
        event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory that keeps attributes usable after commit."""
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def init_schema(engine: Engine) -> None:
    """Create missing tables from ORM metadata (Alembic stays the migration path)."""
    from app import models as _models  # noqa: F401  (registers every table)

    Base.metadata.create_all(engine)
