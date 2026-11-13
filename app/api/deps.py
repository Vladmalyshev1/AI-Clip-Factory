"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.config import get_settings as get_cached_settings


def get_settings(request: Request) -> Settings:
    """Return the settings bound to this application instance."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_cached_settings()


def get_session_factory(request: Request) -> sessionmaker[Session]:
    """Return the session factory created during application startup."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise RuntimeError("Session factory is not initialised; application startup did not run")
    return factory


def get_db(
    factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> Iterator[Session]:
    """Yield a short-lived session for one request."""
    with factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionFactoryDep = Annotated[sessionmaker[Session], Depends(get_session_factory)]
DbSessionDep = Annotated[Session, Depends(get_db)]
