"""Application assembly only: settings, middleware, error handlers, routers, lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.db import init_schema, make_engine, make_session_factory
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.request_id import REQUEST_ID_HEADER, RequestIDMiddleware

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application for the given settings."""
    resolved = settings if settings is not None else get_settings()
    engine = make_engine(resolved)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved.log_level, resolved.logs_dir)
        resolved.ensure_dirs()
        init_schema(engine)
        try:
            yield
        finally:
            engine.dispose()

    application = FastAPI(
        title="AI Clip Factory API",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.engine = engine
    application.state.session_factory = make_session_factory(engine)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    application.add_middleware(RequestIDMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix=API_PREFIX)
    return application


app = create_app()
