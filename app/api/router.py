"""API router: every versioned route is mounted here."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, jobs, system

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(jobs.router)

__all__ = ["api_router"]
