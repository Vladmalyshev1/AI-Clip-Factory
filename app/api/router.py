"""API router: every versioned route is mounted here."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, highlights, jobs, system, transcripts, uploads, videos
from app.api.routes.clips import clip_router as generated_clip_router
from app.api.routes.clips import video_router as clip_video_router
from app.api.routes.editor import clip_router as export_clip_router
from app.api.routes.editor import editor_router, export_router, source_router

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(jobs.router)
api_router.include_router(uploads.router)
api_router.include_router(videos.router)
api_router.include_router(transcripts.router)
api_router.include_router(highlights.router)
api_router.include_router(source_router)
api_router.include_router(clip_video_router)
# The export routes under /clips are mounted next to the generated-clip routes.
api_router.include_router(export_clip_router)
api_router.include_router(generated_clip_router)
api_router.include_router(editor_router)
api_router.include_router(export_router)

__all__ = ["api_router"]
