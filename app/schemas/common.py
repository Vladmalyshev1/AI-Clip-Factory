"""Schemas shared by several endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Machine-readable error payload."""

    code: str
    message: str
    details: Any | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    """Canonical error response: ``{"error": {...}}``."""

    error: ErrorBody


class Page[ItemT](BaseModel):
    """Offset/limit pagination envelope."""

    items: list[ItemT] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
