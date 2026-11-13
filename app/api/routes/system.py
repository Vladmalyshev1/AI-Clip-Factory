"""System readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import SessionFactoryDep, SettingsDep
from app.schemas.system import SystemStatus
from app.services.system_service import get_system_status

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
def system_status(settings: SettingsDep, session_factory: SessionFactoryDep) -> SystemStatus:
    """Return a readable status report even when the environment is degraded."""
    return get_system_status(settings, session_factory)
