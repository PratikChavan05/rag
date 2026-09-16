"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.schemas import HealthResponse
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — no auth required."""
    settings = get_settings()
    return HealthResponse(
        ok=True,
        version=__version__,
        provider=settings.ai_provider,
    )
