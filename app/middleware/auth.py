from __future__ import annotations
import secrets
from fastapi import Depends, HTTPException, Request
from app.config import Settings, get_settings

async def verify_internal_token(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate X-Internal-Token header. Raises 401 on mismatch."""
    token = request.headers.get("X-Internal-Token", "")
    if not token or not secrets.compare_digest(token, settings.internal_token):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")
