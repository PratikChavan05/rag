from __future__ import annotations
import logging
from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings

logger = logging.getLogger("ragdms.db")

_client: AsyncIOMotorClient | None = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongo_uri)
        logger.info("MongoDB client created (uri=%s)", settings.mongo_uri.split("@")[-1])
    return _client

def get_database() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return get_client()[settings.mongo_db]

async def ping_db() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        logger.exception("MongoDB ping failed")
        return False
