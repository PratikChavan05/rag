from __future__ import annotations
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import __version__
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging_config import configure_logging
from app.api.routes import health, jobs, rag

logger = logging.getLogger("ragdms.main")

def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="ragDMS", version=__version__, description="AI service for Smart Cloud DMS")
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Internal-Token"],
        allow_credentials=False,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(rag.router)

    @app.on_event("startup")
    async def _startup() -> None:
        from app.db.mongo import ping_db
        logger.info("ragDMS %s starting (provider=%s)", __version__, settings.ai_provider)
        if await ping_db():
            logger.info("MongoDB connected")
        else:
            logger.warning("MongoDB not reachable at startup")

    return app

app = create_app()
