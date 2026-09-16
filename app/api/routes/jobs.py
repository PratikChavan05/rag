"""Document processing endpoint — POST /jobs/process."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas import ProcessRequest, ProcessResponse
from app.deps import get_db, get_embed_provider, get_llm_provider
from app.middleware.auth import verify_internal_token
from app.providers.base import BaseEmbedProvider, BaseLLMProvider
from app.services.process_service import process_document

logger = logging.getLogger("ragdms.routes.jobs")

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/process", response_model=ProcessResponse)
async def process_job(
    req: ProcessRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    llm: BaseLLMProvider = Depends(get_llm_provider),
    embedder: BaseEmbedProvider = Depends(get_embed_provider),
) -> ProcessResponse:
    """Process a document: extract → chunk → embed → summarize → categorize.

    Called by Express after upload or version edit. Requires X-Internal-Token.
    """
    logger.info(
        "Processing job: doc=%s version=%s workspace=%s",
        req.documentId, req.versionId, req.workspaceId,
    )

    try:
        result = await process_document(
            workspace_id=req.workspaceId,
            document_id=req.documentId,
            version_id=req.versionId,
            s3_key=req.s3Key,
            categories=req.categories,
            db=db,
            llm=llm,
            embedder=embedder,
        )
        return ProcessResponse(
            status="ready",
            summary=result.get("summary", ""),
            aiCategory=result.get("aiCategory", ""),
            aiKeywords=result.get("aiKeywords", []),
            chunkCount=result.get("chunkCount", 0),
        )
    except Exception as exc:
        logger.exception("Job processing failed")
        return ProcessResponse(
            status="failed",
            summary="",
            aiCategory="",
            aiKeywords=[],
            chunkCount=0,
        )
