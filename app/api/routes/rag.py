"""RAG chat endpoint — POST /rag/chat."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.schemas import ChatRequest, ChatResponse, CitedChunk
from app.deps import get_db, get_embed_provider, get_llm_provider
from app.middleware.auth import verify_internal_token
from app.providers.base import BaseEmbedProvider, BaseLLMProvider
from app.services.rag_service import chat

logger = logging.getLogger("ragdms.routes.rag")

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/chat", response_model=ChatResponse)
async def rag_chat(
    req: ChatRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    llm: BaseLLMProvider = Depends(get_llm_provider),
    embedder: BaseEmbedProvider = Depends(get_embed_provider),
) -> ChatResponse:
    """RAG query over permitted documents.

    Called by Express with workspaceId, allowedDocumentIds (pre-computed by ACL),
    question, and optional history for multi-turn context.

    If allowedDocumentIds is empty, returns refusal immediately.
    If no relevant chunks are found, returns refusal.
    """
    logger.info(
        "RAG chat: workspace=%s, docs=%d, session=%s",
        req.workspaceId,
        len(req.allowedDocumentIds),
        req.sessionId,
    )

    history = [{"role": m.role, "content": m.content} for m in req.history]

    result = await chat(
        workspace_id=req.workspaceId,
        session_id=req.sessionId,
        allowed_document_ids=req.allowedDocumentIds,
        question=req.question,
        history=history,
        db=db,
        llm=llm,
        embedder=embedder,
    )

    cited_chunks = [
        CitedChunk(**c) for c in result.get("citedChunks", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        citedDocumentIds=result.get("citedDocumentIds", []),
        citedChunks=cited_chunks,
        refused=result.get("refused", False),
    )
