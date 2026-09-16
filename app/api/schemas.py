"""Pydantic schemas for request / response bodies."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    """POST /jobs/process — triggered by Express after upload or version edit."""
    workspaceId: str = Field(min_length=1, max_length=64)
    documentId: str = Field(min_length=1, max_length=64)
    versionId: str = Field(min_length=1, max_length=64)
    s3Key: str = Field(min_length=1)
    categories: list[str] = Field(
        default_factory=lambda: ["HR", "Finance", "Projects", "Legal", "General"],
        description="Workspace-defined category list for classification",
    )


class ProcessResponse(BaseModel):
    """Response from document processing."""
    status: Literal["ready", "failed"]
    summary: str = ""
    aiCategory: str = ""
    aiKeywords: list[str] = Field(default_factory=list)
    chunkCount: int = 0


# ---------------------------------------------------------------------------
# RAG Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in chat history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """POST /rag/chat — triggered by Express for Q&A."""
    workspaceId: str = Field(min_length=1, max_length=64)
    sessionId: str | None = Field(
        default=None,
        description="Chat session ID for multi-turn conversations",
    )
    allowedDocumentIds: list[str] = Field(
        default_factory=list,
        description="Document IDs the user has access to (computed by Express)",
    )
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Last N messages for multi-turn context",
    )


class CitedChunk(BaseModel):
    """A chunk cited in the RAG response."""
    documentId: str
    chunkIndex: int
    page: int | None = None
    heading: str | None = None
    text: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """Response from RAG chat."""
    answer: str
    citedDocumentIds: list[str] = Field(default_factory=list)
    citedChunks: list[CitedChunk] = Field(default_factory=list)
    refused: bool = False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /health response."""
    ok: bool = True
    version: str = ""
    provider: str = ""
