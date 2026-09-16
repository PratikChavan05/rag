"""RAG chat service — vector search → LLM answer with citations."""
from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import Settings, get_settings
from app.providers.base import BaseEmbedProvider, BaseLLMProvider
from app.services.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger("ragdms.rag_service")

# Refusal message when no relevant chunks are found
_REFUSAL = (
    "I couldn't find relevant information in the available documents "
    "to answer this question."
)


async def chat(
    *,
    workspace_id: str,
    session_id: str | None,
    allowed_document_ids: list[str],
    question: str,
    history: list[dict[str, str]],
    db: AsyncIOMotorDatabase,
    llm: BaseLLMProvider,
    embedder: BaseEmbedProvider,
) -> dict[str, Any]:
    """Execute a RAG chat query.

    1. Early return refusal if allowedDocumentIds is empty
    2. Embed query
    3. Atlas Vector Search on document_chunks filtered by workspaceId + allowedDocumentIds
    4. If no chunks above threshold → refuse
    5. Build context string from retrieved chunks
    6. Build messages: system prompt + history + user question
    7. Call LLM
    8. Extract cited document IDs
    9. Return answer + citations

    Returns:
        {
            "answer": str,
            "citedDocumentIds": list[str],
            "citedChunks": list[dict],
            "refused": bool,
        }
    """
    settings = get_settings()

    # --- Step 1: Guard empty ACL ---
    if not allowed_document_ids:
        return {
            "answer": _REFUSAL,
            "citedDocumentIds": [],
            "citedChunks": [],
            "refused": True,
        }

    ws_oid = ObjectId(workspace_id)
    doc_oids = [ObjectId(did) for did in allowed_document_ids]

    # --- Step 2: Embed query ---
    query_embeddings = await embedder.embed([question])
    if not query_embeddings:
        return {
            "answer": _REFUSAL,
            "citedDocumentIds": [],
            "citedChunks": [],
            "refused": True,
        }
    query_vector = query_embeddings[0]

    # --- Step 3: Atlas Vector Search ---
    chunks = await _vector_search(
        db=db,
        query_vector=query_vector,
        workspace_id=ws_oid,
        document_ids=doc_oids,
        top_k=settings.rag_top_k,
        similarity_threshold=settings.rag_similarity_threshold,
    )

    # --- Step 4: Check for relevant results ---
    if not chunks:
        logger.info("No relevant chunks found for query in workspace=%s", workspace_id)
        return {
            "answer": _REFUSAL,
            "citedDocumentIds": [],
            "citedChunks": [],
            "refused": True,
        }

    logger.info("Retrieved %d chunks for query in workspace=%s", len(chunks), workspace_id)

    # --- Step 5: Build context ---
    context = _build_context(chunks)

    # --- Step 6: Build messages ---
    history_block = _build_history_block(history)

    system_prompt = RAG_SYSTEM_PROMPT.format(
        context=context,
        history_block=history_block,
        question=question,
    )

    messages = [{"role": "user", "content": system_prompt}]

    # --- Step 7: Call LLM ---
    try:
        answer = await llm.chat(
            messages,
            temperature=0.3,
            max_tokens=1500,
        )
        answer = _clean_response(answer)
    except Exception:
        logger.exception("LLM call failed for RAG query")
        return {
            "answer": "I encountered an error processing your question. Please try again.",
            "citedDocumentIds": [],
            "citedChunks": [],
            "refused": False,
        }

    # --- Step 8: Extract cited document IDs ---
    cited_doc_ids = _extract_cited_doc_ids(chunks)

    # --- Step 9: Build cited chunks for response ---
    cited_chunks = [
        {
            "documentId": str(c["documentId"]),
            "chunkIndex": c["chunkIndex"],
            "page": c.get("page"),
            "heading": c.get("heading"),
            "text": c["text"][:300],  # truncate for response payload
            "score": c.get("score", 0.0),
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "citedDocumentIds": cited_doc_ids,
        "citedChunks": cited_chunks,
        "refused": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _vector_search(
    *,
    db: AsyncIOMotorDatabase,
    query_vector: list[float],
    workspace_id: ObjectId,
    document_ids: list[ObjectId],
    top_k: int,
    similarity_threshold: float,
) -> list[dict]:
    """Run Atlas Vector Search with pre-filter on workspaceId + documentId.

    Uses the $vectorSearch aggregation stage. Requires an Atlas Vector Search
    index named 'vector_index' on the document_chunks collection with:
        - path: 'embedding'
        - filter fields: 'workspaceId', 'documentId'
        - dimensions: 1536
        - similarity: cosine
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,
                "limit": top_k,
                "filter": {
                    "workspaceId": workspace_id,
                    "documentId": {"$in": document_ids},
                },
            }
        },
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        {
            "$project": {
                "embedding": 0,  # don't return the large vector
            }
        },
    ]

    try:
        results = []
        async for doc in db.document_chunks.aggregate(pipeline):
            score = doc.get("score", 0.0)
            if score >= similarity_threshold:
                results.append(doc)
        return results
    except Exception:
        logger.exception("Atlas Vector Search failed, falling back to empty results")
        return []


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks):
        doc_id = str(c.get("documentId", ""))
        page = c.get("page", "?")
        heading = c.get("heading") or ""
        kind = c.get("kind", "paragraph")
        score = c.get("score", 0.0)

        header_parts = [f"[Chunk {i}]"]
        header_parts.append(f"(Doc: {doc_id[-8:]}, Page {page})")
        if heading:
            header_parts.append(f"[{heading}]")
        if kind == "table":
            header_parts.append("<TABLE>")

        parts.append(f"{' '.join(header_parts)}\n{c['text']}\n")

    return "\n".join(parts)


def _build_history_block(history: list[dict[str, str]]) -> str:
    """Format chat history for the prompt."""
    if not history:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for msg in history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        # Truncate long history messages to keep context manageable
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


def _extract_cited_doc_ids(chunks: list[dict]) -> list[str]:
    """Extract unique document IDs from retrieved chunks."""
    seen: set[str] = set()
    result: list[str] = []
    for c in chunks:
        doc_id = str(c.get("documentId", ""))
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def _clean_response(text: str) -> str:
    """Clean LLM response: strip preamble and thinking tags."""
    import re

    # Remove <think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip common preambles
    preamble_re = re.compile(
        r"^(?:\s*(?:sure|certainly|of course|absolutely|okay|ok|"
        r"let me|i(?:'|')?ll|based on|here(?:'s| is))"
        r"[\s,:.!—-]*)+",
        re.IGNORECASE,
    )
    text = preamble_re.sub("", text).lstrip()

    return text
