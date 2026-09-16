"""Document processing orchestration — extract, chunk, embed, summarize, categorize."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import Settings, get_settings
from app.ingestion.chunker import build_chunks
from app.ingestion.pipeline import SUPPORTED_EXTENSIONS, extract_blocks
from app.providers.base import BaseEmbedProvider, BaseLLMProvider
from app.services.prompts import CATEGORIZE_PROMPT, KEYWORDS_PROMPT, SUMMARY_PROMPT
from app.storage.s3 import download_to_tempfile

logger = logging.getLogger("ragdms.process_service")


async def process_document(
    *,
    workspace_id: str,
    document_id: str,
    version_id: str,
    s3_key: str,
    categories: list[str],
    db: AsyncIOMotorDatabase,
    llm: BaseLLMProvider,
    embedder: BaseEmbedProvider,
) -> dict:
    """Full document processing pipeline.

    1. Update ai_jobs status → running
    2. Download file from S3
    3. Parse (extract blocks)
    4. Chunk (500–800 tokens)
    5. Embed all chunks
    6. Replace document_chunks in MongoDB (with embeddings for Atlas Vector Search)
    7. Generate summary via LLM
    8. Classify category via LLM
    9. Extract keywords via LLM
    10. Update documents collection with AI fields
    11. Update document_versions.processing → ready
    12. Update ai_jobs status → ready

    Returns dict with summary, aiCategory, aiKeywords, chunkCount.
    """
    settings = get_settings()
    doc_oid = ObjectId(document_id)
    ver_oid = ObjectId(version_id)
    ws_oid = ObjectId(workspace_id)

    # --- Step 1: Mark job as running ---
    await db.ai_jobs.update_one(
        {"documentId": doc_oid, "versionId": ver_oid, "workspaceId": ws_oid},
        {"$set": {"status": "running"}},
    )

    tmp_path: Path | None = None
    try:
        # --- Step 2: Download from S3 ---
        ext = Path(s3_key).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
        tmp_path = await download_to_tempfile(s3_key, suffix=ext)
        logger.info("Downloaded s3://%s for doc=%s", s3_key, document_id)

        # --- Step 3: Parse ---
        blocks = extract_blocks(tmp_path)
        logger.info("Extracted %d blocks from %s", len(blocks), s3_key)

        # --- Step 4: Chunk ---
        chunks = build_chunks(
            blocks,
            min_tokens=settings.chunk_min_tokens,
            max_tokens=settings.chunk_max_tokens,
        )
        logger.info("Created %d chunks for doc=%s", len(chunks), document_id)

        # --- Step 5: Embed ---
        texts = [c["text"] for c in chunks]
        embeddings = await embedder.embed(texts) if texts else []
        logger.info("Embedded %d chunks for doc=%s", len(embeddings), document_id)

        # --- Step 6: Replace document_chunks ---
        await db.document_chunks.delete_many({"documentId": doc_oid})

        if chunks and embeddings:
            chunk_docs = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_docs.append({
                    "workspaceId": ws_oid,
                    "documentId": doc_oid,
                    "versionId": ver_oid,
                    "chunkIndex": chunk["chunkIndex"],
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "heading": chunk.get("heading"),
                    "kind": chunk.get("kind", "paragraph"),
                    "embedding": embedding,
                })
            await db.document_chunks.insert_many(chunk_docs)
            logger.info("Stored %d chunks in document_chunks for doc=%s", len(chunk_docs), document_id)

        # --- Step 7: Summarize ---
        summary = await _generate_summary(texts, llm)

        # --- Step 8: Categorize ---
        ai_category = await _classify_category(texts, categories, llm)

        # --- Step 9: Keywords ---
        ai_keywords = await _extract_keywords(texts, llm)

        # --- Step 10: Update documents collection ---
        await db.documents.update_one(
            {"_id": doc_oid},
            {"$set": {
                "summary": summary,
                "aiCategory": ai_category,
                "aiKeywords": ai_keywords,
            }},
        )

        # --- Step 11: Update document_versions.processing ---
        await db.document_versions.update_one(
            {"_id": ver_oid},
            {"$set": {
                "processing.extract": "ready",
                "processing.embed": "ready",
                "processing.classify": "ready",
            }},
        )

        # --- Step 12: Mark job as ready ---
        await db.ai_jobs.update_one(
            {"documentId": doc_oid, "versionId": ver_oid, "workspaceId": ws_oid},
            {"$set": {"status": "ready", "error": None}},
        )

        result = {
            "status": "ready",
            "summary": summary,
            "aiCategory": ai_category,
            "aiKeywords": ai_keywords,
            "chunkCount": len(chunks),
        }
        logger.info("Processing complete for doc=%s: %d chunks, category=%s", document_id, len(chunks), ai_category)
        return result

    except Exception as exc:
        logger.exception("Processing failed for doc=%s", document_id)
        # Mark job as failed
        await db.ai_jobs.update_one(
            {"documentId": doc_oid, "versionId": ver_oid, "workspaceId": ws_oid},
            {"$set": {"status": "failed", "error": str(exc)[:500]}},
        )
        await db.document_versions.update_one(
            {"_id": ver_oid},
            {"$set": {
                "processing.extract": "failed",
                "processing.embed": "failed",
                "processing.classify": "failed",
            }},
        )
        raise

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_MAX_SUMMARY_CHARS = 6000  # ~1500 tokens of input for summary


async def _generate_summary(texts: list[str], llm: BaseLLMProvider) -> str:
    """Generate a 2–3 sentence document summary from the first ~3000 tokens."""
    if not texts:
        return ""
    combined = "\n\n".join(texts)[:_MAX_SUMMARY_CHARS]
    prompt = SUMMARY_PROMPT.format(text=combined)
    try:
        summary = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return summary.strip()
    except Exception:
        logger.exception("Summary generation failed")
        return ""


async def _classify_category(
    texts: list[str],
    categories: list[str],
    llm: BaseLLMProvider,
) -> str:
    """Classify the document into one of the workspace categories."""
    if not texts:
        return "General"
    if not categories:
        categories = ["HR", "Finance", "Projects", "Legal", "General"]

    combined = "\n\n".join(texts)[:_MAX_SUMMARY_CHARS]
    categories_str = "\n".join(f"- {c}" for c in categories)
    prompt = CATEGORIZE_PROMPT.format(text=combined, categories=categories_str)
    try:
        result = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        result = result.strip().strip('"').strip("'")
        # Validate the response is one of the given categories
        for cat in categories:
            if cat.lower() == result.lower():
                return cat
        return result if result else "General"
    except Exception:
        logger.exception("Category classification failed")
        return "General"


async def _extract_keywords(texts: list[str], llm: BaseLLMProvider) -> list[str]:
    """Extract 5–10 keywords from the document."""
    if not texts:
        return []
    combined = "\n\n".join(texts)[:_MAX_SUMMARY_CHARS]
    prompt = KEYWORDS_PROMPT.format(text=combined)
    try:
        result = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        # Parse JSON array from response
        result = result.strip()
        # Handle markdown code blocks
        if result.startswith("```"):
            result = re.sub(r"^```\w*\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
        keywords = json.loads(result)
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if k][:10]
        return []
    except (json.JSONDecodeError, Exception):
        logger.exception("Keyword extraction failed")
        return []
