"""Deterministic chunker targeting 500–800 tokens per chunk."""
from __future__ import annotations
import logging
import tiktoken

logger = logging.getLogger("ragdms.chunker")

_enc = tiktoken.get_encoding("cl100k_base")


def tok_count(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_enc.encode(text))


def build_chunks(
    blocks: list[dict],
    *,
    min_tokens: int = 500,
    max_tokens: int = 800,
) -> list[dict]:
    """Split parsed blocks into chunks targeting min_tokens–max_tokens.
    
    Returns list of chunk dicts with keys:
        chunkIndex: int
        text: str
        page: int  (first page of the chunk)
        heading: str | None
        kind: str  ('paragraph' | 'table')
    """
    if not blocks:
        return []
    
    chunks: list[dict] = []
    chunk_idx = 0
    
    buf_blocks: list[dict] = []
    buf_tokens: int = 0
    
    def flush():
        nonlocal chunk_idx
        if not buf_blocks:
            return
        text = "\n\n".join(b["text"] for b in buf_blocks)
        has_table = any(b["kind"] == "table_row" for b in buf_blocks)
        chunks.append({
            "chunkIndex": chunk_idx,
            "text": text,
            "page": buf_blocks[0]["page"],
            "heading": buf_blocks[0].get("heading"),
            "kind": "table" if has_table else "paragraph",
        })
        chunk_idx += 1
        buf_blocks.clear()
    
    i = 0
    while i < len(blocks):
        b = blocks[i]
        
        # Handle table_row blocks: group all consecutive table rows together
        if b["kind"] == "table_row":
            flush()  # flush any buffered body text first
            table_group: list[dict] = []
            while i < len(blocks) and blocks[i]["kind"] == "table_row":
                table_group.append(blocks[i])
                i += 1
            # Tables are never split
            text = "\n".join(tb["text"] for tb in table_group)
            chunks.append({
                "chunkIndex": chunk_idx,
                "text": text,
                "page": table_group[0]["page"],
                "heading": table_group[0].get("heading"),
                "kind": "table",
            })
            chunk_idx += 1
            continue
        
        # Body or heading block
        block_text = b["text"]
        block_toks = tok_count(block_text)
        
        # If adding this block would exceed max_tokens, flush first
        if buf_tokens + block_toks > max_tokens and buf_blocks:
            flush()
            buf_tokens = 0
        
        buf_blocks.append(b)
        buf_tokens += block_toks
        
        # Flush when we've reached the target range
        if buf_tokens >= min_tokens:
            flush()
            buf_tokens = 0
        
        i += 1
    
    flush()
    
    # Post-process: merge heading-only first chunk into successor
    if len(chunks) >= 2:
        first = chunks[0]
        if tok_count(first["text"]) < 20 and first["kind"] == "paragraph":
            second = chunks[1]
            second["text"] = first["text"] + "\n\n" + second["text"]
            second["chunkIndex"] = first["chunkIndex"]
            chunks.pop(0)
            # Re-index
            for idx, c in enumerate(chunks):
                c["chunkIndex"] = idx
    
    logger.info("Chunked %d blocks -> %d chunks", len(blocks), len(chunks))
    return chunks
