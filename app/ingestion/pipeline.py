"""Dispatch incoming documents to the right parser by extension."""
from __future__ import annotations
import logging
from pathlib import Path
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.docx_parser import parse_docx
from app.ingestion.pptx_parser import parse_pptx

logger = logging.getLogger("ragdms.pipeline")

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "txt",
}


def extract_blocks(file_path: Path) -> list[dict]:
    """Parse a document file and return a flat list of text blocks.
    
    Each block is a dict with keys:
        text: str - the text content
        page: int - 1-indexed page number
        kind: str - 'heading', 'body', 'table_row'
        heading: str | None - section heading this block belongs to
    """
    ext = file_path.suffix.lower()
    kind = SUPPORTED_EXTENSIONS.get(ext)
    if kind is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    logger.info("Parsing %s as %s", file_path.name, kind)
    
    if kind == "pdf":
        return parse_pdf(file_path)
    elif kind == "docx":
        return parse_docx(file_path)
    elif kind == "pptx":
        return parse_pptx(file_path)
    elif kind == "txt":
        return _parse_txt(file_path)
    
    raise ValueError(f"Unknown kind: {kind}")


def _parse_txt(path: Path) -> list[dict]:
    """Parse plain text / markdown files."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[dict] = []
    current_heading: str | None = None
    
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Treat ALL-CAPS short lines or markdown headings as headings
        is_heading = (
            (len(s) <= 80 and s.upper() == s and any(c.isalpha() for c in s))
            or s.startswith("# ")
        )
        if is_heading:
            current_heading = s.lstrip("# ").strip()
            blocks.append({
                "text": current_heading,
                "page": 1,
                "kind": "heading",
                "heading": current_heading,
            })
        else:
            blocks.append({
                "text": s,
                "page": 1,
                "kind": "body",
                "heading": current_heading,
            })
    return blocks
