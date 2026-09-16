"""PPTX parsing via python-pptx — extract text from slides."""
from __future__ import annotations
import logging
from pathlib import Path
from pptx import Presentation
from pptx.util import Pt

logger = logging.getLogger("ragdms.pptx_parser")


def parse_pptx(path: Path) -> list[dict]:
    """Extract blocks from a PPTX file. Each slide becomes a page."""
    prs = Presentation(str(path))
    blocks: list[dict] = []
    
    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_title: str | None = None
        
        # Extract title first
        if slide.shapes.title:
            title_text = (slide.shapes.title.text or "").strip()
            if title_text:
                slide_title = title_text
                blocks.append({
                    "text": title_text,
                    "page": slide_num,
                    "kind": "heading",
                    "heading": slide_title,
                })
        
        # Extract text from all shapes
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue  # already handled
            if not shape.has_text_frame:
                continue
            
            # Check if shape has a table
            if shape.has_table:
                table = shape.table
                for row_idx, row in enumerate(table.rows):
                    cells = [(cell.text or "").strip() for cell in row.cells]
                    text = " | ".join(cells)
                    if text.strip():
                        blocks.append({
                            "text": text,
                            "page": slide_num,
                            "kind": "table_row",
                            "heading": slide_title,
                        })
                continue
            
            for para in shape.text_frame.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                blocks.append({
                    "text": text,
                    "page": slide_num,
                    "kind": "body",
                    "heading": slide_title,
                })
    
    logger.info("Parsed PPTX: %d blocks from %d slides", len(blocks), len(prs.slides))
    return blocks
