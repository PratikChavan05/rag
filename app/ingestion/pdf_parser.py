"""PDF parsing via pdfplumber — extract text blocks and tables."""
from __future__ import annotations
import logging
import re
import statistics
from pathlib import Path
from typing import Any
import pdfplumber

logger = logging.getLogger("ragdms.pdf_parser")

LINE_TOL = 3.0
PARA_GAP_TOL = 14.0


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Extract blocks from a PDF file."""
    line_blocks: list[dict] = []
    table_blocks: list[dict] = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_num = page.page_number
            
            # Extract tables and track their bounding boxes
            page_table_bboxes: list[tuple[float, float, float, float]] = []
            try:
                tables = page.find_tables()
                for table in tables:
                    try:
                        rows = table.extract()
                    except Exception:
                        continue
                    if not rows:
                        continue
                    bbox = (float(table.bbox[0]), float(table.bbox[1]),
                            float(table.bbox[2]), float(table.bbox[3]))
                    page_table_bboxes.append(bbox)
                    
                    # Clean and format table rows
                    cleaned = _clean_table_rows(rows)
                    if cleaned:
                        header = cleaned[0]
                        for row in cleaned[1:]:
                            text = " | ".join(row)
                            if text.strip():
                                table_blocks.append({
                                    "text": text,
                                    "page": page_num,
                                    "kind": "table_row",
                                    "heading": None,
                                    "_bbox": bbox,
                                    "_table_header": header,
                                })
            except Exception:
                pass
            
            # Extract words with font sizes
            words = page.extract_words(extra_attrs=["size"]) or []
            page_lines = _words_to_lines(words, page_num)
            
            # Filter out lines inside table regions
            for lb in page_lines:
                lb_bbox = lb.get("_bbox")
                if lb_bbox and any(_is_inside(lb_bbox, tb) for tb in page_table_bboxes):
                    continue
                line_blocks.append(lb)
    
    # Classify blocks by font size
    _classify_by_font_size(line_blocks)
    
    # Merge all blocks and sort by reading order
    all_blocks = line_blocks + table_blocks
    all_blocks.sort(key=lambda b: (
        b["page"],
        round(b.get("_bbox", (0, 0, 0, 0))[1], 1),
        b.get("_bbox", (0, 0, 0, 0))[0],
    ))
    
    # Merge consecutive body lines into paragraphs
    all_blocks = _merge_paragraphs(all_blocks)
    
    # Track current heading and clean output
    current_heading: str | None = None
    result: list[dict] = []
    for b in all_blocks:
        text = b.get("text", "").strip()
        if not text:
            continue
        kind = b.get("kind", "body")
        if kind == "heading":
            current_heading = text
        b["heading"] = current_heading
        result.append({
            "text": text,
            "page": b["page"],
            "kind": kind,
            "heading": current_heading,
        })
    
    logger.info("Parsed PDF: %d blocks", len(result))
    return result


def _words_to_lines(words: list[dict[str, Any]], page_num: int) -> list[dict]:
    """Group words into visual lines."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines: list[list[dict]] = []
    for w in words:
        wt = float(w["top"])
        placed = False
        for line in lines:
            if abs(float(line[0]["top"]) - wt) <= LINE_TOL:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    
    blocks: list[dict] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: float(w["x0"]))
        text = " ".join(w["text"] for w in line_sorted).strip()
        if not text:
            continue
        sizes = [float(w.get("size", 0)) for w in line_sorted if w.get("size")]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        x0 = min(float(w["x0"]) for w in line_sorted)
        top = min(float(w["top"]) for w in line_sorted)
        x1 = max(float(w["x1"]) for w in line_sorted)
        bottom = max(float(w["bottom"]) for w in line_sorted)
        blocks.append({
            "text": text,
            "page": page_num,
            "kind": "body",  # classified later
            "heading": None,
            "_bbox": (x0, top, x1, bottom),
            "_avg_size": avg_size,
        })
    return blocks


def _classify_by_font_size(blocks: list[dict]) -> None:
    """Classify line blocks as heading/body/footnote using modal font size."""
    sizes = [round(b["_avg_size"], 1) for b in blocks if b.get("_avg_size")]
    if not sizes:
        return
    try:
        modal = statistics.mode(sizes)
    except statistics.StatisticsError:
        modal = statistics.median(sizes)
    for b in blocks:
        s = b.get("_avg_size", 0.0)
        if modal and s >= modal + 1.0:
            b["kind"] = "heading"
        elif modal and s <= modal - 1.0:
            b["kind"] = "footnote"
        else:
            b["kind"] = "body"


def _merge_paragraphs(blocks: list[dict]) -> list[dict]:
    """Merge consecutive body/footnote lines into paragraph blocks."""
    if not blocks:
        return []
    merged: list[dict] = []
    buf: list[dict] = []
    
    def flush():
        if not buf:
            return
        if len(buf) == 1:
            merged.append(buf[0])
        else:
            text = "\n".join(b["text"] for b in buf)
            merged.append({**buf[0], "text": text})
        buf.clear()
    
    for b in blocks:
        kind = b.get("kind", "body")
        if kind in ("heading", "table_row"):
            flush()
            merged.append(b)
            continue
        if buf:
            prev = buf[-1]
            prev_bbox = prev.get("_bbox", (0, 0, 0, 0))
            curr_bbox = b.get("_bbox", (0, 0, 0, 0))
            gap = curr_bbox[1] - prev_bbox[3]
            if b["page"] != prev["page"] or gap > PARA_GAP_TOL:
                flush()
        buf.append(b)
    flush()
    return merged


def _is_inside(block_bbox: tuple, table_bbox: tuple) -> bool:
    """Check if block center falls inside a table bounding box."""
    bx0, btop, bx1, bbot = block_bbox
    tx0, ttop, tx1, tbot = table_bbox
    cx = (bx0 + bx1) / 2.0
    cy = (btop + bbot) / 2.0
    return tx0 <= cx <= tx1 and ttop <= cy <= tbot


def _clean_table_rows(rows: list[list]) -> list[list[str]]:
    """Clean table rows: normalize whitespace, drop empty rows."""
    ws_re = re.compile(r"\s+")
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [ws_re.sub(" ", str(c or "").replace("\n", " ")).strip() for c in row]
        if any(c for c in cells):
            cleaned.append(cells)
    return cleaned
