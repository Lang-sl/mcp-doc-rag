"""PDF parser.

Extracts text from PDF files using pdfplumber, splits into paragraph-based
chunks of ~1500 characters each.
"""

from __future__ import annotations

from typing import Any


def parse_pdf(file_path: str, source_label: str, source_module: str) -> list[dict[str, Any]]:
    """Parse a PDF file into a list of structured element dicts.

    Returns [] on errors (encrypted PDF, no text, file not found, etc.).
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    try:
        pdf = pdfplumber.open(file_path)
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                continue

            chunks = _chunk_paragraphs(paragraphs, max_chars=1500)

            for chunk_text in chunks:
                results.append({
                    "type": "pdf_section",
                    "symbol_id": None,
                    "class_name": None,
                    "function_name": None,
                    "signature": None,
                    "params": [],
                    "return_desc": None,
                    "remarks": chunk_text,
                    "example": None,
                    "see_also": [],
                    "references": [],
                    "contains_code": False,
                    "section_title": f"Page {page_num}",
                    "page": page_num,
                    "source_label": source_label,
                    "source_module": source_module,
                    "file_path": file_path,
                })
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return results


def _chunk_paragraphs(paragraphs: list[str], max_chars: int = 1500) -> list[str]:
    """Group *paragraphs* into chunks, each at most *max_chars* characters long.

    Each chunk is a single string of joined paragraphs separated by blank lines.
    Paragraphs longer than *max_chars* are split at sentence boundaries or,
    failing that, at word boundaries.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        for sub_para in _split_long_paragraph(para, max_chars):
            sub_len = len(sub_para)

            if current and current_len + sub_len > max_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0

            current.append(sub_para)
            current_len += sub_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """Split a single paragraph into sub-paragraphs if it exceeds *max_chars*.

    Splitting prefers sentence boundaries (``. `` ``! `` ``? ``), then falls
    back to word boundaries or character slicing.
    """
    if len(para) <= max_chars:
        return [para]

    parts: list[str] = []
    remaining = para

    while len(remaining) > max_chars:
        split_at = _find_sentence_boundary(remaining, max_chars)
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def _find_sentence_boundary(text: str, max_pos: int) -> int:
    """Find the best split position in *text* at or before *max_pos*.

    Prefers sentence-ending punctuation followed by a space, then falls back
    to the last space, then to *max_pos* itself.
    """
    window = text[:max_pos + 1]

    # Prefer sentence boundary: . ! ? followed by space
    for marker in (". ", "! ", "? "):
        pos = window.rfind(marker)
        if pos > 0:
            return pos + len(marker)

    # Fall back to last space
    pos = window.rfind(" ")
    if pos > 0:
        return pos + 1

    return max_pos
