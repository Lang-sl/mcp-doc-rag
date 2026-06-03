"""Chunk assembler.

Converts parsed element dicts (from parser_html, parser_pdf, parser_header)
into Chunk objects from rag.models.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rag.config import Config
from rag.models import Chunk


def _build_embed_text(parsed: dict) -> str:
    """Flatten a structured parsed dict into a string for embedding."""
    lines: list[str] = []

    symbol_id = parsed.get("symbol_id")
    if symbol_id:
        lines.append(f"[SYMBOL] {symbol_id}")

    class_name = parsed.get("class_name")
    if class_name:
        lines.append(f"[CLASS] {class_name}")

    function_name = parsed.get("function_name")
    if function_name:
        lines.append(f"[FUNCTION] {function_name}")

    section_title = parsed.get("section_title")
    if section_title:
        lines.append(f"[SECTION] {section_title}")

    signature = parsed.get("signature")
    if signature:
        lines.append(f"[SIGNATURE] {signature}")

    params = parsed.get("params", [])
    if params:
        lines.append("[PARAMS]")
        for p in params:
            name = p.get("name", "")
            ptype = p.get("type", "")
            desc = p.get("desc", "")
            if ptype:
                lines.append(f"  - {name} ({ptype}): {desc}")
            else:
                lines.append(f"  - {name}: {desc}")

    return_desc = parsed.get("return_desc")
    if return_desc:
        lines.append(f"[RETURN] {return_desc}")

    remarks = parsed.get("remarks")
    if remarks:
        lines.append(f"[REMARKS] {remarks}")

    example = parsed.get("example")
    if example:
        lines.append(f"[EXAMPLE] {example}")

    return "\n".join(lines)


def _build_bm25_fields(parsed: dict) -> dict[str, str]:
    """Build field-weighted BM25 text fields for hybrid search."""
    symbol_parts: list[str] = []
    for key in ("symbol_id", "class_name", "function_name"):
        val = parsed.get(key)
        if val:
            symbol_parts.append(val)
    symbol_name = " ".join(symbol_parts)

    sig_parts: list[str] = []
    signature = parsed.get("signature")
    if signature:
        sig_parts.append(signature)
    for p in parsed.get("params", []):
        ptype = p.get("type", "")
        name = p.get("name", "")
        if ptype and name:
            sig_parts.append(f"{ptype} {name}")
        elif name:
            sig_parts.append(name)
    signature_text = " ".join(sig_parts)

    return {
        "symbol_name": symbol_name,
        "signature": signature_text,
        "remarks": parsed.get("remarks") or "",
        "example": parsed.get("example") or "",
    }


def _generate_chunk_id(parsed: dict) -> str:
    """Generate a deterministic chunk ID from parsed dict fields.

    Includes a short hash of remarks text to ensure uniqueness for
    elements that share the same symbol_id and section_title (e.g.
    multiple PDF paragraphs on the same page).
    """
    source_label = parsed.get("source_label", "")
    file_path = parsed.get("file_path", "")
    symbol_id = parsed.get("symbol_id") or ""
    elem_type = parsed.get("type", "")
    section_title = parsed.get("section_title") or ""

    raw = f"{source_label}:{file_path}:{symbol_id}:{elem_type}:{section_title}"
    # Append hash of remarks to disambiguate same-page paragraphs
    remarks = parsed.get("remarks") or ""
    if remarks:
        raw += f":{hashlib.md5(remarks.encode('utf-8')).hexdigest()[:8]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def build_chunk(parsed: dict, config: Config) -> Chunk | None:
    """Create a Chunk from a parsed element dict.

    Returns None if both remarks and signature are empty/None, or if the
    generated embed_text is empty.
    """
    remarks = parsed.get("remarks")
    signature = parsed.get("signature")
    if not remarks and not signature:
        return None

    embed_text = _build_embed_text(parsed)
    if not embed_text:
        return None

    max_chars = config.chunk_max_chars
    embed_text = embed_text[:max_chars]

    bm25_fields = _build_bm25_fields(parsed)
    if len(bm25_fields.get("remarks", "")) > max_chars:
        bm25_fields["remarks"] = bm25_fields["remarks"][:max_chars]

    return Chunk(
        chunk_id=_generate_chunk_id(parsed),
        type=parsed.get("type", "narrative"),
        symbol_id=parsed.get("symbol_id"),
        class_name=parsed.get("class_name"),
        function_name=parsed.get("function_name"),
        signature=parsed.get("signature"),
        params=parsed.get("params", []),
        return_desc=parsed.get("return_desc"),
        remarks=remarks,
        example=parsed.get("example"),
        see_also=parsed.get("see_also", []),
        references=parsed.get("references", []),
        contains_code=bool(parsed.get("contains_code", False)),
        source_label=parsed.get("source_label", ""),
        source_module=parsed.get("source_module", ""),
        source_file=parsed.get("file_path", ""),
        embed_text=embed_text,
        bm25_fields=bm25_fields,
    )


def build_chunks(parsed_list: list[dict], config: Config) -> list[Chunk]:
    """Convert a list of parsed dicts to Chunk objects, filtering None results."""
    return [c for p in parsed_list if (c := build_chunk(p, config)) is not None]
