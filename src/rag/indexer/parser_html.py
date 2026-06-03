"""Doxygen HTML parser.

Parses Doxygen-generated HTML files into structured dicts for chunking.

Supported formats:
1. Modern Doxygen (v1.9+)  -- div.memitem / div.memberdef containers (DepBaseApp)
2. ModuleWorks CHM-based    -- h3-per-member tables with syntax-coloured signatures
3. Legacy Doxygen (v1.3)    -- table-based layout, malformed markup (MachineWorks)
4. Narrative / guide pages  -- split by h1/h2/h3 headings (fallback)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from rag.indexer.parser_registry import register_parser


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@register_parser(file_type="html", extensions=[".html", ".htm"])
def parse_html(file_path: str, source_label: str, source_module: str) -> list[dict[str, Any]]:
    """Parse a Doxygen HTML file into a list of structured element dicts.

    Returns [] on errors, empty files, or files with no extractable content.
    """
    path = Path(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except (OSError, UnicodeDecodeError):
        return []

    if not raw.strip():
        return []

    # ---- Format detection ----
    # Modern Doxygen: <meta name="generator" content="Doxygen ...">
    #   or contains div.memitem elements
    is_modern = bool(re.search(
        r'<meta\s[^>]*name\s*=\s*["\']generator["\'][^>]*content\s*=\s*["\']Doxygen',
        raw, re.IGNORECASE,
    ))
    if not is_modern:
        # Check for memitem classes
        is_modern = 'class="memitem"' in raw or "class='memitem'" in raw

    # ModuleWorks: contains chm-test.js or id="main_chm"
    is_moduleworks = 'chm-test.js' in raw or 'id="main_chm"' in raw

    # Legacy Doxygen: <TITLE> contains "Function:" / "Data type:" / "Shader ...:"
    is_legacy = bool(re.search(
        r"<TITLE[^>]*>\s*(Function|Data type|Shader)\b",
        raw, re.IGNORECASE,
    ))

    # ---- Dispatch ----
    if is_modern:
        results = _parse_modern_doxygen(raw, path)
    elif is_moduleworks:
        results = _parse_moduleworks(raw, path)
    elif is_legacy:
        results = _parse_legacy_doxygen(raw, path)
    else:
        results = _parse_narrative(raw, path)

    # Attach source metadata
    for r in results:
        r["source_label"] = source_label
        r["source_module"] = source_module
        r["file_path"] = str(path)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Collapse all whitespace runs into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_tags(html: str) -> str:
    """Remove all HTML tags, keeping text content, then collapse whitespace."""
    return _clean_text(re.sub(r"<[^>]+>", " ", html))


def _resolve_symbol_id(name: str) -> tuple[str | None, str | None]:
    """Split a C++ qualified name into (class_name, function_name).

    e.g. "MwMultiAxis::CalculateToolpath" -> ("MwMultiAxis", "CalculateToolpath")
    e.g. "magic_enum::detail::static_string" -> ("magic_enum::detail", "static_string")
    Single names go to function_name if they look like a function, else class_name.
    """
    if not name:
        return None, None

    if "::" in name:
        parts = name.rsplit("::", 1)
        return parts[0], parts[1]

    # Heuristic: if name starts with uppercase and contains lowercase, it's a class
    if name[0].isupper() and any(c.islower() for c in name):
        return name, None

    return None, name


# Regex for C++ qualified type names: MyClass, MyClass::Nested, mw::core::Type
_TYPE_RE = re.compile(r"\b([A-Z]\w*(?:::[A-Z]\w*)*)\b")


def _extract_type_names(text: str | None) -> list[str]:
    """Extract C++ type identifiers from a string (signature, return desc, etc.)."""
    if not text:
        return []
    return _TYPE_RE.findall(text)


def _make_element(
    elem_type: str,
    symbol_id: str | None = None,
    class_name: str | None = None,
    function_name: str | None = None,
    signature: str | None = None,
    params: list[dict[str, str]] | None = None,
    return_desc: str | None = None,
    remarks: str | None = None,
    example: str | None = None,
    see_also: list[str] | None = None,
    contains_code: bool = False,
    section_title: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Build a standardised element dict."""
    see_also = see_also or []
    references = list(dict.fromkeys(see_also))
    # Add class name
    if class_name and class_name not in references:
        references.append(class_name)
    # Extract type names from signature and return type (for reference expansion)
    if signature:
        for t in _extract_type_names(signature):
            if t not in references:
                references.append(t)
    if return_desc:
        for t in _extract_type_names(return_desc):
            if t not in references:
                references.append(t)

    return {
        "type": elem_type,
        "symbol_id": symbol_id,
        "class_name": class_name,
        "function_name": function_name,
        "signature": signature,
        "params": params or [],
        "return_desc": return_desc,
        "remarks": remarks,
        "example": example,
        "see_also": see_also,
        "references": references,
        "contains_code": contains_code,
        "section_title": section_title,
        "source_label": "",
        "source_module": "",
        "file_path": str(path) if path else "",
    }


# ---------------------------------------------------------------------------
# 1. Modern Doxygen (DepBaseApp style)
# ---------------------------------------------------------------------------


def _parse_modern_doxygen(raw: str, path: Path) -> list[dict[str, Any]]:
    """Parse modern Doxygen HTML with .memitem containers."""
    soup = BeautifulSoup(raw, "html.parser")

    # Determine page type and class name from title
    title_tag = soup.find("title")
    title_text = _clean_text(title_tag.get_text()) if title_tag else ""

    page_class = _parse_modern_title(title_text)

    # Find all member definitions
    memitems = soup.select(".memitem")
    if not memitems:
        # No members — try narrative fallback
        return _parse_narrative(raw, path)

    results: list[dict[str, Any]] = []
    for item in memitems:
        elem = _extract_modern_member(item, page_class, path)
        if elem:
            results.append(elem)

    return results


def _parse_modern_title(title_text: str) -> str | None:
    """Parse modern Doxygen <title> to get the enclosing class name.

    Titles look like:
      "Framework: ClassName Class Template Reference"
      "Framework: FileName File Reference"
    """
    if not title_text:
        return None

    # Strip leading "Framework:" or similar namespace prefixes
    title_text = re.sub(r"^[^:]+:\s*", "", title_text)

    # Remove trailing qualifiers: "Class Template Reference", "File Reference", etc.
    title_text = re.sub(
        r"\s+(Class|Struct|Enum|File|Namespace|Template)(\s+\w+)*\s+Reference$",
        "", title_text,
    )
    # Remove template parameters like "< N >"
    title_text = re.sub(r"<\s*[^>]+\s*>", "", title_text)

    return _clean_text(title_text) or None


def _extract_modern_member(item: Tag, page_class: str | None, path: Path) -> dict[str, Any] | None:
    """Extract a single member from a .memitem div."""
    # Get the member name / signature from .memname
    memname = item.select_one(".memname")
    signature = _clean_text(memname.get_text(separator=" ", strip=True)) if memname else None

    # Get the full signature from .memproto
    memproto = item.select_one(".memproto")
    full_sig = _clean_text(memproto.get_text(separator=" ", strip=True)) if memproto else signature

    # Description from .memdoc
    memdoc = item.select_one(".memdoc")
    description = _clean_text(memdoc.get_text(separator=" ", strip=True)) if memdoc else None

    # Extract member name from signature
    member_name = None
    if signature:
        # Signature looks like "constexpr ClassName::functionName ( args ) const"
        # Extract the function name (text before first '(')
        m = re.match(r"(?:.*?\s)?(?:[\w:]+::)?(\w+)\s*\(", signature)
        if m:
            member_name = m.group(1)

    # Determine type from surrounding context or title
    elem_type = "function"
    if member_name and page_class and member_name == page_class:
        elem_type = "class"

    class_name, function_name = page_class, member_name
    if elem_type == "class":
        class_name, function_name = member_name, None

    # Extract params from param tables in memdoc
    params = _extract_modern_params(memdoc) if memdoc else []

    # Return value
    return_desc = None
    if memdoc:
        return_section = memdoc.find("dl", class_="section return")
        if return_section:
            return_desc = _clean_text(return_section.get_text(separator=" ", strip=True))
        else:
            # Look for "Returns ..." pattern in description
            ret_m = re.search(r"Returns?\s+(.+?)(?:\.\s|$)", description or "", re.IGNORECASE)
            if ret_m:
                return_desc = _clean_text(ret_m.group(1))

    # Code blocks
    code_blocks, has_code = _extract_code_from_soup(item)

    # See also
    see_also = _extract_see_also_from_soup(item)

    return _make_element(
        elem_type=elem_type,
        symbol_id=f"{class_name}::{function_name}" if class_name and function_name else (class_name or function_name),
        class_name=class_name,
        function_name=function_name,
        signature=full_sig,
        params=params,
        return_desc=return_desc,
        remarks=description,
        example=code_blocks,
        see_also=see_also,
        contains_code=bool(has_code),
        path=path,
    )


def _extract_modern_params(memdoc: Tag | None) -> list[dict[str, str]]:
    """Extract parameters from a .memdoc div's parameter tables."""
    if memdoc is None:
        return []

    params: list[dict[str, str]] = []
    # Modern Doxygen uses <table class="params"> or <dl class="params">
    param_table = memdoc.find("table", class_="params")
    if param_table:
        for row in param_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                param_name = _clean_text(cells[0].get_text())
                param_desc = _clean_text(cells[1].get_text())
                if param_name:
                    params.append({"name": param_name, "type": "", "desc": param_desc})
        return params

    # Also check dl.params
    param_dl = memdoc.find("dl", class_="params")
    if param_dl:
        # Not common in modern doxygen but we handle it
        pass

    return params


# ---------------------------------------------------------------------------
# 2. ModuleWorks CHM format
# ---------------------------------------------------------------------------


def _parse_moduleworks(raw: str, path: Path) -> list[dict[str, Any]]:
    """Parse ModuleWorks CHM-based HTML with h3-per-member structure.

    Each member is in a ``<table style=\"table-layout:...\">`` that contains
    an ``<h3>`` heading with the signature.  The enclosing ``<h2>`` gives the
    class / namespace name.
    """
    soup = BeautifulSoup(raw, "html.parser")
    content = soup.find("div", id="content")
    if content is None:
        return _parse_narrative(raw, path)

    # ---- Determine the enclosing class from h2 ----
    current_class: str | None = None
    h2_tag = content.find("h2")
    if h2_tag:
        h2_text = _clean_text(h2_tag.get_text(separator=" ", strip=True))
        h2_lower = h2_text.lower()
        if h2_lower.startswith("class ") or h2_lower.startswith("struct "):
            # "class mwAdditiveFDMLinkParams : measures::mwMeasurable"
            name_part = re.split(r"\s+", h2_text)[1]  # second word
            current_class = name_part.rstrip(":")
        elif h2_lower.startswith("enum "):
            name_part = re.split(r"\s+", h2_text)[1]
            current_class = name_part.rstrip(":")

    # ---- Find all member tables ----
    # Member tables are *leaf* <table style="table-layout:..."> elements that
    # contain an <h3>.  The outer container table also matches the style but
    # wraps every other table; we exclude it by only keeping tables that do
    # NOT contain further table-layout tables (len(nested) == 0).
    all_tables = content.find_all("table", style=re.compile(r"table-layout"))
    member_tables: list[Tag] = []
    for t in all_tables:
        nested = t.find_all("table", style=re.compile(r"table-layout"))
        if len(nested) > 0:
            continue  # skip wrapper / container tables
        h3 = t.find("h3")
        if h3 is None:
            continue
        sig = _clean_text(h3.get_text(separator=" ", strip=True))
        if not sig:
            continue
        if sig.lower().startswith("inherited"):
            continue
        member_tables.append(t)

    if not member_tables:
        return _parse_narrative(raw, path)

    # ---- Extract each member ----
    results: list[dict[str, Any]] = []
    for mt in member_tables:
        h3 = mt.find("h3")
        signature = _clean_text(h3.get_text(separator=" ", strip=True))
        if not signature:
            continue

        elem_type = "function"
        func_match = re.match(r"(?:#define\s+)?(?:enum\s+)?(~?\w+(?:\.\w+)*)", signature)
        func_name = func_match.group(1) if func_match else None

        if signature.startswith("#define"):
            elem_type = "macro"
        elif signature.lower().startswith("enum "):
            elem_type = "enum"
            # Extract the actual enum name
            enum_match = re.match(r"enum\s+(\w+)", signature, re.IGNORECASE)
            if enum_match:
                func_name = enum_match.group(1)
        # Constructor has same name as class but is still a function
        elif "(" in signature:
            elem_type = "function"
        elif func_name and func_name == current_class:
            elem_type = "class"

        desc_parts: list[str] = []
        param_rows: list[dict[str, str]] = []
        return_desc: str | None = None
        see_also: list[str] = []

        # Walk all content after h3 within this table
        found_h3 = False
        for child in mt.descendants:
            if child == h3:
                found_h3 = True
                continue
            if not found_h3:
                continue
            if not isinstance(child, Tag):
                continue

            if child.name == "p":
                text = _clean_text(child.get_text(separator=" ", strip=True))
                if not text:
                    continue
                ret_m = re.match(r"Returns?\s+(.+?)(?:\.\s|$)", text, re.IGNORECASE)
                if ret_m:
                    return_desc = _clean_text(ret_m.group(1))
                else:
                    desc_parts.append(text)

            # Parameter table: <table style="border:0">
            elif child.name == "table" and child.get("style") and "border:0" in child.get("style", ""):
                for tr in child.find_all("tr"):
                    tds = tr.find_all("td")
                    texts = [_clean_text(td.get_text(separator=" ", strip=True)) for td in tds]
                    texts = [t for t in texts if t]
                    if len(texts) >= 2:
                        param_rows.append({
                            "name": texts[0].rstrip(":"),
                            "type": "",
                            "desc": texts[1],
                        })

            # Links for see-also
            elif child.name == "a":
                link_text = _clean_text(child.get_text())
                if link_text and link_text not in see_also:
                    see_also.append(link_text)

        symbol_id = f"{current_class}::{func_name}" if current_class and func_name else (func_name or current_class)

        results.append(_make_element(
            elem_type=elem_type,
            symbol_id=symbol_id,
            class_name=current_class,
            function_name=func_name,
            signature=signature,
            params=param_rows,
            return_desc=return_desc,
            remarks=" ".join(desc_parts) if desc_parts else None,
            see_also=see_also,
            contains_code=False,
            path=path,
        ))

    return results


# ---------------------------------------------------------------------------
# 3. Legacy Doxygen (MachineWorks style)
# ---------------------------------------------------------------------------


def _parse_legacy_doxygen(raw: str, path: Path) -> list[dict[str, Any]]:
    """Parse legacy Doxygen HTML with table-based layout (MachineWorks).

    Uses regex-based extraction because the HTML is too malformed for BS.
    """
    title_text = ""
    title_m = re.search(r"<TITLE[^>]*>(.*?)</TITLE>", raw, re.IGNORECASE | re.DOTALL)
    if title_m:
        title_text = _clean_text(title_m.group(1))

    page_type, symbol_id = _parse_legacy_title(title_text)
    if page_type is None:
        return _parse_narrative(raw, path)

    # Extract sections by regex
    sections = _extract_legacy_sections(raw)

    # ---- Signature ----
    sig_raw = sections.get("signature_raw", "")
    signature = _extract_legacy_signature(sig_raw) if sig_raw else None

    # ---- Class / function name ----
    class_name: str | None = None
    function_name: str | None = None

    if page_type in ("function", "macro"):
        function_name = symbol_id
        if symbol_id and "::" in symbol_id:
            parts = symbol_id.rsplit("::", 1)
            class_name = parts[0]
            function_name = parts[1]
    elif page_type in ("class", "typedef", "enum"):
        class_name = symbol_id

    # ---- Parameters ----
    params = _extract_legacy_params(sections.get("params_raw", ""))

    # ---- Description (prefer Description over Synopsis) ----
    description = sections.get("description", "") or sections.get("synopsis", "")
    description = _strip_tags(description) if description else None

    # ---- Return value ----
    return_desc = _strip_tags(sections.get("return_value", "")) or None

    # ---- Code blocks ----
    code_text, has_code = _extract_code_from_raw(sections.get("example_raw", ""))

    # ---- See also ----
    see_also = _extract_legacy_see_also(sections.get("see_also_raw", ""))

    return [_make_element(
        elem_type=page_type,
        symbol_id=symbol_id,
        class_name=class_name,
        function_name=function_name,
        signature=signature,
        params=params,
        return_desc=return_desc,
        remarks=description,
        example=code_text,
        see_also=see_also,
        contains_code=has_code,
        path=path,
    )]


def _parse_legacy_title(title_text: str) -> tuple[str | None, str | None]:
    """Parse legacy Doxygen <TITLE> to (page_type, symbol_id)."""
    if not title_text:
        return None, None

    patterns: list[tuple[str, str]] = [
        ("Function:", "function"),
        ("Data type:", "class"),
    ]
    for prefix, ptype in patterns:
        if title_text.startswith(prefix):
            name = title_text[len(prefix):].strip().strip('"').strip()
            return ptype, name if name else None

    # Shader pages: "Shader (Category): \"name\""
    shader_m = re.match(r"Shader\s*\(([^)]+)\)\s*:\s*\"(.+?)\"", title_text)
    if shader_m:
        return "function", shader_m.group(2)

    return None, None


# Known section labels and their normalised keys
_LEGACY_LABELS: dict[str, str] = {
    "function": "signature_raw",
    "type": "signature_raw",
    "synopsis": "synopsis",
    "location": "location",
    "parameters": "params_raw",
    "arguments": "params_raw",
    "return value": "return_value",
    "description": "description",
    "example": "example_raw",
    "see also": "see_also_raw",
    "name": "name",
    "class": "class_label",
}


def _extract_legacy_sections(raw: str) -> dict[str, str]:
    """Split legacy HTML into sections keyed by label name.

    Each section starts at a ``<B>Label</B></TD>`` marker and ends at the
    next label marker (or EOF).
    """
    # Find all <B>KnownLabel</B> positions
    positions: list[tuple[int, int, str]] = []
    for label, key in _LEGACY_LABELS.items():
        pattern = r"<B>\s*" + re.escape(label) + r"\s*</B>"
        for m in re.finditer(pattern, raw, re.IGNORECASE):
            # Content starts after ``</B>``
            positions.append((m.end(), m.start(), key))

    if not positions:
        return {}

    # Sort by position
    positions.sort()

    sections: dict[str, str] = {}
    for i, (content_start, label_pos, key) in enumerate(positions):
        if i + 1 < len(positions):
            content_end = positions[i + 1][1]  # start of next label's <B>
        else:
            content_end = len(raw)

        chunk = raw[content_start:content_end]
        sections[key] = chunk

    return sections


def _extract_legacy_signature(sig_raw: str) -> str | None:
    """Build a clean signature from the Function/Type section raw HTML."""
    # Remove leading </TD><TD> junk
    sig_raw = re.sub(r"^\s*(</?TD[^>]*>)+\s*", "", sig_raw, flags=re.IGNORECASE)

    parts: list[str] = []

    # Text before the first <TABLE> or <BR> is the return type
    pre_table = re.split(r"<(?:TABLE|BR)\b", sig_raw, maxsplit=1, flags=re.IGNORECASE)[0]
    pre_table = _strip_tags(pre_table)
    if pre_table and not pre_table.startswith("<!--"):
        parts.append(pre_table)

    # Extract parameter cells from nested table rows
    nested_rows = re.findall(
        r"<TR[^>]*>(.*?)</TR>",
        sig_raw,
        re.IGNORECASE | re.DOTALL,
    )
    param_texts: list[str] = []
    for row_html in nested_rows:
        cells = re.findall(r"<TD[^>]*>(.*?)</TD>", row_html, re.IGNORECASE | re.DOTALL)
        cleaned = [_strip_tags(c) for c in cells]
        cleaned = [c for c in cleaned if c and c not in (" ", "")]
        row_text = " ".join(cleaned)
        if row_text:
            param_texts.append(row_text)

    if not param_texts and parts:
        return parts[0]

    if param_texts:
        func_open = param_texts[0]
        func_close = param_texts[-1] if len(param_texts) > 1 else ""
        middle = param_texts[1:-1] if len(param_texts) > 2 else []

        sig = func_open
        if middle:
            sig += " " + ", ".join(middle)
        if func_close and func_close != func_open:
            sig += " " + func_close

        if parts:
            sig = f"{parts[0]} {sig}"
    else:
        sig = ""

    result = _clean_text(sig)
    return result if result else None


def _extract_legacy_params(params_raw: str) -> list[dict[str, str]]:
    """Extract parameters from the legacy Parameters / Arguments section."""
    if not params_raw:
        return []

    param_rows = re.findall(
        r"<TR[^>]*>(.*?)</TR>",
        params_raw,
        re.IGNORECASE | re.DOTALL,
    )

    params: list[dict[str, str]] = []
    for row_html in param_rows:
        cells = re.findall(r"<TD[^>]*>(.*?)</TD>", row_html, re.IGNORECASE | re.DOTALL)
        cleaned = [_strip_tags(c).strip() for c in cells]
        cleaned = [c for c in cleaned if c]

        if not cleaned:
            continue

        # Skip header rows
        lower_first = cleaned[0].lower()
        if lower_first in ("name", "type", "default", ""):
            continue

        param: dict[str, str] = {"name": "", "type": "", "desc": ""}
        num_cells = len(cleaned)

        if num_cells == 1:
            param["desc"] = cleaned[0]
        elif num_cells == 2:
            # Try to detect TT content for type
            type_tt = re.search(r"<TT[^>]*>(.*?)</TT>", row_html, re.IGNORECASE | re.DOTALL)
            if type_tt:
                param["type"] = _clean_text(type_tt.group(1))
                param["desc"] = cleaned[1]
            else:
                param["type"] = cleaned[0]
                param["desc"] = cleaned[1]
        elif num_cells >= 3:
            param["type"] = cleaned[0]
            param["name"] = cleaned[1]
            param["desc"] = " ".join(cleaned[2:])

        if param["name"] or param["type"] or param["desc"]:
            params.append(param)

    return params


def _extract_legacy_see_also(see_also_raw: str) -> list[str]:
    """Extract referenced names from See Also links."""
    if not see_also_raw:
        return []

    links = re.findall(
        r"<A\s[^>]*?HREF\s*=\s*['\"][^'\"]+['\"][^>]*>(.*?)</A>",
        see_also_raw,
        re.IGNORECASE | re.DOTALL,
    )
    return [_strip_tags(link) for link in links if _strip_tags(link)]


# ---------------------------------------------------------------------------
# 4. Narrative / guide pages
# ---------------------------------------------------------------------------


def _parse_narrative(raw: str, path: Path) -> list[dict[str, Any]]:
    """Split a generic page into sections by h1/h2/h3 headings."""
    soup = BeautifulSoup(raw, "html.parser")
    body = soup.find("body")
    if body is None:
        body = soup

    # Extract code blocks globally
    pres = body.find_all("pre")
    all_code: list[str] = []
    has_code = False
    for pre in pres:
        text = _clean_text(pre.get_text())
        if text:
            all_code.append(text)
            has_code = True

    # Find headings
    headings = body.find_all(["h1", "h2", "h3"])
    if not headings:
        text = _clean_text(body.get_text(separator=" ", strip=True))
        if not text:
            return []
        return [_make_element(
            elem_type="narrative",
            remarks=text,
            example="\n".join(all_code) if all_code else None,
            contains_code=has_code,
            path=path,
        )]

    sections: list[dict[str, Any]] = []
    for heading in headings:
        section_title = _clean_text(heading.get_text(separator=" ", strip=True))
        if not section_title:
            continue

        content_parts: list[str] = []
        section_code: list[str] = []

        node = heading.next_sibling
        while node is not None:
            if isinstance(node, Tag) and node.name in ("h1", "h2", "h3"):
                break
            if isinstance(node, Tag):
                tag_text = _clean_text(node.get_text(separator=" ", strip=True))
                if tag_text:
                    content_parts.append(tag_text)
                for pre in node.find_all("pre"):
                    pre_text = _clean_text(pre.get_text())
                    if pre_text:
                        section_code.append(pre_text)
            elif hasattr(node, "strip"):
                stripped = node.strip()
                if stripped:
                    content_parts.append(stripped)
            node = node.next_sibling

        remarks = " ".join(content_parts)
        remarks = _clean_text(remarks)

        if not remarks and not section_code:
            continue

        sections.append(_make_element(
            elem_type="narrative",
            remarks=remarks if remarks else None,
            example="\n".join(section_code) if section_code else None,
            contains_code=bool(section_code),
            section_title=section_title,
            path=path,
        ))

    return sections


# ---------------------------------------------------------------------------
# Shared code / see-also extractors (BeautifulSoup-based)
# ---------------------------------------------------------------------------


def _extract_code_from_soup(element: Tag) -> tuple[str | None, bool]:
    """Extract <pre> blocks from a BS element, returning (joined_text, has_code)."""
    pres = element.find_all("pre")
    blocks: list[str] = []
    for pre in pres:
        text = _clean_text(pre.get_text())
        if text:
            blocks.append(text)

    if not blocks:
        return None, False

    return "\n".join(blocks), True


def _extract_see_also_from_soup(element: Tag) -> list[str]:
    """Extract see-also links from a BS element."""
    see_also: list[str] = []

    # Look for <dl class="section see">
    see_section = element.find("dl", class_="section see")
    if see_section:
        for a in see_section.find_all("a"):
            name = _clean_text(a.get_text())
            if name:
                see_also.append(name)

    return see_also


def _extract_code_from_raw(raw_html: str) -> tuple[str | None, bool]:
    """Extract <PRE> blocks from raw HTML (for legacy format)."""
    pres = re.findall(
        r"<PRE[^>]*>(.*?)</PRE>",
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    blocks: list[str] = []
    for pre in pres:
        text = _strip_tags(pre)
        if text:
            blocks.append(text)

    if not blocks:
        return None, False

    return "\n".join(blocks), True
