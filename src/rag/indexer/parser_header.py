"""C++ header parser.

Extracts function signatures, class/struct declarations, and enums from C++
header files using regex patterns. Also extracts preceding comments as
descriptions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.indexer.parser_registry import register_parser


# ---------------------------------------------------------------------------
# Module-level regex patterns
# ---------------------------------------------------------------------------

FUNC_RE = re.compile(
    r'(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:const\s+)?'
    r'(?:[\w:]+(?:<[^>]*>)?[\s*&]+)+'
    r'(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?;'
)

CLASS_RE = re.compile(r'^\s*(?:class|struct)\s+(\w+)')

ENUM_RE = re.compile(r'^\s*enum\s+(?:class\s+)?(\w+)')

# Words that are never function names (language keywords / control-flow)
_SKIP_NAMES: frozenset[str] = frozenset({
    "if", "while", "for", "switch", "return", "catch", "throw",
    "sizeof", "decltype", "typedef", "using", "template", "namespace",
    "class", "struct", "enum", "union", "public", "private", "protected",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@register_parser(file_type="header", extensions=[".h", ".hpp", ".hxx"])
def parse_header(file_path: str, source_label: str, source_module: str) -> list[dict[str, Any]]:
    """Parse a C++ header file into a list of structured element dicts.

    Returns [] on errors, empty files, or files with no extractable content.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except (OSError, UnicodeDecodeError):
        return []

    if not raw.strip():
        return []

    lines = raw.splitlines()
    results: list[dict[str, Any]] = []

    # Track current class context so member functions get a qualified symbol_id
    current_class: str | None = None

    for idx, line in enumerate(lines):
        # ---- class / struct ----
        class_m = CLASS_RE.match(line)
        if class_m:
            class_name = class_m.group(1)
            rest = line[class_m.end():].strip()

            # Skip forward declarations: "class Foo;" has nothing but a ';' after the name
            if rest == ";":
                continue

            # When the first captured word is a macro (e.g. DLL_EXPORT), the real
            # class name follows it on the same line:  "class FOO_DLL_EXPORT Bar"
            if class_name.isupper() and "_" in class_name and rest:
                next_word = rest.split()[0].rstrip(";:")
                if next_word:
                    class_name = next_word

            current_class = class_name
            comments = _extract_comments_before(lines, idx)
            results.append(_make_element(
                elem_type="class",
                symbol_id=class_name,
                class_name=class_name,
                remarks=comments,
                path=Path(file_path),
                source_label=source_label,
                source_module=source_module,
            ))
            continue

        # ---- enum ----
        enum_m = ENUM_RE.match(line)
        if enum_m:
            enum_name = enum_m.group(1)
            comments = _extract_comments_before(lines, idx)
            symbol_id = f"{current_class}::{enum_name}" if current_class else enum_name
            results.append(_make_element(
                elem_type="enum",
                symbol_id=symbol_id,
                class_name=current_class,
                function_name=enum_name,
                remarks=comments,
                path=Path(file_path),
                source_label=source_label,
                source_module=source_module,
            ))
            continue

        # ---- function / method ----
        func_m = FUNC_RE.search(line)
        if func_m:
            func_name = func_m.group(1)
            if func_name in _SKIP_NAMES:
                continue

            params_str = func_m.group(2)
            full_sig = func_m.group(0).strip()
            params = _parse_params(params_str)
            comments = _extract_comments_before(lines, idx)

            class_name = current_class
            if class_name:
                symbol_id = f"{class_name}::{func_name}"
            else:
                symbol_id = func_name

            results.append(_make_element(
                elem_type="function",
                symbol_id=symbol_id,
                class_name=class_name,
                function_name=func_name,
                signature=full_sig,
                params=params,
                remarks=comments,
                path=Path(file_path),
                source_label=source_label,
                source_module=source_module,
            ))

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_comments_before(lines: list[str], idx: int) -> str | None:
    """Walk backwards up to 10 lines, collecting // and /* */ style comments.

    Stops at the first non-comment, non-empty line (or when 10 lines have been
    examined).  Doxygen markup (``@brief``, ``@param``, etc.) is stripped,
    keeping only the descriptive text.
    """
    parts: list[str] = []
    start = max(0, idx - 10)
    in_block = False

    for i in range(idx - 1, start - 1, -1):
        stripped = lines[i].strip()

        # ---- closing ``*/`` of a block comment ----
        if stripped.endswith("*/") and not in_block:
            in_block = True
            content = stripped[:-2].strip()
            if content.startswith("/*"):
                content = content[2:].strip()
            elif content.startswith("*"):
                content = content[1:].strip()
            if content:
                parts.insert(0, _strip_doxygen_tags(content))
            continue

        # ---- opening ``/*`` of a block comment ----
        if stripped.startswith("/*") and not stripped.endswith("*/"):
            content = stripped[2:].strip()
            if content.startswith("*"):
                content = content[1:].strip()
            if content:
                parts.insert(0, _strip_doxygen_tags(content))
            in_block = False
            continue

        # ---- single-line block comment ``/* ... */`` ----
        if stripped.startswith("/*") and stripped.endswith("*/"):
            content = stripped[2:-2].strip()
            if content.startswith("*"):
                content = content[1:].strip()
            if content:
                parts.insert(0, _strip_doxygen_tags(content))
            in_block = False
            continue

        # ---- inside a block comment ----
        if in_block:
            if stripped.startswith("*"):
                content = stripped[1:].strip()
            else:
                content = stripped
            if content.endswith("*/"):
                content = content[:-2].strip()
                in_block = False
            if content:
                parts.insert(0, _strip_doxygen_tags(content))
            continue

        # ---- line comment ``//`` ----
        if stripped.startswith("//"):
            content = stripped[2:].strip()
            if content:
                parts.insert(0, _strip_doxygen_tags(content))
            continue

        # ---- non-comment, non-empty line => stop ----
        if stripped:
            break

    if not parts:
        return None

    return " ".join(parts)


def _strip_doxygen_tags(text: str) -> str:
    """Remove leading doxygen/javadoc tags like ``@brief``, ``@param``, ``\\param``, etc."""
    text = text.strip()
    # Match @tag or \tag at the start
    m = re.match(r"[@\\](?:brief|param|return|returns|throws|tparam|see|note|warning|deprecated|since|author|version|date|todo|remark|remarks|sa)\b\s*", text)
    if m:
        return text[m.end():].strip()
    return text


def _parse_params(params_str: str) -> list[dict[str, str]]:
    """Parse a C++ parameter string into a list of ``{name, type, desc}`` dicts.

    Splits on commas while respecting angle-bracket nesting (templates).
    """
    if not params_str or params_str.strip() in ("", "void"):
        return []

    parts = _split_params(params_str)
    params: list[dict[str, str]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find all word-like tokens; the last one is the parameter name.
        candidates = re.findall(r"\b(\w+)\b", part)
        if not candidates:
            continue

        name = candidates[-1]
        name_pos = part.rfind(name)
        type_str = part[:name_pos].strip().rstrip("*& \t") if name_pos >= 0 else ""

        params.append({"name": name, "type": type_str, "desc": ""})

    return params


def _split_params(params_str: str) -> list[str]:
    """Split a parameter string by top-level commas (respects ``<...>`` nesting)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in params_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


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
    source_label: str = "",
    source_module: str = "",
) -> dict[str, Any]:
    """Build a standardised element dict (same shape across all parsers)."""
    see_also = see_also or []
    references = list(dict.fromkeys(see_also))
    if class_name and class_name not in references:
        references.append(class_name)

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
        "source_label": source_label,
        "source_module": source_module,
        "file_path": str(path) if path else "",
    }
