"""C++ header parser.

Extracts function signatures, class/struct declarations, enums, macros, and
typedefs from C++ header files.  Uses tree-sitter-cpp for AST-level parsing
when available; falls back to regex patterns otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.indexer.parser_registry import register_parser

# ---------------------------------------------------------------------------
# tree-sitter availability
# ---------------------------------------------------------------------------

try:
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser

    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False


# ---------------------------------------------------------------------------
# Module-level regex patterns (fallback)
# ---------------------------------------------------------------------------

FUNC_RE = re.compile(
    r"(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:const\s+)?"
    r"(?:[\w:]+(?:<[^>]*>)?[\s*&]+)+"
    r"(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?;"
)

CLASS_RE = re.compile(r"^\s*(?:class|struct)\s+(\w+)")

ENUM_RE = re.compile(r"^\s*enum\s+(?:class\s+)?(\w+)")

MACRO_RE = re.compile(r"^\s*#define\s+(\w+)(?:\([^)]*\))?\s+(.*)")

TYPEDEF_RE = re.compile(
    r"^\s*typedef\s+(.+?)\s+(\w+)\s*;"
)

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
def parse_header(
    file_path: str, source_label: str, source_module: str
) -> list[dict[str, Any]]:
    """Parse a C++ header file into a list of structured element dicts.

    Returns [] on errors, empty files, or files with no extractable content.
    """
    if _HAS_TREE_SITTER:
        return _parse_with_treesitter(file_path, source_label, source_module)
    else:
        return _parse_with_regex(file_path, source_label, source_module)


# ---------------------------------------------------------------------------
# Tree-sitter path
# ---------------------------------------------------------------------------

if _HAS_TREE_SITTER:

    def _parse_with_treesitter(
        file_path: str, source_label: str, source_module: str
    ) -> list[dict[str, Any]]:
        """Parse using tree-sitter-cpp AST."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            return []

        if not source.strip():
            return []

        try:
            lang = Language(tscpp.language())
            parser = Parser(lang)
            source_bytes = bytes(source, "utf-8")
            tree = parser.parse(source_bytes)
        except Exception:
            return _parse_with_regex(file_path, source_label, source_module)

        results: list[dict[str, Any]] = []
        _extract_elements(
            tree.root_node,
            source_bytes,
            results,
            source_label,
            source_module,
            file_path,
            None,
        )
        return results

    def _extract_elements(
        node: Any,
        source_bytes: bytes,
        results: list[dict[str, Any]],
        source_label: str,
        source_module: str,
        file_path: str,
        current_class: str | None,
    ) -> None:
        """Recursively walk the AST and extract API elements."""
        pending_comments: list[str] = []

        for child in node.children:
            if child.type == "comment":
                text = _extract_comment_text(child, source_bytes)
                if text:
                    pending_comments.append(text)
                continue

            comments = " ".join(pending_comments) if pending_comments else None

            if child.type == "class_specifier":
                class_name = _get_class_name_ts(child, source_bytes)
                if class_name and not _is_forward_decl_ts(child):
                    symbol_id = (
                        f"{current_class}::{class_name}"
                        if current_class
                        else class_name
                    )
                    results.append(
                        _make_element(
                            elem_type="class",
                            symbol_id=symbol_id,
                            class_name=class_name,
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                    # Recurse into class body for nested elements
                    for sub in child.children:
                        if sub.type == "field_declaration_list":
                            _extract_elements(
                                sub,
                                source_bytes,
                                results,
                                source_label,
                                source_module,
                                file_path,
                                class_name,
                            )
                pending_comments = []

            elif child.type == "enum_specifier":
                enum_name = _get_enum_name_ts(child, source_bytes)
                if enum_name:
                    symbol_id = (
                        f"{current_class}::{enum_name}"
                        if current_class
                        else enum_name
                    )
                    results.append(
                        _make_element(
                            elem_type="enum",
                            symbol_id=symbol_id,
                            class_name=current_class,
                            function_name=enum_name,
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                pending_comments = []

            elif child.type == "preproc_def":
                macro_name = _get_macro_name_ts(child, source_bytes)
                if macro_name:
                    macro_body = _get_macro_body_ts(child, source_bytes)
                    results.append(
                        _make_element(
                            elem_type="macro",
                            symbol_id=macro_name,
                            function_name=macro_name,
                            signature=macro_body,
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                pending_comments = []

            elif child.type in ("type_definition", "alias_declaration"):
                td = _get_typedef_info_ts(child, source_bytes)
                if td:
                    results.append(
                        _make_element(
                            elem_type="typedef",
                            symbol_id=td["name"],
                            function_name=td["name"],
                            signature=td["signature"],
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                pending_comments = []

            elif child.type == "template_declaration":
                # Template wraps the actual declaration/class — recurse into it
                _extract_elements(
                    child,
                    source_bytes,
                    results,
                    source_label,
                    source_module,
                    file_path,
                    current_class,
                )
                pending_comments = []

            elif child.type in ("declaration", "function_definition"):
                func_info = _get_func_info_ts(child, source_bytes, current_class)
                if func_info:
                    results.append(
                        _make_element(
                            elem_type="function",
                            symbol_id=func_info["symbol_id"],
                            class_name=func_info["class_name"],
                            function_name=func_info["function_name"],
                            signature=func_info["signature"],
                            params=func_info.get("params", []),
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                    # Recurse into function body for nested declarations
                    for sub in child.children:
                        if sub.type == "compound_statement":
                            _extract_elements(
                                sub,
                                source_bytes,
                                results,
                                source_label,
                                source_module,
                                file_path,
                                current_class,
                            )
                pending_comments = []

            elif child.type == "field_declaration":
                # A field_declaration can contain a function_declarator (method)
                # or a class_specifier (nested class).  Try function extraction
                # first; if that yields nothing, recurse into non-function
                # children (e.g. nested class_specifier).
                func_info = _get_func_info_ts(child, source_bytes, current_class)
                if func_info:
                    results.append(
                        _make_element(
                            elem_type="function",
                            symbol_id=func_info["symbol_id"],
                            class_name=func_info["class_name"],
                            function_name=func_info["function_name"],
                            signature=func_info["signature"],
                            params=func_info.get("params", []),
                            remarks=comments,
                            path=Path(file_path),
                            source_label=source_label,
                            source_module=source_module,
                        )
                    )
                    pending_comments = []
                else:
                    # Not a function — recurse to find nested class/enum/typedef
                    _extract_elements(
                        child,
                        source_bytes,
                        results,
                        source_label,
                        source_module,
                        file_path,
                        current_class,
                    )
                    pending_comments = []

            elif child.type in ("field_declaration_list", "declaration_list"):
                # Class body or extern "C" block body — recurse
                _extract_elements(
                    child,
                    source_bytes,
                    results,
                    source_label,
                    source_module,
                    file_path,
                    current_class,
                )

            elif child.type == "linkage_specification":
                # extern "C" { ... } — recurse
                _extract_elements(
                    child,
                    source_bytes,
                    results,
                    source_label,
                    source_module,
                    file_path,
                    current_class,
                )

    # -- tree-sitter helpers ------------------------------------------------

    def _extract_comment_text(comment_node: Any, source_bytes: bytes) -> str | None:
        """Extract clean text from a comment node, stripping markup."""
        text = comment_node.text.decode("utf-8", errors="replace").strip()
        if text.startswith("//"):
            text = text[2:].strip()
        elif text.startswith("/*") and text.endswith("*/"):
            # Multi-line block comment
            inner = text[2:-2]
            lines = []
            for line in inner.splitlines():
                stripped = line.strip()
                if stripped.startswith("*"):
                    stripped = stripped[1:].strip()
                if stripped:
                    lines.append(stripped)
            text = " ".join(lines)
        text = _strip_doxygen_tags(text)
        return text or None

    def _find_child_ts(node: Any, child_type: str) -> Any | None:
        """Find first direct child of *node* with the given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_in_subtree(node: Any, child_type: str) -> Any | None:
        """Find first node of *child_type* anywhere in *node*'s subtree."""
        if node.type == child_type:
            return node
        for child in node.children:
            found = _find_in_subtree(child, child_type)
            if found is not None:
                return found
        return None

    def _get_class_name_ts(class_node: Any, source_bytes: bytes) -> str | None:
        """Extract class name, handling DLL_EXPORT macros."""
        type_id = _find_child_ts(class_node, "type_identifier")
        if type_id is None:
            return None

        name = type_id.text.decode("utf-8", errors="replace").strip()
        # If the captured name is a macro (all caps with underscores), look for the
        # real class name in a following init_declarator
        if name.isupper() and "_" in name:
            for child in class_node.parent.children:
                if child.type == "init_declarator":
                    ident = _find_child_ts(child, "identifier")
                    if ident is not None:
                        return ident.text.decode("utf-8", errors="replace").strip()
        return name

    def _is_forward_decl_ts(class_node: Any) -> bool:
        """Check if a class_specifier is a forward declaration (no body)."""
        for child in class_node.children:
            if child.type == "field_declaration_list":
                return False
        return True

    def _get_enum_name_ts(enum_node: Any, source_bytes: bytes) -> str | None:
        """Extract enum name from an enum_specifier node."""
        type_id = _find_child_ts(enum_node, "type_identifier")
        if type_id is None:
            return None
        return type_id.text.decode("utf-8", errors="replace").strip()

    def _get_macro_name_ts(macro_node: Any, source_bytes: bytes) -> str | None:
        """Extract macro name from a preproc_def node."""
        ident = _find_child_ts(macro_node, "identifier")
        if ident is None:
            return None
        name = ident.text.decode("utf-8", errors="replace").strip()
        if name in ("if", "ifdef", "ifndef", "else", "endif", "include", "pragma"):
            return None
        return name

    def _get_macro_body_ts(macro_node: Any, source_bytes: bytes) -> str | None:
        """Extract macro body (everything after the name)."""
        text = macro_node.text.decode("utf-8", errors="replace").strip()
        # Strip #define prefix and name
        body = text[len("#define"):].strip()
        # Strip the macro name (may be function-like with params)
        ident = _find_child_ts(macro_node, "identifier")
        if ident is not None:
            name = ident.text.decode("utf-8", errors="replace")
            idx = body.find(name)
            if idx >= 0:
                body = body[idx + len(name):].strip()
        # Clean up line continuation
        body = body.replace("\\\n", " ").replace("\\\r\n", " ").strip()
        # Truncate to reasonable length
        if len(body) > 300:
            body = body[:300]
        return body or None

    def _get_typedef_info_ts(
        node: Any, source_bytes: bytes
    ) -> dict[str, str] | None:
        """Extract typedef/using info from type_definition or alias_declaration."""
        # For alias_declaration (using Foo = Bar)
        if node.type == "alias_declaration":
            type_id = _find_child_ts(node, "type_identifier")
            if type_id is not None:
                name = type_id.text.decode("utf-8", errors="replace").strip()
                sig = node.text.decode("utf-8", errors="replace").strip()
                return {"name": name, "signature": sig[:300]}

        # For type_definition (typedef ... name)
        name = None
        # The name is typically the last type_identifier or primitive_type
        for child in reversed(node.children):
            if child.type in ("type_identifier", "primitive_type"):
                name = child.text.decode("utf-8", errors="replace").strip()
                break

        if name is None:
            return None

        sig = node.text.decode("utf-8", errors="replace").strip()
        return {"name": name, "signature": sig[:300]}

    def _get_func_info_ts(
        node: Any, source_bytes: bytes, current_class: str | None
    ) -> dict[str, Any] | None:
        """Extract function details from a declaration or function_definition node.

        Returns a dict with symbol_id, class_name, function_name, signature,
        and params, or None if the node does not contain a function declarator.
        """
        func_decl = _find_in_subtree(node, "function_declarator")
        if func_decl is None:
            # Check for init_declarator wrapping a function_declarator (pure virtual)
            init_decl = _find_in_subtree(node, "init_declarator")
            if init_decl is not None:
                func_decl = _find_in_subtree(init_decl, "function_declarator")
        if func_decl is None:
            return None

        func_name = _extract_func_name_ts(func_decl, source_bytes)
        if func_name is None or func_name in _SKIP_NAMES:
            return None

        class_name = current_class
        if class_name:
            symbol_id = f"{class_name}::{func_name}"
        else:
            symbol_id = func_name

        full_sig = node.text.decode("utf-8", errors="replace").strip()
        # Clean up: remove trailing newlines and trim
        full_sig = full_sig.replace("\n", " ").replace("\r", " ").strip()
        if len(full_sig) > 500:
            full_sig = full_sig[:500]

        params = _extract_params_ts(func_decl, source_bytes)

        return {
            "symbol_id": symbol_id,
            "class_name": class_name,
            "function_name": func_name,
            "signature": full_sig,
            "params": params,
        }

    def _extract_func_name_ts(
        func_decl: Any, source_bytes: bytes
    ) -> str | None:
        """Extract function name from a function_declarator node."""
        for child in func_decl.children:
            if child.type in ("identifier", "field_identifier"):
                return child.text.decode("utf-8", errors="replace").strip()
            if child.type == "operator_name":
                return child.text.decode("utf-8", errors="replace").strip()
            if child.type == "destructor_name":
                return child.text.decode("utf-8", errors="replace").strip()
        return None

    def _extract_params_ts(
        func_decl: Any, source_bytes: bytes
    ) -> list[dict[str, str]]:
        """Extract parameter info from a function_declarator node."""
        param_list = _find_in_subtree(func_decl, "parameter_list")
        if param_list is None:
            return []

        params: list[dict[str, str]] = []
        for child in param_list.children:
            if child.type == "parameter_declaration":
                text = child.text.decode("utf-8", errors="replace").strip()
                # Try to extract name (last identifier in the parameter)
                identifiers = _collect_identifiers(child)
                if identifiers:
                    name = identifiers[-1]
                    # Build type string from the text before the name
                    name_pos = text.rfind(name)
                    type_str = (
                        text[:name_pos].strip().rstrip("*& \t")
                        if name_pos >= 0
                        else ""
                    )
                else:
                    name = ""
                    type_str = text
                params.append({"name": name, "type": type_str, "desc": ""})
            elif child.type == "optional_parameter_declaration":
                text = child.text.decode("utf-8", errors="replace").strip()
                params.append({"name": "", "type": text, "desc": ""})

        return params

    def _collect_identifiers(node: Any) -> list[str]:
        """Collect all identifier/field_identifier names in a subtree.

        Used to extract parameter names from parameter declarations.  Only
        leaf identifiers (those whose children don't include further
        identifiers) are collected to avoid picking up type names that
        happen to be identifiers (e.g. ``int`` in ``int x`` has no child
        identifiers, but ``std::string`` would).
        """
        result: list[str] = []
        if node.type in ("identifier", "field_identifier"):
            text = node.text.decode("utf-8", errors="replace").strip()
            if text:
                result.append(text)
        for child in node.children:
            result.extend(_collect_identifiers(child))
        return result


# ---------------------------------------------------------------------------
# Regex fallback path
# ---------------------------------------------------------------------------

def _parse_with_regex(
    file_path: str, source_label: str, source_module: str
) -> list[dict[str, Any]]:
    """Parse a C++ header file using regex patterns (fallback)."""
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
        # ---- macro ----
        macro_m = MACRO_RE.match(line)
        if macro_m:
            macro_name = macro_m.group(1)
            if macro_name not in ("if", "ifdef", "ifndef", "else", "endif",
                                  "include", "pragma", "error", "warning"):
                macro_body = macro_m.group(2).strip()
                comments = _extract_comments_before(lines, idx)
                results.append(_make_element(
                    elem_type="macro",
                    symbol_id=macro_name,
                    function_name=macro_name,
                    signature=macro_body[:300] if macro_body else None,
                    remarks=comments,
                    path=Path(file_path),
                    source_label=source_label,
                    source_module=source_module,
                ))
            continue

        # ---- typedef ----
        typedef_m = TYPEDEF_RE.match(line)
        if typedef_m:
            typedef_name = typedef_m.group(2)
            typedef_sig = typedef_m.group(0).strip()
            comments = _extract_comments_before(lines, idx)
            results.append(_make_element(
                elem_type="typedef",
                symbol_id=typedef_name,
                function_name=typedef_name,
                signature=typedef_sig[:300],
                remarks=comments,
                path=Path(file_path),
                source_label=source_label,
                source_module=source_module,
            ))
            continue

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
# Helpers (shared between both paths)
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
