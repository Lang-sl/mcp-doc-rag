from __future__ import annotations

import json
import re
from typing import Any


DOC_TOOL_NAMES = {
    "find_symbol",
    "search_docs",
    "get_api_class",
    "get_api_function",
    "list_modules",
    "build_context",
    "add_doc_source",
    "remove_doc_source",
    "list_doc_sources",
    "reindex",
    "index_status",
}

CODEGRAPH_SEARCH_TOOL = "codegraph_search"

_MARKDOWN_SYMBOL_HEADING = re.compile(r"^###\s+(.+?)\s+\([^)]+\)\s*$", re.MULTILINE)
_SYMBOL_FIELDS = {"symbol", "symbol_id", "symbol_name", "qualified_name"}
_NAME_SYMBOL_HINT_FIELDS = {
    "kind",
    "type",
    "signature",
    "location",
    "file",
    "file_path",
    "path",
    "qualified_name",
    "symbol",
    "symbol_id",
    "symbol_name",
}


def normalize_symbol_candidates(symbol: str) -> list[str]:
    candidates: list[str] = []
    stripped_symbol = symbol.strip()
    _append_unique(candidates, stripped_symbol)

    template_stripped = _strip_templates(stripped_symbol)
    _append_unique(candidates, template_stripped)

    final_member = template_stripped.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    _append_unique(candidates, final_member)

    return candidates


def extract_symbol_names(value: Any) -> list[str]:
    symbols: list[str] = []
    _extract_symbol_names(value, symbols)
    return symbols


def decode_codegraph_result(result: dict) -> Any:
    content = result.get("content")
    if isinstance(content, list):
        text_items = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        if text_items:
            text = "\n".join(text_items)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}

    return result


class GatewayTools:
    def __init__(self, doc_backend: Any, codegraph_client: Any | None = None) -> None:
        self.doc_backend = doc_backend
        self.codegraph_client = codegraph_client

    def smart_search(self, query: str, top_k: int = 10) -> dict:
        if not self._codegraph_search_available():
            return self._degraded_search(query, top_k, "CodeGraph unavailable")

        codegraph_result = self.codegraph_client.call_tool(CODEGRAPH_SEARCH_TOOL, {"query": query})
        if isinstance(codegraph_result, dict) and "error" in codegraph_result:
            return self._degraded_search(query, top_k, str(codegraph_result["error"]))

        code_usages = decode_codegraph_result(codegraph_result)
        code_symbols = extract_symbol_names(code_usages)
        matched_api_symbols: list[dict] = []
        unmatched_code_symbols: list[str] = []
        seen_matched_ids: set[str] = set()

        for code_symbol in code_symbols:
            match = self._find_api_symbol(code_symbol)
            if match is None:
                _append_unique(unmatched_code_symbols, code_symbol)
                continue

            match_key = str(match.get("symbol_id", match))
            if match_key not in seen_matched_ids:
                seen_matched_ids.add(match_key)
                matched_api_symbols.append(match)

        matched_queries = [
            str(symbol["symbol_id"])
            for symbol in matched_api_symbols
            if isinstance(symbol, dict) and "symbol_id" in symbol
        ]
        doc_query = " ".join(matched_queries) if matched_queries else query
        doc_results = self.doc_backend.search_docs(doc_query, top_k=top_k)

        return {
            "query": query,
            "degraded": False,
            "warnings": [],
            "code_usages": code_usages,
            "matched_api_symbols": matched_api_symbols,
            "unmatched_code_symbols": unmatched_code_symbols,
            "doc_results": doc_results,
        }

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        tool_arguments = arguments or {}
        if name == "smart_search":
            return self.smart_search(**tool_arguments)
        if name in DOC_TOOL_NAMES:
            return getattr(self.doc_backend, name)(**tool_arguments)
        if self.codegraph_client is not None and name in getattr(self.codegraph_client, "tool_names", []):
            return self.codegraph_client.call_tool(name, tool_arguments)
        raise KeyError(name)

    def _codegraph_search_available(self) -> bool:
        if self.codegraph_client is None:
            return False
        if not getattr(self.codegraph_client, "available", False):
            return False
        return CODEGRAPH_SEARCH_TOOL in getattr(self.codegraph_client, "tool_names", [])

    def _degraded_search(self, query: str, top_k: int, warning: str) -> dict:
        return {
            "query": query,
            "degraded": True,
            "warnings": [warning],
            "code_usages": [],
            "matched_api_symbols": [],
            "unmatched_code_symbols": [],
            "doc_results": self.doc_backend.search_docs(query, top_k=top_k),
        }

    def _find_api_symbol(self, code_symbol: str) -> dict | None:
        for candidate in normalize_symbol_candidates(code_symbol):
            if _is_unqualified_fallback(code_symbol, candidate):
                continue
            match = self.doc_backend.find_symbol(candidate)
            if match is not None:
                return match
        return None


def _extract_symbol_names(value: Any, symbols: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _SYMBOL_FIELDS and isinstance(item, str):
                _append_unique(symbols, item)
            elif key == "name" and isinstance(item, str) and _looks_like_symbol_object(value):
                _append_unique(symbols, item)
            _extract_symbol_names(item, symbols)
        return

    if isinstance(value, list):
        for item in value:
            _extract_symbol_names(item, symbols)
        return

    if isinstance(value, str):
        for match in _MARKDOWN_SYMBOL_HEADING.finditer(value):
            _append_unique(symbols, match.group(1).strip())


def _append_unique(values: list, value: Any) -> None:
    if value and value not in values:
        values.append(value)


def _strip_templates(symbol: str) -> str:
    result: list[str] = []
    depth = 0
    for char in symbol:
        if char == "<":
            depth += 1
            continue
        if char == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            result.append(char)
    return "".join(result)


def _is_unqualified_fallback(original: str, candidate: str) -> bool:
    if candidate == original:
        return False

    template_stripped = _strip_templates(original)
    if candidate == template_stripped:
        return False

    return "::" in template_stripped or "." in template_stripped


def _looks_like_symbol_object(value: dict) -> bool:
    if not _NAME_SYMBOL_HINT_FIELDS.intersection(value):
        return False

    name = value.get("name")
    return isinstance(name, str) and name.strip().lower() not in {"search result", "search results"}
