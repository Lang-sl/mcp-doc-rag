"""Rule-based query expansion using domain-specific synonym tables.

Generates query variants for natural-language searches to improve BM25
recall.  Symbol/API queries (containing ``::``) and single-word queries
are never expanded — they pass through unchanged.
"""

from __future__ import annotations


# SDK domain synonym table.  Keys are canonical tokens; values are lists
# of known equivalents users might express the concept with.
_SYNONYMS: dict[str, list[str]] = {
    "setup":    ["initialize", "configure", "create", "register"],
    "connect":  ["attach", "bind", "open", "transport"],
    "error":    ["status", "result code", "exception", "error code"],
    "config":   ["settings", "parameters", "options", "properties"],
    "create":   ["instantiate", "construct", "allocate", "new"],
    "run":      ["execute", "start", "launch", "process"],
    "stop":     ["halt", "terminate", "shutdown", "destroy"],
    "get":      ["retrieve", "query", "read", "access", "find"],
    "set":      ["assign", "write", "update", "modify"],
    "toolpath": ["tool path", "toolpath calculation", "cutting path"],
    "axis":     ["multi axis", "5 axis", "rotary axis"],
    "render":   ["rendering", "display", "draw", "viewport"],
    "simulate": ["simulation", "emulate", "preview"],
    "transform": ["transformation", "matrix", "coordinate"],
}


def expand(query: str, max_variants: int = 3) -> list[str]:
    """Generate query variants by substituting known synonyms.

    The original query is always the first element.  Symbol/API lookups
    (queries containing ``::`` or single PascalCase identifiers) and
    single-word queries are never expanded.

    Args:
        query: The original user query string.
        max_variants: Maximum number of *additional* variants (not counting
            the original).  Defaults to 3.

    Returns:
        A list of query strings with the original first, followed by up
        to *max_variants* synonym variants.
    """
    if not query:
        return [query]

    # Symbol / API queries are never expanded
    if "::" in query:
        return [query]

    tokens = query.lower().split()

    # Single-word queries (especially PascalCase identifiers) are not expanded
    if len(tokens) == 1:
        return [query]

    variants = [query]
    seen = {query}

    for word, synonyms in _SYNONYMS.items():
        if word not in tokens:
            continue

        for syn in synonyms[:2]:
            variant = query.lower().replace(word, syn)
            if variant not in seen:
                variants.append(variant)
                seen.add(variant)
                if len(variants) >= max_variants + 1:
                    return variants[:max_variants + 1]

    return variants[:max_variants + 1]


from rag.models import RewriteResult as _RewriteResult  # noqa: E402


class LLMQueryRewriter:
    """LLM-based query rewriter using Ollama chat API.

    Uses a local small model (e.g. qwen2.5:3b) to complete, decompose,
    and generate semantic variants of natural-language queries.
    Falls back to ``expand()`` on any failure.
    """

    _SYSTEM_PROMPT = (
        "You are a query rewriter for a C++ SDK documentation search engine.\n"
        "Given a user query, output JSON only:\n"
        '{"completed": "<polished, complete English question>",'
        '"sub_queries": ["<sub query 1>", ...],'
        '"variants": ["<semantic variant 1>", ...]}\n\n'
        "Rules:\n"
        '- "completed": fix typos, expand abbreviations, make the query a complete sentence\n'
        '- "sub_queries": for complex questions, break into 2-3 single-step queries. '
        "Empty list for simple questions.\n"
        '- "variants": 1-3 different ways to ask the same thing (synonyms, alternative phrasing)\n'
        "- Keep technical terms (class names, function names) unchanged\n"
        "- Output ONLY the JSON, no markdown, no explanation"
    )

    def __init__(self, host: str, model: str, timeout_ms: int = 2000):
        self._host = host
        self._model = model
        self._timeout = timeout_ms

    def rewrite(self, query: str) -> _RewriteResult | None:
        """Main entry. Returns None on any failure → caller uses ``expand()``."""
        if self._is_symbol_lookup(query):
            return None
        try:
            return self._call_ollama(query)
        except Exception:
            return None

    @staticmethod
    def _is_symbol_lookup(query: str) -> bool:
        """Symbol/API queries are never rewritten."""
        q = query.strip()
        if "::" in q:
            return True
        if " " not in q:
            stripped = q.strip("()<>*&[]")
            if stripped and (stripped[0].isupper() or "_" in stripped):
                return True
        return False

    def _call_ollama(self, query: str) -> _RewriteResult:
        """Call Ollama chat API, parse JSON response."""
        import json
        import ollama

        client = ollama.Client(host=self._host)
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1},
        )
        raw = response.get("message", {}).get("content", "").strip()

        # Try direct JSON parse first
        data = self._parse_json(raw)
        if data is None:
            return None

        return _RewriteResult(
            completed=data.get("completed", query),
            sub_queries=data.get("sub_queries", []) or [],
            variants=data.get("variants", []) or [],
        )

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """Parse JSON from LLM output. Tries direct parse, then regex extraction."""
        import json
        import re

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract a JSON object from the text
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None
