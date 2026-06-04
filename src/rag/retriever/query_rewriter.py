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
