"""Context builder — assembles search results into a prompt-ready context block."""
from __future__ import annotations

from rag.models import SearchResult


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: chars / 4."""
    return len(text) // 4


def build_context(
    results: list[SearchResult],
    query: str,
    max_tokens: int = 6000,
) -> str:
    """Build a formatted context block from search results, bounded by max_tokens.

    Results are grouped by source -> module for readability.
    Chunks are added in score order until the token cap is hit.
    """
    if not results:
        return f"[RAG] No relevant documentation found for query: \"{query}\""

    # Sort by score descending before grouping
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    # Group results by source -> module
    groups: dict[str, dict[str, list[SearchResult]]] = {}
    for r in sorted_results:
        source = r.chunk.source_label or "unknown"
        module = r.chunk.source_module or "root"
        if source not in groups:
            groups[source] = {}
        if module not in groups[source]:
            groups[source][module] = []
        groups[source][module].append(r)

    lines: list[str] = []
    lines.append(f"[RAG Context — Query: \"{query}\"]")
    lines.append("=" * 60)

    token_budget = max_tokens - _estimate_tokens("\n".join(lines))
    chunk_count = 0

    for source, modules in groups.items():
        source_header = True

        for module, module_results in modules.items():
            for r in module_results:
                # Build chunk entry
                entry_lines = []

                if source_header:
                    entry_lines.append(f"\n## Source: {source}")
                    source_header = False

                # Header with symbol info
                symbol_str = r.chunk.symbol_id or "(narrative)"
                type_str = r.chunk.type.upper()
                entry_lines.append(f"\n### [{type_str}] {symbol_str}")

                if r.chunk.signature:
                    entry_lines.append(f"```cpp\n{r.chunk.signature}\n```")

                # Source file citation
                entry_lines.append(f"*Source: {r.chunk.source_file} | Score: {r.score:.4f}*")

                # Content
                if r.chunk.remarks:
                    entry_lines.append(r.chunk.remarks)

                if r.chunk.example:
                    entry_lines.append(f"\n**Example:**\n```cpp\n{r.chunk.example}\n```")

                if r.chunk.see_also:
                    entry_lines.append(f"\n*See also: {', '.join(r.chunk.see_also)}*")

                entry_text = "\n".join(entry_lines)
                entry_tokens = _estimate_tokens(entry_text)

                if entry_tokens > token_budget:
                    # Truncate content to fit remaining budget
                    available_chars = token_budget * 4
                    entry_text = entry_text[:available_chars] + "\n[...truncated]"
                    lines.append(entry_text)
                    chunk_count += 1
                    token_budget = 0
                    break

                lines.append(entry_text)
                token_budget -= entry_tokens
                chunk_count += 1

            if token_budget <= 0:
                break

        if token_budget <= 0:
            break

    lines.append(f"\n---\n*{chunk_count} chunks included in context.*")

    return "\n".join(lines)
