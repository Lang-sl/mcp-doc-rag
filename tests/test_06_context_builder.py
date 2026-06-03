"""Stage 6: Context block assembly.

No external dependencies. Verifies prompt-ready context formatting with token limits.

    pytest tests/test_06_context_builder.py -v
"""

from __future__ import annotations

from rag.models import Chunk, SearchResult
from rag.context_builder import build_context


class TestBuildContext:
    """Formatting search results into context blocks."""

    def test_basic_context(self):
        c1 = Chunk(
            chunk_id="1", type="function",
            symbol_id="A::b", class_name="A", function_name="b",
            signature="int b(float x)", remarks="Calculates b.",
            source_label="test", source_module="core", source_file="test.h",
        )
        c2 = Chunk(
            chunk_id="2", type="class",
            symbol_id="A", class_name="A",
            remarks="Class A for testing.",
            source_label="test", source_module="core", source_file="test.h",
        )

        results = [
            SearchResult(chunk=c1, score=0.95),
            SearchResult(chunk=c2, score=0.80),
        ]

        ctx = build_context(results, "what is A::b?", max_tokens=10000)
        assert "A::b" in ctx
        assert "Class A" in ctx
        assert "test" in ctx

    def test_token_cap_enforced(self):
        c1 = Chunk(
            chunk_id="1", type="function",
            symbol_id="A::b", class_name="A", function_name="b",
            remarks="X" * 5000,
            source_label="test", source_module="core", source_file="test.h",
        )

        results = [SearchResult(chunk=c1, score=0.95)]
        ctx = build_context(results, "what is A::b?", max_tokens=500)

        # 500 tokens ≈ 2000 chars, content is 5000 "X" chars
        # Should be truncated significantly
        assert len(ctx) < 5000

    def test_empty_results(self):
        ctx = build_context([], "query", max_tokens=1000)
        # Should return something (header at minimum), not crash
        assert isinstance(ctx, str)

    def test_context_sorted_by_score(self):
        """Higher-scored results should appear earlier in the output."""
        c1 = Chunk(
            chunk_id="high", type="function",
            symbol_id="Best::Match", class_name="Best", function_name="Match",
            remarks="This should appear first.",
            source_label="test", source_module="core", source_file="test.h",
        )
        c2 = Chunk(
            chunk_id="low", type="function",
            symbol_id="Worst::Match", class_name="Worst", function_name="Match",
            remarks="This should appear second.",
            source_label="test", source_module="core", source_file="test.h",
        )

        results = [
            SearchResult(chunk=c2, score=0.3),  # Lower score
            SearchResult(chunk=c1, score=0.9),  # Higher score
        ]
        # Context builder should sort by score desc
        ctx = build_context(results, "match", max_tokens=10000)
        pos_best = ctx.find("Best::Match")
        pos_worst = ctx.find("Worst::Match")
        assert pos_best >= 0 and pos_worst >= 0
        assert pos_best < pos_worst, "Higher scored result should appear first"
