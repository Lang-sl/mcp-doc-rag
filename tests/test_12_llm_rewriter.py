"""Unit tests for LLMQueryRewriter. No Ollama needed — all calls mocked."""
from unittest.mock import patch, MagicMock

from rag.models import RewriteResult
from rag.retriever.query_rewriter import LLMQueryRewriter


def test_symbol_lookup_skipped():
    """C++ qualified names must never be rewritten by LLM."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    assert rewriter.rewrite("Namespace::Class::method") is None


def test_single_pascal_word_skipped():
    """Single PascalCase identifiers are treated as symbol lookups."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    assert rewriter.rewrite("Initialize") is None


def test_normal_query_returns_result():
    """A natural-language query should produce a RewriteResult when LLM responds."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    mock_result = RewriteResult(
        completed="How do I initialize the renderer?",
        sub_queries=[],
        variants=["setup renderer", "start rendering pipeline"],
    )
    with patch.object(rewriter, "_call_ollama", return_value=mock_result):
        result = rewriter.rewrite("how to init renderer")
        assert result is not None
        assert "initialize" in result.completed.lower()
        assert len(result.variants) >= 1


def test_malformed_json_fallback():
    """When _call_ollama returns None, rewrite() returns None (caller uses expand())."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    with patch.object(rewriter, "_call_ollama", return_value=None):
        result = rewriter.rewrite("how to init renderer")
        assert result is None


def test_connection_error_returns_none():
    """Any exception during Ollama call returns None (graceful fallback)."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    with patch.object(rewriter, "_call_ollama", side_effect=ConnectionError("refused")):
        result = rewriter.rewrite("how to init renderer")
        assert result is None


def test_sub_queries_parsed():
    """Complex queries should be decomposed into sub_queries."""
    rewriter = LLMQueryRewriter("http://localhost:11434", "qwen2.5:3b")
    mock_result = RewriteResult(
        completed="How to setup the renderer with 5-axis simulation?",
        sub_queries=["initialize renderer", "configure 5-axis simulation"],
        variants=[],
    )
    with patch.object(rewriter, "_call_ollama", return_value=mock_result):
        result = rewriter.rewrite("setup renderer with 5-axis simulation")
        assert result is not None
        assert len(result.sub_queries) == 2
        assert "initialize renderer" in result.sub_queries


def test_existing_expand_unchanged():
    """Verify the existing expand() function still works — no regression."""
    from rag.retriever.query_rewriter import expand

    # Symbol queries still skipped
    assert expand("MwMultiAxis::CalculateToolpath") == ["MwMultiAxis::CalculateToolpath"]
    # NL queries still expanded
    variants = expand("how to setup the renderer")
    assert len(variants) >= 2
    assert "how to setup the renderer" in variants
