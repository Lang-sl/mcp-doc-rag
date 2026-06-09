"""Unit tests for the rule-based query rewriter."""

from rag.retriever.query_rewriter import expand


def test_symbol_query_not_expanded():
    """C++ qualified names must never be expanded."""
    assert expand("MwMultiAxis::CalculateToolpath") == ["MwMultiAxis::CalculateToolpath"]


def test_single_word_not_expanded():
    """Single-word queries (likely identifiers) are not expanded."""
    assert expand("Initialize") == ["Initialize"]


def test_nl_query_expanded():
    """Natural language queries with known synonyms produce variants."""
    variants = expand("how to setup the renderer")
    assert len(variants) >= 2
    assert "how to setup the renderer" in variants


def test_max_variants_respected():
    """The number of variants never exceeds max_variants + 1."""
    variants = expand("how to setup and configure the renderer", max_variants=2)
    assert len(variants) <= 3  # original + at most 2 variants


def test_no_duplicate_variants():
    """All returned variants must be unique."""
    variants = expand("setup the setup")
    assert len(variants) == len(set(variants))


def test_empty_query():
    """Empty query is passed through unchanged."""
    assert expand("") == [""]


def test_no_synonym_match():
    """Queries without known synonyms return only the original."""
    variants = expand("gibberish nonsense text")
    assert variants == ["gibberish nonsense text"]


def test_original_always_first():
    """The original query must always be the first element."""
    variants = expand("how to setup the renderer")
    assert variants[0] == "how to setup the renderer"


def test_expand_still_importable():
    """LLMQueryRewriter addition must not break expand() import."""
    from rag.retriever.query_rewriter import expand
    result = expand("how to setup")
    assert len(result) >= 1
    assert result[0] == "how to setup"
