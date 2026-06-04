"""Stage 9: Search pipeline.

Needs: Ollama running + indexed data in ChromaDB.
Verifies vector search, BM25 search, hybrid search, and symbol lookup.

    pytest tests/test_09_search.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

from rag.config import load_config
from rag.models import Chunk, SearchResult
from tests.conftest import requires_ollama, is_ollama_available


def _get_config():
    """Load config from env var or default location."""
    return load_config(os.environ.get("RAG_CONFIG_PATH"))


def _has_indexed_data():
    """Check if ChromaDB has searchable data."""
    try:
        import chromadb

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)
        collections = client.list_collections()
        if not collections:
            return False

        for c in collections:
            try:
                coll = client.get_collection(name=c.name)
                if coll.count() > 0:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _get_first_source_label():
    """Return the first available source label from the index."""
    try:
        import chromadb

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)
        for c in client.list_collections():
            name = c.name
            if "." in name:
                return name.split(".")[0]
        return None
    except Exception:
        return None


requires_search_data = pytest.mark.skipif(
    not _has_indexed_data(),
    reason="No indexed data in ChromaDB — run 'python -m rag reindex' first",
)

requires_ollama_and_data = pytest.mark.skipif(
    not (is_ollama_available() and _has_indexed_data()),
    reason="Needs both Ollama running and indexed data in ChromaDB",
)


@requires_ollama_and_data
class TestVectorSearch:
    """ANN vector search across ChromaDB collections."""

    def test_vector_search_returns_results(self):
        from rag.retriever.vector_search import vector_search
        from rag.indexer.embedder import Embedder
        import chromadb

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        results = vector_search(client, embedder, config, "initialize", 10)
        assert len(results) > 0, "Vector search should return results"

    def test_vector_search_respects_top_k(self):
        from rag.retriever.vector_search import vector_search
        from rag.indexer.embedder import Embedder
        import chromadb

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        results = vector_search(client, embedder, config, "function", 3)
        assert len(results) <= 3

    def test_vector_search_source_filter(self):
        from rag.retriever.vector_search import vector_search
        from rag.indexer.embedder import Embedder
        import chromadb

        source_label = _get_first_source_label()
        if not source_label:
            pytest.skip("No source labels found in index")

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        results = vector_search(
            client, embedder, config, "function", 5, source_label=source_label
        )
        assert len(results) > 0, f"Source-filtered search for '{source_label}' returned no results"


@requires_ollama_and_data
class TestBM25Search:
    """Field-weighted BM25 keyword search."""

    def test_bm25_returns_results(self):
        from rag.retriever.bm25_search import bm25_search
        import chromadb

        config = _get_config()
        client = chromadb.PersistentClient(path=config.chroma_dir)

        results = bm25_search(client, config, "initialize", 10)
        assert len(results) > 0, "BM25 should return results"


@requires_ollama_and_data
class TestHybridSearch:
    """Full hybrid retrieval pipeline."""

    def test_hybrid_search_returns_results(self):
        from rag.retriever.hybrid import HybridRetriever

        config = _get_config()
        retriever = HybridRetriever(config)

        results = retriever.search("initialize", top_k=5)
        assert len(results) > 0

    def test_hybrid_search_scores_are_ordered(self):
        from rag.retriever.hybrid import HybridRetriever

        config = _get_config()
        retriever = HybridRetriever(config)

        results = retriever.search("function initialize", top_k=10)
        assert len(results) >= 2, "Need at least 2 results to verify ordering"
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, (
                "Results should be sorted descending by score"
            )

    def test_hybrid_search_module_filter(self):
        from rag.retriever.hybrid import HybridRetriever
        import chromadb

        source_label = _get_first_source_label()
        if not source_label:
            pytest.skip("No source labels found in index")

        config = _get_config()
        # Find a module under this source (non-empty collections only)
        client = chromadb.PersistentClient(path=config.chroma_dir)
        prefix = f"{source_label}."
        modules = [
            c.name[len(prefix):]
            for c in client.list_collections()
            if c.name.startswith(prefix) and c.count() > 0
        ]
        if not modules:
            pytest.skip(f"No modules found for source '{source_label}'")

        # Pick a search term from actual chunk text in this module
        coll = client.get_collection(f"{source_label}.{modules[0]}")
        sample = coll.get(limit=1)
        query_term = "function"
        if sample["documents"]:
            # Grab the first 2-3 words from the chunk's text as a query
            words = sample["documents"][0].split()
            if words:
                query_term = " ".join(words[:3])

        retriever = HybridRetriever(config)
        results = retriever.search(
            query_term, top_k=5, source_label=source_label, module=modules[0]
        )
        assert len(results) > 0, (
            f"No results for query='{query_term}' in module {modules[0]}"
        )
        for r in results:
            assert r.chunk.source_module == modules[0]
            assert r.chunk.source_label == source_label


@requires_search_data
class TestSymbolLookup:
    """O(1) symbol index lookup (doesn't need Ollama)."""

    def test_find_existing_symbol(self):
        from rag.symbol_index import SymbolIndex

        config = _get_config()
        idx = SymbolIndex(config.symbol_index_path)
        if len(idx) == 0:
            pytest.skip("Symbol index is empty — run 'python -m rag reindex' first")

        # Try the first available symbol
        result = None
        for sym, info in idx._index.items():
            result = info
            break

        assert result is not None, (
            "Symbol index has entries but couldn't retrieve any. "
            "Check symbol_index.json integrity."
        )


# ---------------------------------------------------------------------------
# RRF Weighted Fusion — unit tests (no external deps)
# ---------------------------------------------------------------------------


def _make_test_chunk(chunk_id: str, symbol_id: str):
    """Create a minimal Chunk for search testing."""
    from rag.models import Chunk

    return Chunk(
        chunk_id=chunk_id,
        type="function",
        symbol_id=symbol_id,
        source_label="test",
        source_module="core",
        source_file="test.h",
        embed_text=f"text for {chunk_id}",
    )


class TestRRFWeighting:
    """BM25-Vector weighted RRF fusion."""

    def test_equal_weight_same_score(self):
        """bm25_weight=1.0: BM25 and vector at same rank get same RRF score."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        c2 = _make_test_chunk("b", "Bar")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c2, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10, bm25_weight=1.0)
        scores = {r.chunk.chunk_id: r.score for r in fused}
        assert abs(scores["a"] - scores["b"]) < 0.0001

    def test_2x_weight_bm25_ranks_higher(self):
        """bm25_weight=2.0: BM25 result outranks vector."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        c2 = _make_test_chunk("b", "Bar")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c2, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10, bm25_weight=2.0)
        assert fused[0].chunk.chunk_id == "a"

    def test_3x_weight_gives_3x_score_ratio(self):
        """bm25_weight=3.0: BM25 score is exactly 3x vector score."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        c2 = _make_test_chunk("b", "Bar")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c2, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10, bm25_weight=3.0)
        scores = {r.chunk.chunk_id: r.score for r in fused}
        assert abs(scores["a"] / scores["b"] - 3.0) < 0.0001

    def test_below_1_weight_deprioritizes_bm25(self):
        """bm25_weight=0.5: vector ranks above BM25."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        c2 = _make_test_chunk("b", "Bar")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c2, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10, bm25_weight=0.5)
        assert fused[0].chunk.chunk_id == "b"

    def test_overlapping_chunks_sum_scores(self):
        """Same chunk in both lists: scores are summed."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c1, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10, bm25_weight=2.0)
        assert len(fused) == 1
        expected = 3.0 / 31.0  # 1/31 + 2/31
        assert abs(fused[0].score - expected) < 0.0001

    def test_empty_inputs(self):
        """Empty lists produce empty result."""
        from rag.retriever.hybrid import _rrf_fuse

        fused = _rrf_fuse([], [], k=30, max_results=10, bm25_weight=2.0)
        assert fused == []

    def test_only_vector_results(self):
        """Only vector results: pass through unchanged."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        vec = [SearchResult(chunk=c1, score=0.9)]

        fused = _rrf_fuse(vec, [], k=30, max_results=10, bm25_weight=2.0)
        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "a"

    def test_only_bm25_results(self):
        """Only BM25 results: pass through unchanged (with weight applied)."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        bm25 = [SearchResult(chunk=c1, score=10.0)]

        fused = _rrf_fuse([], bm25, k=30, max_results=10, bm25_weight=2.0)
        assert len(fused) == 1
        assert abs(fused[0].score - 2.0 / 31.0) < 0.0001

    def test_backward_compatible_default(self):
        """Default (no bm25_weight arg) gives equal weight = original behavior."""
        from rag.retriever.hybrid import _rrf_fuse

        c1 = _make_test_chunk("a", "Foo")
        c2 = _make_test_chunk("b", "Bar")

        bm25 = [SearchResult(chunk=c1, score=10.0)]
        vec = [SearchResult(chunk=c2, score=0.9)]

        fused = _rrf_fuse(vec, bm25, k=30, max_results=10)
        scores = {r.chunk.chunk_id: r.score for r in fused}
        assert abs(scores["a"] - scores["b"]) < 0.0001


# ---------------------------------------------------------------------------
# BM25 Disk Persistence — unit tests (no external deps)
# ---------------------------------------------------------------------------


class TestBM25Persistence:
    """BM25 disk save/load for fast process restart."""

    def test_save_and_load_roundtrip(self):
        from rag.retriever.bm25_search import BM25Searcher

        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "bm25")
            searcher = BM25Searcher(cache_dir=cache_dir)

            searcher._cache["test.coll"] = {
                "chunks": [
                    {"id": "1", "metadata": {"symbol_id": "Foo"}, "document": "d1"},
                    {"id": "2", "metadata": {"symbol_id": "Bar"}, "document": "d2"},
                ],
                "symbol_corpus": [["foo"], ["bar"]],
                "signature_corpus": [["void"], ["int"]],
                "remarks_corpus": [["d1"], ["d2"]],
                "example_corpus": [[], []],
            }
            searcher._counts["test.coll"] = 2
            searcher.save_to_disk("test.coll")

            assert os.path.isfile(os.path.join(cache_dir, "test.coll.pkl"))

            searcher2 = BM25Searcher(cache_dir=cache_dir)
            loaded = searcher2.load_from_disk("test.coll", 2)
            assert loaded
            assert "test.coll" in searcher2._cache
            assert searcher2._counts["test.coll"] == 2

    def test_stale_count_rejected(self):
        from rag.retriever.bm25_search import BM25Searcher

        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "bm25")
            searcher = BM25Searcher(cache_dir=cache_dir)

            searcher._cache["col"] = {
                "chunks": [{"id": "1", "metadata": {}, "document": "doc"}],
                "symbol_corpus": [["x"]],
                "signature_corpus": [["y"]],
                "remarks_corpus": [["z"]],
                "example_corpus": [[]],
            }
            searcher._counts["col"] = 1
            searcher.save_to_disk("col")

            searcher2 = BM25Searcher(cache_dir=cache_dir)
            assert not searcher2.load_from_disk("col", 99)

    def test_missing_file_returns_false(self):
        from rag.retriever.bm25_search import BM25Searcher

        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "bm25")
            searcher = BM25Searcher(cache_dir=cache_dir)
            assert not searcher.load_from_disk("nonexistent", 0)

    def test_no_cache_dir_skip_save(self):
        from rag.retriever.bm25_search import BM25Searcher

        searcher = BM25Searcher()
        searcher._cache["col"] = {
            "chunks": [{"id": "1", "metadata": {}, "document": "doc"}],
            "symbol_corpus": [["x"]],
            "signature_corpus": [["y"]],
            "remarks_corpus": [["z"]],
            "example_corpus": [[]],
        }
        searcher._counts["col"] = 1
        searcher.save_to_disk("col")  # should not raise

    def test_no_cache_dir_load_returns_false(self):
        from rag.retriever.bm25_search import BM25Searcher

        searcher = BM25Searcher()
        assert not searcher.load_from_disk("col", 0)

    def test_save_all_to_disk(self):
        from rag.retriever.bm25_search import BM25Searcher

        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "bm25")
            searcher = BM25Searcher(cache_dir=cache_dir)

            for name in ("a.col", "b.col"):
                searcher._cache[name] = {
                    "chunks": [{"id": "1", "metadata": {}, "document": "doc"}],
                    "symbol_corpus": [["x"]],
                    "signature_corpus": [["y"]],
                    "remarks_corpus": [["z"]],
                    "example_corpus": [[]],
                }
                searcher._counts[name] = 1

            searcher.save_all_to_disk()
            assert os.path.isfile(os.path.join(cache_dir, "a.col.pkl"))
            assert os.path.isfile(os.path.join(cache_dir, "b.col.pkl"))

    def test_clear_removes_memory_cache(self):
        from rag.retriever.bm25_search import BM25Searcher

        searcher = BM25Searcher()
        searcher._cache["col"] = {"chunks": []}
        searcher._counts["col"] = 5
        searcher.clear()
        assert len(searcher._cache) == 0
        assert len(searcher._counts) == 0
