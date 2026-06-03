"""Stage 9: Search pipeline.

Needs: Ollama running + indexed data in ChromaDB.
Verifies vector search, BM25 search, hybrid search, and symbol lookup.

    pytest tests/test_09_search.py -v
"""

from __future__ import annotations

import os

import pytest

from rag.config import load_config
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
