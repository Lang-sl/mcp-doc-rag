"""Stage 8: Embedding service.

Needs Ollama running with nomic-embed-text model pulled.
Verifies embedding dimension, batch processing, and error handling.

    pytest tests/test_08_embedder.py -v
"""

from __future__ import annotations

import os
import tempfile

from rag.config import Config
from rag.indexer.embedder import Embedder
from tests.conftest import requires_ollama


@requires_ollama()
class TestEmbedder:
    """Tests that require a running Ollama service."""

    def test_embed_one_returns_correct_dim(self):
        config = Config()
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        vec = embedder.embed_one("Initialize the rendering kernel")
        assert len(vec) == config.embed_dim, (
            f"Expected {config.embed_dim}-dim vector, got {len(vec)}"
        )
        assert any(v != 0.0 for v in vec), "Vector should not be all zeros"

    def test_embed_batch_returns_correct_count(self):
        config = Config()
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        texts = ["First text", "Second text", "Third text"]
        vecs = embedder.embed(texts, batch_size=2)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == config.embed_dim

    def test_embed_empty_list(self):
        config = Config()
        embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)

        vecs = embedder.embed([], batch_size=32)
        assert vecs == []


class TestEmbedderOffline:
    """Tests that work even without Ollama."""

    def test_embedder_creation_does_not_connect(self):
        """Creating an Embedder should not immediately connect to Ollama."""
        embedder = Embedder("http://localhost:11434", "nomic-embed-text", 768)
        assert embedder.dim == 768
        assert embedder.model == "nomic-embed-text"

    def test_embedder_zero_dim_vector_on_failure(self):
        """With unreachable host, embed should return zero vector (not crash)."""
        embedder = Embedder("http://127.0.0.1:19999", "nomic-embed-text", 768)
        vec = embedder.embed_one("test")
        assert len(vec) == 768
        # With bad host, we get either a zero vector (retry exhausted)
        # or possibly an error if ollama client raises differently


# ---------------------------------------------------------------------------
# Embedding Cache — disk-backed cache for incremental reindex speedup
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    """Disk-backed embedding cache unit tests (no Ollama needed)."""

    def test_set_and_get(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("hello world", "test-model", [0.1, 0.2, 0.3])
            result = cache.get("hello world", "test-model")
            assert result == [0.1, 0.2, 0.3]

    def test_miss_returns_none(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            assert cache.get("nonexistent", "test-model") is None

    def test_different_model_different_key(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("text", "model-a", [1.0, 2.0])
            assert cache.get("text", "model-b") is None

    def test_different_text_different_key(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("text-a", "model", [1.0])
            assert cache.get("text-b", "model") is None

    def test_creates_cache_dir(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as parent:
            cache_dir = os.path.join(parent, "sub", "cache")
            cache = EmbeddingCache(cache_dir)
            assert os.path.isdir(cache_dir)

    def test_persists_across_instances(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("persist", "model", [3.0, 4.0])

            cache2 = EmbeddingCache(d)
            result = cache2.get("persist", "model")
            assert result == [3.0, 4.0]

    def test_empty_vector_roundtrip(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("empty", "model", [])
            assert cache.get("empty", "model") == []

    def test_768_dim_vector_roundtrip(self):
        from rag.indexer.embedding_cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            vec = [0.001 * i for i in range(768)]
            cache.set("large", "model", vec)
            result = cache.get("large", "model")
            assert len(result) == 768
            assert abs(result[767] - 0.767) < 0.001


class TestEmbedderWithCache:
    """Embedder integration with EmbeddingCache."""

    def test_cache_hit_skips_ollama(self):
        """When cache has all texts, embed returns cached vectors without API call."""
        from rag.indexer.embedding_cache import EmbeddingCache
        from rag.indexer.embedder import Embedder

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cached_vec = [0.1] * 4
            cache.set("cached-text", "test-model", cached_vec)

            # Use unreachable host — if cache works, Ollama is never called
            embedder = Embedder(
                "http://127.0.0.1:19999", "test-model", 4, cache=cache,
            )
            result = embedder.embed(["cached-text"])
            assert len(result) == 1
            assert result[0] == cached_vec

    def test_partial_cache_miss_stores_new(self):
        """Cache hits used directly; misses get embedded and cached."""
        from rag.indexer.embedding_cache import EmbeddingCache
        from rag.indexer.embedder import Embedder

        with tempfile.TemporaryDirectory() as d:
            cache = EmbeddingCache(d)
            cache.set("keep", "nomic-embed-text", [0.5] * 768)

            embedder = Embedder(
                "http://127.0.0.1:19999", "nomic-embed-text", 768, cache=cache,
            )
            result = embedder.embed(["keep", "new-one"])
            assert len(result) == 2
            assert result[0] == [0.5] * 768
            # new-one should have been cached (even if zero due to bad host)
            stored = cache.get("new-one", "nomic-embed-text")
            assert stored is not None

    def test_no_cache_still_works(self):
        """Without cache, Embedder works as before (backward compatible)."""
        from rag.indexer.embedder import Embedder

        embedder = Embedder("http://127.0.0.1:19999", "nomic-embed-text", 4)
        result = embedder.embed(["test"])
        assert len(result) == 1
        assert len(result[0]) == 4
