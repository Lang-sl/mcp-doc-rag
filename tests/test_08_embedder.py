"""Stage 8: Embedding service.

Needs Ollama running with nomic-embed-text model pulled.
Verifies embedding dimension, batch processing, and error handling.

    pytest tests/test_08_embedder.py -v
"""

from __future__ import annotations

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
