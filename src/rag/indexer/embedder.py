"""Embedding wrapper around the Ollama embedding API.

Provides batch processing with retry for the indexer pipeline.
"""

from __future__ import annotations

import time
from typing import Sequence

import ollama


class Embedder:
    """Wrapper around the Ollama embedding API with batch processing and retry.

    Uses Ollama's batch ``embed`` API (``input=[text, ...]``) to embed
    multiple texts in a single HTTP round-trip — orders of magnitude faster
    than per-text ``embeddings`` calls on large document sets.
    """

    def __init__(self, host: str, model: str, dim: int):
        self.host = host
        self.model = model
        self.dim = dim
        self._client = ollama.Client(host=host)

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of texts in batches via Ollama's batch ``embed`` API.

        Returns a list of embedding vectors, one per input text.
        """
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._embed_batch(list(batch))
            all_vectors.extend(vectors)

        return all_vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text], batch_size=1)[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the batch ``embed`` API with one retry on failure.

        Returns zero vectors on persistent failure to avoid blocking the
        pipeline.
        """
        for attempt in range(2):
            try:
                response = self._client.embed(model=self.model, input=texts)
                return list(response.embeddings)
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                else:
                    return [[0.0] * self.dim for _ in texts]

        # Should be unreachable, but satisfy the type checker
        return [[0.0] * self.dim for _ in texts]
