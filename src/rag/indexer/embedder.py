"""Embedding wrapper around the Ollama embedding API.

Provides batch processing with retry for the indexer pipeline.
"""

from __future__ import annotations

import time
from typing import Sequence

import ollama


class Embedder:
    """Wrapper around the Ollama embedding API with batch processing and retry."""

    def __init__(self, host: str, model: str, dim: int):
        self.host = host
        self.model = model
        self.dim = dim
        self._client = ollama.Client(host=host)

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts in batches. Returns list of embedding vectors."""
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                vec = self._embed_single(text)
                all_vectors.append(vec)

        return all_vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text], batch_size=1)[0]

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string with one retry on failure.

        Returns a zero vector on persistent failure to avoid blocking the pipeline.
        """
        for attempt in range(2):
            try:
                response = self._client.embeddings(model=self.model, prompt=text)
                return response["embedding"]
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                else:
                    return [0.0] * self.dim

        # Should be unreachable, but satisfy the type checker
        return [0.0] * self.dim
