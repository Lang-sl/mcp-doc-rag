"""Disk-backed embedding cache keyed by sha256(text + model).

Allows incremental reindex to skip embedding computation for texts
that have not changed since the last run.
"""

from __future__ import annotations

import hashlib
import json
import os


class EmbeddingCache:
    """Disk cache for embedding vectors, keyed by ``sha256(text|model)``.

    Each cached vector is stored as a small JSON file under *cache_dir*.
    """

    def __init__(self, cache_dir: str):
        self._dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, text: str, model: str) -> str:
        return hashlib.sha256(f"{text}|{model}".encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        p = os.path.join(self._dir, self._key(text, model) + ".json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return None

    def set(self, text: str, model: str, vec: list[float]) -> None:
        p = os.path.join(self._dir, self._key(text, model) + ".json")
        with open(p, "w") as f:
            json.dump(vec, f)
