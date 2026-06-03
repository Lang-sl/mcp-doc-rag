from __future__ import annotations

import json
import os

from .models import Chunk


class SymbolIndex:
    """In-memory symbol_id -> metadata hash map with JSON persistence."""

    def __init__(self, path: str):
        """Load index from JSON file at path."""
        self._path = path
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load from disk. Empty dict if file doesn't exist."""
        if os.path.isfile(self._path):
            with open(self._path, "r", encoding="utf-8") as fh:
                self._index = json.load(fh)

    def _save(self) -> None:
        """Persist to disk. Create parent dirs."""
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2, ensure_ascii=False)

    def add_chunk(self, chunk: Chunk) -> None:
        """Register chunk's symbol_id if present and not already in index."""
        if chunk.symbol_id is None or chunk.symbol_id in self._index:
            return
        self._index[chunk.symbol_id] = self._chunk_to_metadata(chunk)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Bulk register."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def lookup(self, symbol: str) -> dict | None:
        """O(1) lookup. Returns metadata dict or None."""
        return self._index.get(symbol)

    def remove_source(self, source_label: str) -> int:
        """Remove all symbols for a source. Returns count removed."""
        toRemove = [
            sid for sid, meta in self._index.items()
            if meta.get("source_label") == source_label
        ]
        for sid in toRemove:
            del self._index[sid]
        return len(toRemove)

    def flush(self) -> None:
        """Force persist."""
        self._save()

    def __len__(self) -> int:
        """Number of symbols in index."""
        return len(self._index)

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk) -> dict:
        return {
            "type": chunk.type,
            "symbol_id": chunk.symbol_id,
            "class_name": chunk.class_name,
            "function_name": chunk.function_name,
            "source_label": chunk.source_label,
            "source_module": chunk.source_module,
            "file_path": chunk.source_file,
        }
