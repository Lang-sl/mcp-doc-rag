"""Field-weighted BM25 keyword search with per-collection index caching.

BM25Okapi indices are built once and cached in memory.  Subsequent searches
reuse the cached indices unless the underlying ChromaDB collection has
changed (detected via chunk count).
"""

from __future__ import annotations

import chromadb

from rag.config import Config
from rag.models import Chunk, SearchResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer, lowercase."""
    return text.lower().split()


class BM25Searcher:
    """Cached field-weighted BM25 search across ChromaDB collections.

    Builds :class:`rank_bm25.BM25Okapi` indices per collection on first use
    and reuses them on subsequent searches.  Indices are automatically
    invalidated when a collection's chunk count changes (e.g. after reindex).
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._counts: dict[str, int] = {}

    def clear(self) -> None:
        """Clear all cached indices (call after reindex)."""
        self._cache.clear()
        self._counts.clear()

    def _get_collection_names(
        self,
        client: chromadb.PersistentClient,
        source_label: str | None,
    ) -> list[str]:
        """Return collection names filtered by optional source_label prefix."""
        all_names = [c.name for c in client.list_collections()]

        if source_label is not None:
            prefix = f"{source_label}."
            matching = [n for n in all_names if n.startswith(prefix)]
            if matching:
                return matching

        return all_names

    def _build_indices(
        self,
        client: chromadb.PersistentClient,
        collection_names: list[str],
    ) -> None:
        """Build BM25Okapi indices for collections not already cached."""
        from rank_bm25 import BM25Okapi

        for name in collection_names:
            try:
                collection = client.get_collection(name)
                current_count = collection.count()
            except Exception:
                continue

            # Skip if cache is still valid for this collection
            if name in self._cache and self._counts.get(name) == current_count:
                continue

            try:
                data = collection.get(include=["metadatas", "documents"])
            except Exception:
                continue

            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            documents = data.get("documents", [])

            if not ids:
                self._cache.pop(name, None)
                self._counts.pop(name, None)
                continue

            chunks: list[dict] = []
            symbol_corpus: list[list[str]] = []
            signature_corpus: list[list[str]] = []
            remarks_corpus: list[list[str]] = []
            example_corpus: list[list[str]] = []

            for chunk_id, metadata, document in zip(ids, metadatas, documents):
                meta = metadata or {}
                doc = document or ""

                chunks.append({"id": chunk_id, "metadata": meta, "document": doc})
                symbol_corpus.append(_tokenize(meta.get("symbol_id") or ""))
                signature_corpus.append(_tokenize(meta.get("signature") or ""))
                remarks_corpus.append(_tokenize(doc))
                example_corpus.append(_tokenize(meta.get("example") or ""))

            def _mk_bm25(corpus: list[list[str]]) -> BM25Okapi | None:
                if any(corpus):
                    try:
                        return BM25Okapi(corpus)
                    except Exception:
                        return None
                return None

            self._cache[name] = {
                "symbol_bm25": _mk_bm25(symbol_corpus),
                "signature_bm25": _mk_bm25(signature_corpus),
                "remarks_bm25": _mk_bm25(remarks_corpus),
                "example_bm25": _mk_bm25(example_corpus),
                "chunks": chunks,
            }
            self._counts[name] = current_count

    def search(
        self,
        client: chromadb.PersistentClient,
        config: Config,
        query: str,
        top_k: int,
        source_label: str | None = None,
    ) -> list[SearchResult]:
        """Field-weighted BM25 keyword search with cached indices.

        Same scoring algorithm as :func:`bm25_search`.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        collection_names = self._get_collection_names(client, source_label)

        # Build / refresh cached indices for relevant collections
        self._build_indices(client, collection_names)

        query_lower = query.lower()
        enable_example = any(
            trigger in query_lower
            for trigger in ["example", "code", "sample", "how to"]
        )

        results: list[SearchResult] = []
        seen: set[str] = set()
        w = config.bm25_weights

        for name in collection_names:
            cached = self._cache.get(name)
            if cached is None:
                continue

            chunks = cached["chunks"]
            if not chunks:
                continue

            n = len(chunks)

            sym = [0.0] * n
            sig = [0.0] * n
            rem = [0.0] * n
            ex = [0.0] * n

            if cached["symbol_bm25"] is not None:
                try:
                    sym = cached["symbol_bm25"].get_scores(query_tokens)
                except Exception:
                    pass

            if cached["signature_bm25"] is not None:
                try:
                    sig = cached["signature_bm25"].get_scores(query_tokens)
                except Exception:
                    pass

            if cached["remarks_bm25"] is not None:
                try:
                    rem = cached["remarks_bm25"].get_scores(query_tokens)
                except Exception:
                    pass

            if enable_example and cached["example_bm25"] is not None:
                try:
                    ex = cached["example_bm25"].get_scores(query_tokens)
                except Exception:
                    pass

            for i, c in enumerate(chunks):
                total = (
                    w.symbol_name * sym[i]
                    + w.signature * sig[i]
                    + w.remarks * rem[i]
                    + w.example * ex[i]
                )

                if total <= 0.0:
                    continue

                chunk = Chunk.from_metadata(c["metadata"], c["document"])

                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)

                results.append(SearchResult(chunk=chunk, score=total))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Backward-compatible standalone function (uses cached module-level singleton)
# ---------------------------------------------------------------------------

_bm25_searcher: BM25Searcher | None = None


def _get_all_chunks_from_collections(
    client: chromadb.PersistentClient,
    source_label: str | None,
) -> dict[str, list[dict]]:
    """Returns {collection_name: [{id, metadata, document}, ...]}

    Filter collections by source_label prefix. Empty collections and
    those that fail to load are silently skipped.
    """
    all_collections = client.list_collections()

    if source_label is not None:
        prefix = f"{source_label}."
        target_names = [c.name for c in all_collections if c.name.startswith(prefix)]
        if not target_names:
            return {}
    else:
        target_names = [c.name for c in all_collections]

    result: dict[str, list[dict]] = {}

    for name in target_names:
        try:
            collection = client.get_collection(name)
            data = collection.get(include=["metadatas", "documents"])

            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            documents = data.get("documents", [])

            if not ids:
                continue

            chunks: list[dict] = []
            for chunk_id, metadata, document in zip(ids, metadatas, documents):
                chunks.append({
                    "id": chunk_id,
                    "metadata": metadata or {},
                    "document": document or "",
                })

            if chunks:
                result[name] = chunks
        except Exception:
            continue

    return result


def bm25_search(
    client: chromadb.PersistentClient,
    config: Config,
    query: str,
    top_k: int,
    source_label: str | None = None,
) -> list[SearchResult]:
    """Field-weighted BM25 keyword search (uses cached module singleton).

    Prefer :class:`BM25Searcher` for repeated searches where you control
    the instance lifecycle.  This function delegates to a module-level
    singleton for backward compatibility.
    """
    global _bm25_searcher
    if _bm25_searcher is None:
        _bm25_searcher = BM25Searcher()
    return _bm25_searcher.search(client, config, query, top_k, source_label)
