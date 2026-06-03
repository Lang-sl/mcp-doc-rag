"""Field-weighted BM25 keyword search with per-collection indexing."""

from __future__ import annotations

import chromadb
from rank_bm25 import BM25Okapi

from rag.config import Config
from rag.models import Chunk, SearchResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer, lowercase."""
    return text.lower().split()


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
    """Field-weighted BM25 keyword search across matching ChromaDB collections.

    1. Get all chunks from matching collections (filtered by source_label prefix)
    2. For each collection, build 3 BM25 corpora from chunk metadata:
       - ``symbol_corpus``: tokenize ``symbol_id``, weighted by
         ``config.bm25_weights.symbol_name``
       - ``signature_corpus``: tokenize ``signature``, weighted by
         ``config.bm25_weights.signature``
       - ``remarks_corpus``: tokenize ``document`` (embed_text), weighted by
         ``config.bm25_weights.remarks``
    3. Optionally score the example field when the query contains trigger words
       ("example", "code", "sample", "how to").
    4. Compute weighted sum per chunk, deduplicate by ``chunk_id``, and return
       the top *top_k* results sorted by descending score.
    """
    collections_data = _get_all_chunks_from_collections(client, source_label)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    query_lower = query.lower()
    enable_example = any(
        trigger in query_lower for trigger in ["example", "code", "sample", "how to"]
    )

    results: list[SearchResult] = []
    seen: set[str] = set()

    for _collection_name, chunks in collections_data.items():
        if not chunks:
            continue

        n = len(chunks)

        # Build tokenized corpora from metadata fields
        symbol_corpus: list[list[str]] = []
        signature_corpus: list[list[str]] = []
        remarks_corpus: list[list[str]] = []
        example_corpus: list[list[str]] = []

        for c in chunks:
            metadata = c["metadata"]
            document = c["document"]

            symbol_corpus.append(_tokenize(metadata.get("symbol_id") or ""))
            signature_corpus.append(_tokenize(metadata.get("signature") or ""))
            remarks_corpus.append(_tokenize(document))
            example_corpus.append(_tokenize(metadata.get("example") or ""))

        # Compute per-field BM25 scores (skip fields with no tokens)
        symbol_scores: list[float] = [0.0] * n
        sig_scores: list[float] = [0.0] * n
        rem_scores: list[float] = [0.0] * n
        ex_scores: list[float] = [0.0] * n

        if any(symbol_corpus):
            try:
                symbol_scores = BM25Okapi(symbol_corpus).get_scores(query_tokens)
            except Exception:
                pass

        if any(signature_corpus):
            try:
                sig_scores = BM25Okapi(signature_corpus).get_scores(query_tokens)
            except Exception:
                pass

        if any(remarks_corpus):
            try:
                rem_scores = BM25Okapi(remarks_corpus).get_scores(query_tokens)
            except Exception:
                pass

        if enable_example and any(example_corpus):
            try:
                ex_scores = BM25Okapi(example_corpus).get_scores(query_tokens)
            except Exception:
                pass

        # Compute weighted totals and assemble results
        w = config.bm25_weights

        for i, c in enumerate(chunks):
            total = (
                w.symbol_name * symbol_scores[i]
                + w.signature * sig_scores[i]
                + w.remarks * rem_scores[i]
                + w.example * ex_scores[i]
            )

            if total <= 0.0:
                continue

            metadata = c["metadata"]
            chunk = Chunk.from_metadata(metadata, c["document"])

            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            results.append(SearchResult(chunk=chunk, score=total))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
