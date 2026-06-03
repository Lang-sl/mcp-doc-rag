"""Per-collection ChromaDB ANN search. Merges results across multiple collections."""

from __future__ import annotations

import chromadb

from rag.config import Config
from rag.indexer.embedder import Embedder
from rag.models import Chunk, SearchResult


def _get_collections_for_source(
    client: chromadb.PersistentClient,
    source_label: str | None,
) -> list[str]:
    """Get collection names, optionally filtered by source prefix.

    If *source_label* is given, only collections whose name starts with
    ``{source_label}.`` are returned.  When the filter yields no matches,
    all collection names are returned as a fallback.
    """
    all_names: list[str] = [c.name for c in client.list_collections()]

    if source_label is not None:
        prefix = f"{source_label}."
        matching = [n for n in all_names if n.startswith(prefix)]
        if matching:
            return matching

    return all_names


def vector_search(
    client: chromadb.PersistentClient,
    embedder: Embedder,
    config: Config,
    query: str,
    top_k: int,
    source_label: str | None = None,
) -> list[SearchResult]:
    """ANN search across matching ChromaDB collections.

    1. Embed *query* via *embedder*.
    2. Determine target collection names.
    3. Query each collection independently (a failure in one collection
       does not abort the others).
    4. Merge results, deduplicate by ``chunk_id``, and return the top *top_k*
       entries sorted by descending score.
    """
    # 1. Embed query
    query_vec: list[float] = embedder.embed_one(query)

    # 2. Get collection names
    collection_names = _get_collections_for_source(client, source_label)

    results: list[SearchResult] = []
    seen: set[str] = set()

    # 3. Query each collection
    for name in collection_names:
        try:
            collection = client.get_collection(name)
            response = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )

            metadatas_all = response.get("metadatas")
            documents_all = response.get("documents")
            distances_all = response.get("distances")

            if not metadatas_all or not documents_all or not distances_all:
                continue

            for metadata_list, doc_list, dist_list in zip(
                metadatas_all, documents_all, distances_all
            ):
                if metadata_list is None or doc_list is None or dist_list is None:
                    continue

                for metadata, document, distance in zip(
                    metadata_list, doc_list, dist_list
                ):
                    if metadata is None or document is None or distance is None:
                        continue

                    chunk = Chunk.from_metadata(metadata, document)
                    score = 1.0 - distance

                    if chunk.chunk_id in seen:
                        continue
                    seen.add(chunk.chunk_id)

                    results.append(SearchResult(chunk=chunk, score=score))
        except Exception:
            # One failing collection should not break the others
            continue

    # 4. Sort by score desc, return top_k
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_k]
