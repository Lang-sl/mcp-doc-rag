"""Hybrid retriever — orchestrates the full retrieval pipeline.

BM25 + vector -> RRF -> reranker (optional) -> code boost -> reference expansion.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import chromadb

from rag.config import Config
from rag.indexer.embedder import Embedder
from rag.models import Chunk, SearchResult
from rag.retriever.vector_search import vector_search
from rag.retriever.bm25_search import BM25Searcher
from rag.retriever.reranker import Reranker


def _is_symbol_lookup(query: str) -> bool:
    """Detect exact symbol / API lookups that don't benefit from reranking.

    Returns True when *query* looks like an identifier lookup rather than
    a natural-language question.  Conservative: only flags clear symbol
    patterns.  Callers should pass ``skip_rerank=True`` explicitly for
    known API lookups.
    """
    q = query.strip()

    # C++ qualified name: "Foo::Bar" or "Namespace::Class::method"
    if "::" in q:
        return True

    # Single-word PascalCase or snake_case identifier
    if " " not in q:
        stripped = q.strip("()<>*&[]")
        if stripped and (stripped[0].isupper() or "_" in stripped):
            return True

    return False


class HybridRetriever:
    """Full pipeline: BM25 + vector -> RRF -> reranker -> code boost -> reference expansion.

    The reranker is skipped automatically for symbol/API lookups.  Callers
    can override via the *skip_rerank* parameter.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.chroma_dir)
        self.embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)
        self.reranker = Reranker(config.reranker_model, config.reranker_max_length)
        self._bm25 = BM25Searcher()

        # LRU cache keyed by (query, source_label, module, top_k)
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()

    def _cache_key(self, query: str, source_label: str | None, module: str | None, top_k: int) -> str:
        return hashlib.md5(f"{query}|{source_label}|{module}|{top_k}".encode()).hexdigest()

    def _get_cached(self, query: str, source_label: str | None, module: str | None, top_k: int) -> list[SearchResult] | None:
        key = self._cache_key(query, source_label, module, top_k)
        if key in self._cache:
            self._cache.move_to_end(key)
            raw = self._cache[key]
            return [_deserialize_result(r) for r in raw]
        return None

    def _set_cache(self, query: str, source_label: str | None, module: str | None, top_k: int, results: list[SearchResult]) -> None:
        key = self._cache_key(query, source_label, module, top_k)
        self._cache[key] = [_serialize_result(r) for r in results]

        # Evict least recently used if over capacity
        if len(self._cache) > self.config.cache_max_entries:
            self._cache.popitem(last=False)

    def invalidate_cache(self) -> None:
        """Clear BM25 and LRU caches.  Call after reindexing."""
        self._bm25.clear()
        self._cache.clear()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source_label: str | None = None,
        module: str | None = None,
        skip_rerank: bool = False,
        enable_rewrite: bool = False,
    ) -> list[SearchResult]:
        """Run the full retrieval pipeline.

        Set *skip_rerank* to True for exact symbol/API lookups where
        the reranker adds latency without meaningful relevance gain.
        Auto-detects symbol-like queries when *skip_rerank* is False.

        Set *enable_rewrite* to True to expand natural-language queries
        with domain synonyms before BM25 search, improving recall for
        conceptual queries without affecting symbol/API lookups.
        """
        if top_k is None:
            top_k = self.config.top_k_default

        # Build a cache key that includes the rerank decision
        cache_key_extra = 1 if skip_rerank else 0
        cached = self._get_cached(query, source_label, module, top_k)
        if cached is not None:
            return cached

        candidate_count = top_k * self.config.candidate_expand_factor

        # Step 1: Vector search
        vec_results = vector_search(
            self.client, self.embedder, self.config,
            query, candidate_count, source_label,
        )

        # Step 2: BM25 search (uses cached indices)
        bm25_results = self._bm25.search(
            self.client, self.config, query, candidate_count, source_label,
        )

        # Step 2b: Query rewrite — expand BM25 with synonym variants
        if enable_rewrite:
            from rag.retriever.query_rewriter import expand

            variants = expand(query, self.config.query_rewrite_max_variants)
            if len(variants) > 1:
                all_bm25: list[SearchResult] = list(bm25_results)
                for v in variants[1:]:  # first variant is the original query
                    all_bm25.extend(
                        self._bm25.search(
                            self.client, self.config, v, candidate_count, source_label,
                        )
                    )
                # Deduplicate by chunk_id, preserving the highest score
                seen_bm25: dict[str, SearchResult] = {}
                for r in all_bm25:
                    if r.chunk.chunk_id not in seen_bm25 or r.score > seen_bm25[r.chunk.chunk_id].score:
                        seen_bm25[r.chunk.chunk_id] = r
                bm25_results = sorted(seen_bm25.values(), key=lambda r: r.score, reverse=True)
                bm25_results = bm25_results[:candidate_count]

        # Step 3: RRF fusion
        fused = _rrf_fuse(vec_results, bm25_results, self.config.rrf_k, candidate_count)

        # Step 4: Reranker (skip for symbol/API lookups)
        should_rerank = not skip_rerank and not _is_symbol_lookup(query)
        if fused and should_rerank:
            try:
                fused = self.reranker.rerank(query, fused)
            except Exception:
                pass  # Reranker unavailable — continue with RRF scores

        # Step 5: Code boost
        query_lower = query.lower()
        if any(trigger in query_lower for trigger in self.config.code_boost_triggers):
            for r in fused:
                if r.chunk.contains_code:
                    r.score *= self.config.code_boost_ratio

            # Re-sort after boost
            fused.sort(key=lambda x: x.score, reverse=True)

        # Step 6: Reference expansion
        fused = _expand_references(fused, self.client, self.config.ref_expansion_max)

        # Step 7: Module filter (if specified, apply after expansion for best recall)
        if module:
            fused = [r for r in fused if r.chunk.source_module == module]

        top_results = fused[:top_k]

        # Cache
        self._set_cache(query, source_label, module, top_k, top_results)

        return top_results


def _rrf_fuse(
    vec_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int,
    max_results: int,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of two ranked lists."""
    rrf_scores: dict[str, tuple[float, SearchResult]] = {}

    for rank, r in enumerate(vec_results[:max_results], start=1):
        rrf = 1.0 / (k + rank)
        if r.chunk.chunk_id in rrf_scores:
            prev_score, prev_result = rrf_scores[r.chunk.chunk_id]
            rrf_scores[r.chunk.chunk_id] = (prev_score + rrf, prev_result)
        else:
            rrf_scores[r.chunk.chunk_id] = (rrf, r)

    for rank, r in enumerate(bm25_results[:max_results], start=1):
        rrf = 1.0 / (k + rank)
        if r.chunk.chunk_id in rrf_scores:
            prev_score, prev_result = rrf_scores[r.chunk.chunk_id]
            rrf_scores[r.chunk.chunk_id] = (prev_score + rrf, prev_result)
        else:
            rrf_scores[r.chunk.chunk_id] = (rrf, r)

    fused = []
    for chunk_id, (score, result) in rrf_scores.items():
        result.score = score
        fused.append(result)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def _expand_references(
    results: list[SearchResult],
    client: chromadb.PersistentClient,
    max_expansion: int,
) -> list[SearchResult]:
    """One-hop reference expansion: add chunks referenced by top results."""
    if not results:
        return results

    existing_ids = {r.chunk.chunk_id for r in results}
    expanded = list(results)
    added = 0

    # Pre-fetch collection list once
    collections = client.list_collections()

    for r in results[:5]:  # Only expand from top 5
        for ref_symbol in r.chunk.references:
            if added >= max_expansion:
                break

            # Search for the referenced symbol across all collections
            for coll in collections:
                try:
                    collection = client.get_collection(name=coll.name)
                    response = collection.get(
                        where={"symbol_id": ref_symbol},
                        include=["metadatas", "documents"],
                        limit=1,
                    )
                except Exception:
                    continue

                ids = response.get("ids", [])
                if ids and ids[0] not in existing_ids:
                    metadata = response["metadatas"][0] if response.get("metadatas") else {}
                    document = response["documents"][0] if response.get("documents") else ""
                    chunk = Chunk.from_metadata(metadata, document)
                    expanded.append(SearchResult(
                        chunk=chunk,
                        score=r.score * 0.8,  # Slightly lower score than the referencing chunk
                    ))
                    existing_ids.add(ids[0])
                    added += 1
                    break

    return expanded


def _serialize_result(r: SearchResult) -> dict[str, Any]:
    """Serialize SearchResult to JSON-serializable dict for caching."""
    return {
        "chunk_id": r.chunk.chunk_id,
        "type": r.chunk.type,
        "symbol_id": r.chunk.symbol_id,
        "class_name": r.chunk.class_name,
        "function_name": r.chunk.function_name,
        "signature": r.chunk.signature,
        "source_label": r.chunk.source_label,
        "source_module": r.chunk.source_module,
        "source_file": r.chunk.source_file,
        "contains_code": r.chunk.contains_code,
        "references": r.chunk.references,
        "embed_text": r.chunk.embed_text,
        "score": r.score,
    }


def _deserialize_result(data: dict[str, Any]) -> SearchResult:
    """Deserialize cached result dict back to SearchResult."""
    chunk = Chunk(
        chunk_id=data["chunk_id"],
        type=data["type"],
        symbol_id=data.get("symbol_id"),
        class_name=data.get("class_name"),
        function_name=data.get("function_name"),
        signature=data.get("signature"),
        source_label=data["source_label"],
        source_module=data["source_module"],
        source_file=data["source_file"],
        contains_code=data.get("contains_code", False),
        references=data.get("references", []),
        embed_text=data["embed_text"],
    )
    return SearchResult(chunk=chunk, score=data["score"])
