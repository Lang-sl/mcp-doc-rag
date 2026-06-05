"""Indexer orchestrator -- crawl -> parse -> chunk -> embed -> store.

Central coordinator that ties the entire indexing pipeline together.

The pipeline is structured in four phases so that embedding runs in a
single global batch across all files — orders of magnitude faster than
embedding per-file.
"""

from __future__ import annotations

import time
from typing import Any

import chromadb

from rag.config import Config
from rag.indexer.chunker import build_chunks
from rag.indexer.crawler import (
    FileEntry,
    crawl_all,
    crawl_source,
    detect_deleted_files,
    remove_deleted_from_state,
    update_state,
)
from rag.indexer.embedder import Embedder
from rag.indexer.embedding_cache import EmbeddingCache
from rag.indexer.parser_registry import get_parser
from rag.retriever.bm25_search import BM25Searcher
from rag.symbol_index import SymbolIndex


def _store_chunks(
    client: chromadb.PersistentClient,
    chunks: list[Any],
    embeddings: list[list[float]],
    config: Config,
) -> tuple[int, list[str]]:
    """Store pre-embedded chunks in ChromaDB grouped by ``collection_name``.

    The caller is responsible for embedding all chunks before calling
    this function — the *embeddings* list must align 1:1 with *chunks*.

    Returns ``(stored_count, affected_collections)``.
    """
    by_collection: dict[str, list[tuple[Any, list[float]]]] = {}
    for chunk, vec in zip(chunks, embeddings):
        coll_name = chunk.collection_name
        by_collection.setdefault(coll_name, []).append((chunk, vec))

    stored = 0
    batch_size = config.index_batch_size
    affected: list[str] = []
    for coll_name, items in by_collection.items():
        try:
            collection = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            try:
                ids = [c.chunk_id for c, _ in batch]
                texts = [c.embed_text for c, _ in batch]
                metadatas = [c.to_metadata() for c, _ in batch]
                vecs = [v for _, v in batch]

                try:
                    collection.delete(ids=ids)
                except Exception:
                    pass

                collection.add(
                    ids=ids,
                    embeddings=vecs,
                    documents=texts,
                    metadatas=metadatas,
                )
                stored += len(batch)
            except Exception:
                continue

        affected.append(coll_name)

    return stored, affected


def _cleanup_deleted_chunks(
    client: chromadb.PersistentClient,
    source_label: str,
    source_path: str,
    config: Config,
    symbol_index: SymbolIndex | None = None,
) -> int:
    """Remove stale chunks for files deleted from *source_path* since the
    last indexing run.

    Queries ChromaDB for chunks whose ``source_file`` metadata field points
    to a file that no longer exists on disk, deletes them, and updates the
    symbol index and index state file accordingly.

    Returns the number of chunks removed.
    """
    deleted_files = detect_deleted_files(source_label, source_path, config.index_state_path)
    if not deleted_files:
        return 0

    removed = 0
    try:
        collections = client.list_collections()
    except Exception:
        collections = []

    for coll in collections:
        try:
            result = coll.get(
                where={"source_label": source_label},
                include=["metadatas"],
            )
        except Exception:
            continue

        if not result["ids"]:
            continue

        ids_to_delete = []
        for chunk_id, meta in zip(result["ids"], result["metadatas"]):
            if meta.get("source_file", "") in deleted_files:
                ids_to_delete.append(chunk_id)

        if ids_to_delete:
            try:
                coll.delete(ids=ids_to_delete)
                removed += len(ids_to_delete)
            except Exception:
                continue

    # Clean symbol index
    if symbol_index and deleted_files:
        symbol_index.remove_by_files(deleted_files)
        symbol_index.flush()

    # Clean state file
    remove_deleted_from_state(
        config.index_state_path, source_label, source_path, deleted_files
    )

    return removed


def _index_source(
    label: str,
    path: str,
    config: Config,
    embedder: Embedder,
    client: chromadb.PersistentClient,
) -> dict:
    """Orchestrate indexing for a single source in four phases.

    Phase 1 — Crawl: walk the directory, yield :class:`FileEntry` items.
    Phase 2 — Parse & Chunk: parse changed files and build chunks.
    Phase 3 — Embed: embed all chunks in a single global batch.
    Phase 4 — Store: write to ChromaDB per collection.

    Returns a stats dict with per-phase timings.
    """
    t0 = time.time()

    # -- Phase 0: Cleanup deleted files --
    symbol_index = SymbolIndex(config.symbol_index_path)
    deleted_count = _cleanup_deleted_chunks(
        client, label, path, config, symbol_index
    )

    # -- Phase 1: Crawl --
    t1 = time.time()
    entries = list(crawl_source(label, path, config.index_state_path))
    crawl_time = time.time() - t1

    files_total = len(entries)
    files_indexed = 0
    files_skipped = 0

    # -- Phase 2: Parse & Chunk (collect globally) --
    t2 = time.time()
    all_chunks: list[Any] = []
    parse_time = 0.0
    chunk_time = 0.0

    for entry in entries:
        if not entry.needs_index:
            files_skipped += 1
            continue

        parser = get_parser(entry.file_type)
        if parser is None:
            files_skipped += 1
            continue

        ta = time.time()
        try:
            parsed = parser(entry.abs_path, entry.source_label, entry.source_module)
        except Exception:
            files_skipped += 1
            continue
        parse_time += time.time() - ta

        if not parsed:
            files_skipped += 1
            continue

        tb = time.time()
        chunks = build_chunks(parsed, config)
        chunk_time += time.time() - tb

        if chunks:
            all_chunks.extend(chunks)
            files_indexed += 1
        else:
            files_skipped += 1

    total_chunks = len(all_chunks)

    # -- Phase 3: Embed (global batch) --
    t3 = time.time()
    embed_texts = [c.embed_text for c in all_chunks]
    embeddings = embedder.embed(embed_texts, batch_size=config.embed_batch_size)
    embed_time = time.time() - t3

    # -- Phase 4: Store to ChromaDB --
    t4 = time.time()
    stored = 0
    affected_collections: list[str] = []
    if embeddings:
        stored, affected_collections = _store_chunks(client, all_chunks, embeddings, config)
    chroma_time = time.time() - t4

    # -- Phase 4b: Persist BM25 disk cache --
    if affected_collections and config.bm25_cache_dir:
        try:
            bm25 = BM25Searcher(cache_dir=config.bm25_cache_dir)
            bm25._build_indices(client, affected_collections)
            for name in affected_collections:
                bm25.save_to_disk(name)
        except Exception:
            pass

    # Persist index state for incremental future runs
    update_state(config.index_state_path, label, entries)

    elapsed = time.time() - t0
    return {
        "files_total": files_total,
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "chunks": stored,
        "deleted": deleted_count,
        "elapsed_seconds": round(elapsed, 2),
        "crawl_time": round(crawl_time, 2),
        "parse_time": round(parse_time, 2),
        "chunk_time": round(chunk_time, 2),
        "embed_time": round(embed_time, 2),
        "chroma_time": round(chroma_time, 2),
    }


def _rebuild_symbol_index(config: Config) -> int:
    """Rebuild symbol index from ChromaDB metadata (single source of truth).

    Returns the number of symbols indexed.
    """
    symbol_index = SymbolIndex(config.symbol_index_path)
    symbol_index._index.clear()

    for coll in chromadb.PersistentClient(path=config.chroma_dir).list_collections():
        try:
            response = coll.get(include=["metadatas"])
        except Exception:
            continue
        for metadata in response.get("metadatas", []):
            symbol_id = metadata.get("symbol_id", "")
            if not symbol_id or symbol_id in symbol_index._index:
                continue
            symbol_index._index[symbol_id] = {
                "type": metadata.get("type", ""),
                "symbol_id": symbol_id,
                "class_name": metadata.get("class_name") or None,
                "function_name": metadata.get("function_name") or None,
                "source_label": metadata.get("source_label", ""),
                "source_module": metadata.get("source_module", ""),
                "file_path": metadata.get("source_file", ""),
            }
    symbol_index.flush()
    return len(symbol_index)


def index_source(config: Config, label: str) -> dict:
    """Index a single source by *label*.

    Looks up *label* in ``config.doc_sources``.  Returns per-source stats.
    Raises :class:`ValueError` when *label* is not a registered source.
    """
    path = config.doc_sources.get(label)
    if path is None:
        raise ValueError(f"Unknown source label: {label!r}")

    cache = EmbeddingCache(config.embedding_cache_dir)
    embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim, cache=cache)
    client = chromadb.PersistentClient(path=config.chroma_dir)

    try:
        result = _index_source(label, path, config, embedder, client)
        _rebuild_symbol_index(config)
        return result
    finally:
        del embedder
        del client


def index_all(config: Config) -> dict:
    """Run the full indexing pipeline across ALL registered sources.

    Returns ``{"total_chunks": <int>, "sources": {label: stats, ...}}``
    where each per-source *stats* dict includes per-phase timing fields:
    ``crawl_time``, ``parse_time``, ``chunk_time``, ``embed_time``,
    ``chroma_time``, and ``elapsed_seconds``.
    """
    cache = EmbeddingCache(config.embedding_cache_dir)
    embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim, cache=cache)
    client = chromadb.PersistentClient(path=config.chroma_dir)

    total_chunks = 0
    sources: dict[str, dict] = {}

    for label, path in config.doc_sources.items():
        try:
            stats = _index_source(label, path, config, embedder, client)
        except Exception as exc:
            stats = {
                "files_total": 0,
                "files_indexed": 0,
                "files_skipped": 0,
                "chunks": 0,
                "elapsed_seconds": 0,
                "crawl_time": 0,
                "parse_time": 0,
                "chunk_time": 0,
                "embed_time": 0,
                "chroma_time": 0,
                "error": str(exc),
            }
        sources[label] = stats
        total_chunks += stats.get("chunks", 0)

    _rebuild_symbol_index(config)

    del embedder
    del client

    return {"total_chunks": total_chunks, "sources": sources}
