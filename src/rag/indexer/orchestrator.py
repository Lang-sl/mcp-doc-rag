"""Indexer orchestrator -- crawl -> parse -> chunk -> embed -> store.

Central coordinator that ties the entire indexing pipeline together.
"""

from __future__ import annotations

import time
from typing import Any

import chromadb

from rag.config import Config
from rag.indexer.chunker import build_chunks
from rag.indexer.crawler import FileEntry, crawl_all, crawl_source, update_state
from rag.indexer.embedder import Embedder
from rag.indexer.parser_header import parse_header
from rag.indexer.parser_html import parse_html
from rag.indexer.parser_pdf import parse_pdf

PARSERS = {
    "html": parse_html,
    "pdf": parse_pdf,
    "header": parse_header,
}


def _get_parser(file_type: str):
    """Return the parser callable for *file_type*, or None when unknown."""
    return PARSERS.get(file_type)


def _store_chunks(
    client: chromadb.PersistentClient,
    chunks: list[Any],
    embedder: Embedder,
    config: Config,
) -> int:
    """Store chunks in ChromaDB grouped by ``collection_name``.

    Returns the number of chunks successfully stored.
    """
    by_collection: dict[str, list[Any]] = {}
    for chunk in chunks:
        coll_name = chunk.collection_name
        by_collection.setdefault(coll_name, []).append(chunk)

    stored = 0
    for coll_name, coll_chunks in by_collection.items():
        try:
            collection = client.get_or_create_collection(name=coll_name)
        except Exception:
            continue

        batch_size = config.index_batch_size
        for i in range(0, len(coll_chunks), batch_size):
            batch = coll_chunks[i : i + batch_size]
            try:
                ids = [c.chunk_id for c in batch]
                texts = [c.embed_text for c in batch]
                metadatas = [c.to_metadata() for c in batch]
                embeddings = embedder.embed(texts)

                if not embeddings or len(embeddings) != len(batch):
                    continue

                try:
                    collection.delete(ids=ids)
                except Exception:
                    pass

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                stored += len(batch)
            except Exception:
                continue

    return stored


def _index_source(
    label: str,
    path: str,
    config: Config,
    embedder: Embedder,
    client: chromadb.PersistentClient,
) -> dict:
    """Orchestrate indexing for a single source.

    Crawl, parse, chunk, embed, and store.  Returns a stats dict with keys
    ``files_total``, ``files_indexed``, ``files_skipped``, ``chunks``, and
    ``elapsed_seconds``.
    """
    start = time.time()
    entries = list(crawl_source(label, path, config.index_state_path))

    files_total = len(entries)
    files_indexed = 0
    files_skipped = 0
    total_chunks = 0

    for entry in entries:
        if not entry.needs_index:
            files_skipped += 1
            continue

        parser = _get_parser(entry.file_type)
        if parser is None:
            files_skipped += 1
            continue

        try:
            parsed = parser(entry.abs_path, entry.source_label, entry.source_module)
        except Exception:
            files_skipped += 1
            continue

        if not parsed:
            files_skipped += 1
            continue

        chunks = build_chunks(parsed, config)
        if chunks:
            stored = _store_chunks(client, chunks, embedder, config)
            total_chunks += stored

        files_indexed += 1

    update_state(config.index_state_path, label, entries)

    elapsed = time.time() - start
    return {
        "files_total": files_total,
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "chunks": total_chunks,
        "elapsed_seconds": round(elapsed, 2),
    }


def index_source(config: Config, label: str) -> dict:
    """Index a single source by *label*.

    Looks up *label* in ``config.doc_sources``.  Returns per-source stats.
    Raises :class:`ValueError` when *label* is not a registered source.
    """
    path = config.doc_sources.get(label)
    if path is None:
        raise ValueError(f"Unknown source label: {label!r}")

    embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)
    client = chromadb.PersistentClient(path=config.chroma_dir)

    try:
        return _index_source(label, path, config, embedder, client)
    finally:
        del embedder
        del client


def index_all(config: Config) -> dict:
    """Run the full indexing pipeline across ALL registered sources.

    Returns ``{"total_chunks": <int>, "sources": {label: stats, ...}}``.
    """
    embedder = Embedder(config.ollama_host, config.embed_model, config.embed_dim)
    client = chromadb.PersistentClient(path=config.chroma_dir)

    total_chunks = 0
    sources: dict[str, dict] = {}

    for label, path in config.doc_sources.items():
        try:
            stats = _index_source(label, path, config, embedder, client)
        except Exception:
            stats = {
                "files_total": 0,
                "files_indexed": 0,
                "files_skipped": 0,
                "chunks": 0,
                "elapsed_seconds": 0,
                "error": True,
            }
        sources[label] = stats
        total_chunks += stats.get("chunks", 0)

    del embedder
    del client

    return {"total_chunks": total_chunks, "sources": sources}
