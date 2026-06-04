# Changelog

## [Unreleased]

### Added
- Evaluation system: `python -m rag eval` CLI with Recall@K, MRR, NDCG@K metrics and latency percentiles
- Query Rewrite (rule-based): domain synonym-based query expansion for BM25 search, improving recall for natural-language queries
- `tests/eval/queries.jsonl`: annotated query evaluation dataset (35 pairs for baseline measurement)
- **BM25-Vector weighted RRF fusion**: BM25 contribution weight configurable via `rrf_bm25_weight` (default 2.0). Improves Recall@1 for API/symbol name queries by prioritizing exact keyword matches over semantic similarity.
- **Embedding cache**: disk-based cache keyed by `sha256(embed_text + model)`, skipping redundant Ollama embedding computation on incremental reindex. Cache-hit reindex embed phase drops from 1-2 minutes to near-instant.
- **BM25 disk persistence**: pickle tokenized corpora for fast process restart. BM25Searcher loads from disk instead of pulling full ChromaDB data, reducing first-query-after-restart latency from 1-5s to <0.1s.
- **Reranker gap skip**: automatically skip reranker when RRF top1-top2 score gap exceeds `reranker_score_gap_threshold` (default 0.15), saving ~100ms CPU inference when reranker is unlikely to change ordering.
- **Context-aware reranker candidate selection**: prioritize API chunk types (function/class/enum/macro/typedef) in reranker input, reducing max candidates from 40 to 30 (configurable via `reranker_max_candidates`). Narrative chunks fill remaining slots only when API types are exhausted.
- **File deletion auto-cleanup**: reindex detects files deleted from source directories and automatically removes stale chunks from ChromaDB, the symbol index, and the index state file.
- **Header AST chunking**: tree-sitter-cpp based C++ header parsing replaces regex heuristics, providing accurate extraction of complex templates, nested classes, macros, typedefs, and `using` declarations. Falls back to regex when tree-sitter is not installed.

### Added (config)
- `query_rewrite_enabled: true`
- `query_rewrite_max_variants: 3`
- `rrf_bm25_weight: 2.0`
- `embedding_cache_dir: ./chroma_db/embedding_cache`
- `bm25_cache_dir: ./chroma_db/bm25_cache`
- `reranker_score_gap_threshold: 0.15`
- `reranker_max_candidates: 30`

### Added (dependencies)
- `tree-sitter` and `tree-sitter-cpp` (optional, for C++ header AST parsing). Install with `pip install ".[header-ast]"`.

### Changed
- `HybridRetriever.search()` now accepts `enable_rewrite` parameter (default False, True for MCP server)
- `handle_search_docs` in MCP server enables query rewrite by default
- `Config` dataclass extended with `query_rewrite_enabled` and `query_rewrite_max_variants` fields
- **BM25 indices are now cached in memory** — first search builds indices once; subsequent searches reuse them. Cache auto-invalidates when ChromaDB collections change (chunk count mismatch).
- **Reranker is skipped for symbol/API lookups** — exact symbol queries (`Foo::bar`, `MwMultiAxis`) and MCP tools (`get_api_class`, `get_api_function`) bypass the expensive CPU cross-encoder, falling straight through to RRF-fused BM25+vector scores.
- `search_docs` / `get_api_class` / `get_api_function` handlers pass `skip_rerank=True` for identifier queries.
- `HybridRetriever.invalidate_cache()` added; called automatically after `reindex`.
- `_rrf_fuse()` now accepts `bm25_weight` parameter (backward-compatible, default 1.0)
- `Embedder` now accepts optional `cache` parameter for EmbeddingCache integration
- `BM25Searcher` now accepts optional `cache_dir` parameter for disk persistence
- `_store_chunks()` returns `(count, affected_collections)` and orchestrator builds BM25 disk cache after storing
- `HybridRetriever` passes `bm25_cache_dir` to `BM25Searcher`
- `HybridRetriever.search()` now skips reranker when RRF top1-top2 gap > `reranker_score_gap_threshold`
- `_select_for_rerank()` prioritizes API chunk types in reranker input (reduces max candidates from 40 → 30)
- `crawler.detect_deleted_files()` returns absolute paths of files removed from source directories since the last indexing run
- `crawler.remove_deleted_from_state()` cleans up stale state file entries after chunk deletion
- `orchestrator._cleanup_deleted_chunks()` removes stale ChromaDB chunks and symbol index entries before reindex (Phase 0)
- `orchestrator._index_source()` now returns `deleted` count in stats for visibility into cleanup operations
- `SymbolIndex.remove_by_files()` removes symbols by file path list (complements `remove_source()`)
- `parser_header.parse_header()` dispatches to tree-sitter-cpp AST parser when available, with transparent fallback to regex
- `parser_header` regex path now extracts `#define` macros and `typedef` declarations in addition to classes/enums/functions
- `_collect_identifiers()` extracts parameter names from tree-sitter AST nodes for accurate function signatures

## [0.1.0] — 2026-06-03

### Added
- Doxygen HTML parser (modern v1.9+, legacy v1.3, ModuleWorks CHM)
- PDF text extractor via pdfplumber
- C++ header parser (.h/.hpp/.hxx)
- Decorator-based modular parser registry
- Structured chunking with BM25 field weights
- Ollama batch embedding API (64 texts/round-trip)
- ChromaDB vector storage per source.module
- jina-reranker-v2 cross-encoder for result re-ranking
- Code boost for "how-to"/"example" queries
- Reference expansion (1-hop)
- MCP server with 11 tools
- CLI (`python -m rag status/reindex/query/symbol`)
- Incremental indexing (mtime/size fast-path + SHA1)
- 10-stage test suite (53 tests)
- Interactive `setup_config.py` wizard
- Per-phase timing breakdown in reindex output

[0.1.0]: https://github.com/Lang-sl/mcp-doc-rag/releases/tag/v0.1.0
