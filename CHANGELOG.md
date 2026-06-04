# Changelog

## [Unreleased]

### Added
- Evaluation system: `python -m rag eval` CLI with Recall@K, MRR, NDCG@K metrics and latency percentiles
- Query Rewrite (rule-based): domain synonym-based query expansion for BM25 search, improving recall for natural-language queries
- `tests/eval/queries.jsonl`: annotated query evaluation dataset (35 pairs for baseline measurement)

### Changed
- `HybridRetriever.search()` now accepts `enable_rewrite` parameter (default False, True for MCP server)
- `handle_search_docs` in MCP server enables query rewrite by default
- `Config` dataclass extended with `query_rewrite_enabled` and `query_rewrite_max_variants` fields

### Added (config)
- `query_rewrite_enabled: true`
- `query_rewrite_max_variants: 3`

### Changed
- **BM25 indices are now cached in memory** — first search builds indices once; subsequent searches reuse them. Cache auto-invalidates when ChromaDB collections change (chunk count mismatch).
- **Reranker is skipped for symbol/API lookups** — exact symbol queries (`Foo::bar`, `MwMultiAxis`) and MCP tools (`get_api_class`, `get_api_function`) bypass the expensive CPU cross-encoder, falling straight through to RRF-fused BM25+vector scores.
- `search_docs` / `get_api_class` / `get_api_function` handlers pass `skip_rerank=True` for identifier queries.
- `HybridRetriever.invalidate_cache()` added; called automatically after `reindex`.

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
