# Changelog

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
