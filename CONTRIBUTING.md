# Contributing

## Setup

```bash
git clone https://github.com/Lang-sl/mcp-doc-rag.git
cd mcp-doc-rag
pip install -e .
```

## Running Tests

```bash
# All tests except slow E2E
pytest tests/ -q -k "not slow"

# Everything
pytest tests/ -q

# Unit tests only (stages 1-6, 12-13 — no external dependencies)
pytest tests/test_01_config.py tests/test_02_source_manager.py \
       tests/test_03_symbol_index.py tests/test_04_parser.py \
       tests/test_05_chunker.py tests/test_06_context_builder.py \
       tests/test_12_llm_rewriter.py tests/test_13_eval_trace.py -v

# Integration tests (needs Ollama running)
pytest tests/test_08_embedder.py tests/test_09_search.py \
       tests/test_10_query_rewriter.py -v

# Gateway tests (stages 14-18)
pytest tests/test_14_gateway_config.py tests/test_15_gateway_tools.py \
       tests/test_16_gateway_server.py tests/test_17_gateway_cli.py \
       tests/test_18_gateway_lifecycle.py -v
```

## Architecture

### Doc-RAG Retrieval Pipeline
```
query → BM25 + vector ANN → RRF fusion → reranker → code boost → results
```

### Gateway Architecture
```
Claude Code (MCP client)
    │
    └── Gateway MCP Server (server.py)
           ├── DocRagBackend (doc_backend.py, in-process)
           │      └── HybridRetriever + SymbolIndex
           └── CodeGraphClient (codegraph_client.py, subprocess)
                  └── npx codegraph serve --mcp
```

Key components:

| Layer | Files |
|-------|-------|
| Config | `config.py` (YAML → dataclass), `gateway/config.py` |
| Indexer | `crawler.py` → `parser_*.py` → `chunker.py` → `embedder.py` → `orchestrator.py` |
| Retriever | `bm25_search.py` + `vector_search.py` → `hybrid.py` → `reranker.py` |
| Doc-RAG Server | `server.py` (MCP stdio JSON-RPC, 11 tools) |
| Gateway Server | `gateway/server.py`, `gateway/tools.py`, `gateway/doc_backend.py`, `gateway/codegraph_client.py`, `gateway/codegraph_lifecycle.py` |
| CLI | `cli.py` (dispatches to `rag server`, `rag gateway`, etc.) |
| Eval | `eval.py` |

## Adding a New Parser

1. Create `src/rag/indexer/parser_<type>.py`
2. Implement `parse_<type>(file_path, source_label, source_module) -> list[dict]`
3. Decorate with `@register_parser(file_type="<type>", extensions=[...])`
4. Import in `src/rag/indexer/__init__.py`

## Pull Requests

- Keep PRs focused — one concern per PR
- Add tests for new features
- Run `pytest tests/ -v -k "not slow"` before submitting
- Match existing code style (Allman braces equivalent — opening on new line, 4-space indent, CRLF)
