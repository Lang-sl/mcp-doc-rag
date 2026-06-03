# Contributing

## Setup

```bash
git clone https://github.com/Lang-sl/mcp-doc-rag.git
cd mcp-doc-rag
pip install -e .
```

## Running Tests

```bash
# Unit tests (no external dependencies — fast)
pytest tests/test_01_config.py tests/test_02_source_manager.py \
       tests/test_03_symbol_index.py tests/test_04_parser.py \
       tests/test_05_chunker.py tests/test_06_context_builder.py -v

# Integration tests (needs Ollama running)
pytest tests/test_08_embedder.py tests/test_09_search.py -v

# All except slow E2E
pytest tests/ -v -k "not slow"
```

## Architecture

```
query → BM25 + vector ANN → RRF fusion → reranker → code boost → results
```

Key components:

| Layer | Files |
|-------|-------|
| Config | `config.py` (YAML → dataclass) |
| Indexer | `crawler.py` → `parser_*.py` → `chunker.py` → `embedder.py` → `orchestrator.py` |
| Retriever | `bm25_search.py` + `vector_search.py` → `hybrid.py` → `reranker.py` |
| Server | `server.py` (MCP stdio JSON-RPC, 11 tools) |
| CLI | `cli.py` |

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
