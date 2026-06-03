# mcp-doc-rag

**Fully local, MCP-integrated RAG system for C/C++ SDK documentation retrieval.**

A retrieval-augmented generation (RAG) engine that indexes C++ SDK documentation (Doxygen HTML, PDFs, C++ headers) and exposes hybrid search via an MCP (Model Context Protocol) server — enabling AI coding assistants like Claude Code to retrieve precise API documentation on demand.

## Why mcp-doc-rag

- **100% Local** — No cloud API calls. Embeddings via Ollama, vectors in ChromaDB, reranker from HuggingFace. All data stays on your machine.
- **MCP-Native** — Designed as an MCP server first. Claude Code (and other MCP clients) can auto-invoke RAG tools during coding.
- **Hybrid Search** — Combines field-weighted BM25 (symbol×10, signature×5) + vector ANN → RRF fusion → jina-reranker cross-encoder → code boost → reference expansion.
- **Structured Chunking** — Doxygen-aware HTML parser extracts symbol_id, class, function, signature, params, return type, remarks, and code examples into structured JSON chunks.
- **O(1) Symbol Lookup** — Exact symbol ID lookup via in-memory hash index, bypassing full search for known API names.
- **Incremental Indexing** — SHA1 content hashing with mtime/size pre-filter. Only re-indexes changed files.
- **Customizable** — Add/remove document sources at runtime via MCP tools or CLI.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | Ollama `nomic-embed-text` (768d, ~275MB) |
| Vector DB | ChromaDB (per-source.module collections) |
| Reranker | `jinaai/jina-reranker-v2-base-multilingual` (HuggingFace) |
| Keyword Search | `rank-bm25` (field-weighted multi-index) |
| Cross-Encoder Runtime | `transformers` + `torch` + `einops` |
| PDF Extraction | `pdfplumber` |
| HTML Parsing | `BeautifulSoup4` (Doxygen structure-aware) |
| Integration | MCP Server (stdio JSON-RPC) |
| Config | YAML |

## Prerequisites

- **Python** >= 3.11
- **Ollama** (for embeddings)

### Install Ollama

```bash
# Windows
winget install Ollama.Ollama

# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Set model storage path (optional, defaults to Ollama's default location):

```bash
# Windows
setx OLLAMA_MODELS "C:\path\to\models"

# macOS / Linux
export OLLAMA_MODELS=/path/to/models
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mcp-doc-rag.git
cd mcp-doc-rag

# Install in development mode
pip install -e .
```

**Note:** The first time the reranker is used, it will automatically download the jina-reranker model (~1.1GB) from HuggingFace. This is a one-time download. If the reranker is unavailable (e.g., transformers version mismatch), search degrades gracefully — RRF fusion scores are used directly.

## Quick Start

### 1. Run the Setup Wizard

```bash
python setup_config.py
```

This interactive script will:
- Create `config.yaml` from the template
- Help you add document source paths
- Verify Ollama is running

Alternatively, copy and edit the template manually:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml: set doc_sources paths
```

### 2. Index Documents

```bash
# Full index (incremental — skips unchanged files)
python -m rag reindex

# Index a single source
python -m rag reindex --source my_sdk

# Force full rebuild (ignores cached hashes)
python -m rag reindex --full
```

### 3. Search

```bash
# Hybrid search
python -m rag query "How to initialize the rendering kernel"

# With source filter
python -m rag query --source my_sdk "CalculateToolpath"

# Exact symbol lookup
python -m rag symbol MwMultiAxis::CalculateToolpath

# Build context block (prompt-ready)
python -m rag context "5-axis simulation setup"
```

### 4. Manage Sources

```bash
python -m rag source add my_new_docs D:/path/to/docs
python -m rag source list
python -m rag source remove my_new_docs
```

### 5. Check Status

```bash
python -m rag status
# Output: total_chunks, per_source breakdown, collections, symbol count
```

## Configuration Reference

Full `config.yaml` with defaults (see also `config.example.yaml`):

```yaml
# ---- Paths (relative to project root) ----
chroma_dir: ./chroma_db
symbol_index_path: ./symbol_index.json

# ---- Document Sources (label: path) ----
doc_sources:
  my_sdk: /absolute/path/to/your/docs

# ---- Ollama ----
ollama_host: http://localhost:11434
embed_model: nomic-embed-text
embed_dim: 768
embed_batch_size: 32

# ---- Reranker ----
reranker_model: jinaai/jina-reranker-v2-base-multilingual
reranker_max_length: 512

# ---- Chunking ----
chunk_max_chars: 2000

# ---- Retrieval ----
top_k_default: 10
candidate_expand_factor: 4
rrf_k: 30

# ---- BM25 Field Weights ----
bm25_weights:
  symbol_name: 10.0
  signature: 5.0
  remarks: 1.0
  example: 0.5

# ---- Code Boost ----
code_boost_ratio: 1.20
code_boost_triggers:
  - "how to"
  - "create"
  - "example"
  - "sample"
  - "write"
  - "implement"
  - "setup"

# ---- Reference Expansion ----
ref_expansion_max: 5

# ---- Context Building ----
context_max_tokens: 6000

# ---- Cache ----
cache_max_entries: 128

# ---- Index ----
index_batch_size: 100
```

## MCP Server Integration

mcp-doc-rag is an MCP server — AI coding assistants can call its tools directly.

### Configuration for Claude Code

Add to `.claude/settings.local.json` (or `settings.json`):

```json
{
  "mcpServers": {
    "mcp-doc-rag": {
      "command": "python",
      "args": ["-m", "rag.server"],
      "cwd": "/path/to/mcp-doc-rag",
      "env": {
        "RAG_CONFIG_PATH": "/path/to/mcp-doc-rag/config.yaml"
      }
    }
  }
}
```

Replace `/path/to/mcp-doc-rag` with the absolute path to your cloned repository.

After configuration, restart Claude Code or use `/mcp-reload`.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `find_symbol` | O(1) exact symbol lookup by symbol_id |
| `search_docs` | Full hybrid search pipeline |
| `get_api_class` | Get complete class documentation with members |
| `get_api_function` | Get complete function documentation |
| `list_modules` | List all known sub-module names |
| `build_context` | Build token-bounded prompt-ready context block |
| `add_doc_source` | Register a new document source |
| `remove_doc_source` | Remove a source and its chunks |
| `list_doc_sources` | List all registered sources |
| `reindex` | Incremental reindex (per-source or all) |
| `index_status` | Aggregate index statistics |

### Example Claude Code Usage

```
User: "How do I use MwMultiAxis::CalculateToolpath?"

Claude internally calls:
  find_symbol("MwMultiAxis::CalculateToolpath")
  get_api_function("CalculateToolpath", class_name="MwMultiAxis")
  → Returns signature, params, remarks, example code

User gets a precise answer with code examples from the actual SDK docs.
```

## Document Format Support

| Format | Detection | Extracted |
|--------|-----------|-----------|
| Doxygen HTML (modern) | `<meta generator="Doxygen">`, `.memitem` | functions, classes, enums, macros, params, return types |
| Doxygen HTML (legacy) | `<TITLE>Function:...</TITLE>` | functions, signatures, params, descriptions |
| ModuleWorks CHM | `chm-test.js`, `id="main_chm"` | functions, classes, enums, params |
| PDF | `.pdf` extension | page-paragraph sections |
| C++ Headers | `.h`, `.hpp`, `.hxx` | function/class/enum signatures with comments |

### Adding Custom Parsers

Parsers use decorator-based registration — add a new format without touching the orchestrator or crawler:

```python
# parser_markdown.py
from rag.indexer.parser_registry import register_parser

@register_parser(file_type="markdown", extensions=[".md", ".mdx"])
def parse_markdown(file_path, source_label, source_module):
    # parse .md files → return list[dict]
    ...
```

Then import it in `src/rag/indexer/__init__.py` to trigger registration. The crawler auto-discovers `.md` files, and the orchestrator auto-picks the right parser.

## Retrieval Pipeline

```
Query
  │
  ├─ [CACHE] LRU check (128 entries)
  │
  ├─ BM25 (field-weighted, per-collection)
  ├─ Vector ANN (per-collection ChromaDB)
  │
  └─ RRF Fusion (k=30) → 80 candidates
      │
      ├─ Reranker (jina-reranker-v2 cross-encoder, graceful fallback)
      ├─ Code Boost (+20% on trigger words)
      ├─ Reference Expansion (1-hop, max +5)
      │
      └─ Top-K Results
```

## Important Tips

### Performance

- **First index.** Embedding runs in a single global batch via Ollama's batch API — ~10k chunks takes 1-2 minutes for embedding, not tens of minutes. Subsequent incremental indexes are fast (seconds — unchanged files are skipped via mtime/size pre-check).
- **Pipeline phases.** `python -m rag reindex` prints per-phase timing (crawl, parse, chunk, embed, chroma) so you can see exactly where time is spent. Embedding is typically < 30% of total time.
- **Ollama must be running.** Start it with `ollama serve` or ensure the Windows service is running.
- **Reranker download.** The first `search_docs` call will download the jina-reranker model (~1.1GB). This is one-time. Pre-download by running a test search after indexing. The reranker includes an automatic compatibility patch for transformers >= 4.46.
- **ChromaDB storage.** The vector database defaults to `./chroma_db` inside the project directory. It can grow to several GB for large doc sets — configure `chroma_dir` in config.yaml if you need it elsewhere.

### Document Sources

- **Auto module detection.** Sub-modules are auto-detected from directory structure (first path component). For example, `5axis/5axcore/public/mwmulti.h` → module `5axis`.
- **Add sources at any time.** Use `add_doc_source` MCP tool or CLI. Run `reindex --source <label>` after adding.
- **Supported extensions.** `.html`, `.htm`, `.pdf`, `.h`, `.hpp`, `.hxx`. Other files are skipped.

### Search Quality

- **Use exact symbol names** when possible. `find_symbol` → `get_api_class` is the fast path.
- **Code boost triggers** — queries containing "how to", "example", "create", "implement", etc. get a +20% boost on code-containing chunks.
- **Reference expansion** — top results automatically pull in referenced symbols (1-hop, +5 max). This is most effective when your source docs have `see_also` sections.
- **BM25 weights** — symbol_name (×10) and signature (×5) are weighted higher than remarks (×1) and examples (×0.5) for API-focused searches. Adjust in config.yaml for narrative-heavy docs.

### Troubleshooting

- **"0 chunks" after indexing.** Check that your document paths in config.yaml exist and contain supported file types. Run `python -m rag status` to verify.
- **Ollama connection errors.** Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- **Import errors.** Run `pip install -e .` from the project directory to ensure all dependencies are installed.
- **Symbol index is empty.** The symbol index is built after `reindex` completes. If indexing was interrupted, run `python -m rag reindex` again.

## Step-by-Step Verification (Tests)

The test suite is organized into 10 numbered stages — run them in order to verify each layer of the system. Each stage builds on the previous one.

### Quick Run

```bash
# Stages 1–7: pure unit + file-crawler tests (no Ollama needed)
pytest tests/test_01_config.py tests/test_02_source_manager.py \
       tests/test_03_symbol_index.py tests/test_04_parser.py \
       tests/test_05_chunker.py tests/test_06_context_builder.py \
       tests/test_07_crawler.py -v

# Stage 8: embedding (needs Ollama running)
pytest tests/test_08_embedder.py -v

# Stage 9: search pipeline (needs Ollama + indexed data)
pytest tests/test_09_search.py -v

# Stage 10: full end-to-end (slow — needs everything)
pytest tests/test_10_e2e.py -v -m slow

# Run everything except slow E2E
pytest tests/ -v -k "not slow"
```

### Stage Reference

| Stage | File | What It Verifies | Prerequisites |
|-------|------|-----------------|---------------|
| 1 | `test_01_config.py` | Default values, YAML loading, env var override, BM25 weights | None |
| 2 | `test_02_source_manager.py` | Source CRUD: add, remove, list, duplicate detection | None |
| 3 | `test_03_symbol_index.py` | O(1) hash-map lookup, source-scoped removal | None |
| 4 | `test_04_parser.py` | Type name extraction, narrative HTML, Doxygen function parsing | None |
| 5 | `test_05_chunker.py` | Chunk assembly, BM25 fields, embed text, discard rules | None |
| 6 | `test_06_context_builder.py` | Context formatting, token cap, score ordering | None |
| 7 | `test_07_crawler.py` | File discovery, SHA1 incremental check, second-pass skip | Real doc files at configured paths |
| 8 | `test_08_embedder.py` | Embedding dimension, batch count, offline error handling | Ollama + `nomic-embed-text` |
| 9 | `test_09_search.py` | Vector ANN, BM25 keyword, hybrid pipeline, symbol lookup | Stage 8 + indexed ChromaDB data |
| 10 | `test_10_e2e.py` | Full pipeline: index small doc set → search → verify | Stage 8 + document files |

**Stage 1–6** run instantly (no network, no disk I/O beyond temp files). If any of these fail, you have a code or dependency issue.

**Stage 7** verifies the file crawler against real documents. Set `RAG_TEST_DOC_DIR` to a directory containing Doxygen HTML or PDF files. Without it, tests auto-skip.

**Stage 8** is your first Ollama dependency. If tests skip with "Ollama not running", start it with `ollama serve` and ensure `nomic-embed-text` is pulled.

**Stage 9** requires indexed data. Run `python -m rag reindex` before these tests.

**Stage 10** is marked `@pytest.mark.slow` — it indexes a small document subset and runs the full search pipeline. Set `RAG_TEST_DOC_DIR` before running. Useful as a pre-release smoke test.

### Interpreting Results

```
53 passed, 1 deselected  ← ✅ All systems operational
40 passed, 13 skipped    ← ⚠️ Stages 8+ skipped. Check Ollama and index.
3 failed, 50 passed      ← ❌ Failures indicate specific component issues.
                            Run stages individually to isolate.
```

## Project Structure

```
mcp-doc-rag/
├── pyproject.toml
├── setup_config.py            # Interactive config setup wizard
├── config.example.yaml        # Configuration template
├── .gitignore
├── LICENSE
├── README.md
├── tests/
│   ├── conftest.py              # Shared fixtures, Ollama detection
│   ├── test_01_config.py        # Stage 1: Config loading
│   ├── test_02_source_manager.py # Stage 2: Source CRUD
│   ├── test_03_symbol_index.py  # Stage 3: Symbol index
│   ├── test_04_parser.py        # Stage 4: HTML parser
│   ├── test_05_chunker.py       # Stage 5: Chunk assembly
│   ├── test_06_context_builder.py # Stage 6: Context builder
│   ├── test_07_crawler.py       # Stage 7: File crawler
│   ├── test_08_embedder.py      # Stage 8: Embedding
│   ├── test_09_search.py        # Stage 9: Search pipeline
│   └── test_10_e2e.py           # Stage 10: Full E2E (slow)
└── src/rag/
    ├── server.py              # MCP Server (11 tools, stdio JSON-RPC)
    ├── cli.py                 # CLI entry point
    ├── config.py              # YAML config loader
    ├── models.py              # Chunk, SearchResult, IndexStats dataclasses
    ├── symbol_index.py        # O(1) symbol hash map
    ├── source_manager.py      # CRUD for doc sources
    ├── context_builder.py     # Token-bounded context formatter
    ├── indexer/
    │   ├── crawler.py           # File walker with SHA1 incremental check
    │   ├── parser_registry.py   # Decorator-based parser registration
    │   ├── parser_html.py       # Doxygen HTML parser (4 formats)
    │   ├── parser_pdf.py        # PDF text extractor
    │   ├── parser_header.py     # C++ header signature extractor
    │   ├── chunker.py           # Structured chunk assembler
    │   ├── embedder.py          # Ollama batch embedding wrapper
    │   └── orchestrator.py      # Full index pipeline
    └── retriever/
        ├── vector_search.py   # ChromaDB ANN per collection
        ├── bm25_search.py     # Field-weighted BM25
        ├── hybrid.py          # Full pipeline orchestration
        └── reranker.py        # jina-reranker-v2 cross-encoder
```

## License

MIT — see [LICENSE](LICENSE) for details.
