# mcp-doc-rag

[![CI](https://github.com/Lang-sl/mcp-doc-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Lang-sl/mcp-doc-rag/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-native-purple)](https://modelcontextprotocol.io/)
[![中文文档](https://img.shields.io/badge/docs-中文-blue)](README.zh-CN.md)

**Local-first MCP server for C/C++ SDK documentation retrieval, with optional source code knowledge via CodeGraph gateway.**

A retrieval-augmented generation (RAG) engine that indexes C++ SDK documentation (Doxygen HTML, PDFs, C++ headers) and exposes search via an MCP (Model Context Protocol) server — enabling AI coding assistants like Claude Code to retrieve precise API documentation on demand. Deploy it standalone for doc-only search, or via the gateway to combine documentation with source code analysis.

## Deployment Modes

mcp-doc-rag offers two deployment modes. Choose based on whether you need source code analysis alongside documentation.

| Mode | Command | What You Get | Requirements |
|------|---------|-------------|--------------|
| **Doc-RAG** (standalone) | `python -m rag server` | 11 doc search tools — hybrid BM25+vector retrieval, O(1) symbol lookup, context builder | Python 3.11+, Ollama |
| **Gateway Adapter** (recommended) | `python -m rag adapter` | All doc tools + `smart_search` + CodeGraph lifecycle, daemon-backed, shared across MCP sessions | Python 3.11+, Ollama, Node.js/npm (optional) |
| **Gateway** (direct stdio) | `python -m rag gateway` | Same tools as adapter, but one process per MCP client — fallback for compatibility | Python 3.11+, Ollama, Node.js/npm (optional) |

The gateway adapter starts a long-lived daemon that MCP clients connect to via loopback HTTP. The daemon keeps doc-rag indexes and optional CodeGraph subprocesses alive across MCP sessions. Use `python -m rag daemon status` to check daemon health. CodeGraph is optional in all gateway modes — without it, you still get full doc-rag tools and `smart_search` with graceful degradation.

## Why mcp-doc-rag

- **100% Local** — No cloud API calls. Embeddings via Ollama, vectors in ChromaDB, reranker from HuggingFace. All data stays on your machine.
- **MCP-Native** — Designed as an MCP server first. Claude Code (and other MCP clients) can auto-invoke RAG tools during coding.
- **Hybrid Search** — Combines field-weighted BM25 (symbol×10, signature×5) + vector ANN → RRF fusion → conditional jina-reranker cross-encoder → code boost → reference expansion. Reranker is automatically skipped for symbol/API identifier queries (e.g. `MwMultiAxis::CalculateToolpath`) to keep latency low.
- **Structured Chunking** — Doxygen-aware HTML parser and tree-sitter-cpp C++ header parser extract symbol_id, class, function, signature, params, return type, remarks, and code examples into structured JSON chunks. Tree-sitter provides AST-level accuracy for complex templates and nested classes; falls back to regex when tree-sitter is unavailable.
- **O(1) Symbol Lookup** — Exact symbol ID lookup via in-memory hash index, bypassing full search for known API names.
- **Incremental Indexing** — SHA1 content hashing with mtime/size pre-filter. Only re-indexes changed files. Automatically detects and cleans up chunks from deleted files.
- **Customizable** — Add/remove document sources at runtime via MCP tools or CLI.
- **Optional CodeGraph Integration** — When deployed via the gateway, combine documentation search with source code analysis. `smart_search` queries CodeGraph for code usages, extracts symbol names, and maps them back to API documentation — so you can ask "how is `Renderer::Initialize` used in the codebase?" and get both the implementation patterns and the reference docs in one response. If CodeGraph is not installed or fails to start, the gateway degrades gracefully to doc-only search.

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
| C++ Header Parsing | `tree-sitter-cpp` (AST-level, falls back to regex) |
| Gateway (optional) | MCP stdio JSON-RPC, subprocess management for CodeGraph |
| CodeGraph (optional) | `@colbymchenry/codegraph` via npx (TypeScript, external MCP server) |
| Integration | MCP Server (stdio JSON-RPC) |
| Config | YAML |

## Prerequisites

- **Python** >= 3.11
- **Ollama** (for embeddings)
- **NVIDIA GPU + CUDA** (recommended for reranker; CPU fallback works but is slower)

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
setx OLLAMA_MODELS "<path-to-models>"

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
git clone https://github.com/Lang-sl/mcp-doc-rag.git
cd mcp-doc-rag

# Install in development mode
pip install -e .
```

### Optional: Improved C++ Header Parsing

Install `tree-sitter-cpp` for AST-level header parsing (complex templates, nested classes, macros):

```bash
pip install -e ".[header-ast]"
```

Without this, the system falls back to regex-based parsing which handles most cases but may be less accurate for complex C++ constructs.

### GPU Acceleration (Recommended)

The reranker runs ~500× faster on GPU vs CPU. Install the CUDA-enabled PyTorch:

```bash
# Verify your NVIDIA GPU is detected
nvidia-smi

# Uninstall CPU-only PyTorch if present, then install CUDA version
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** The `cu124` index above targets CUDA 12.4 — adjust to match your driver version.
> Visit [pytorch.org](https://pytorch.org/get-started/locally/) to see the latest available builds.
> CUDA 11.8 users would use `cu118`, CUDA 12.6 users `cu126`, etc.

If no NVIDIA GPU is available, the default CPU PyTorch works — reranker queries will take 2-5s each instead of 5-20ms. See [Performance](#performance) for benchmarks.

**Note:** The first time the reranker is used, it will automatically download the jina-reranker model (~1.1GB) from HuggingFace. This is a one-time download. The first inference call includes a ~10s JIT compilation warmup on GPU. If the reranker is unavailable (e.g., transformers version mismatch), search degrades gracefully — RRF fusion scores are used directly.

### Optional: CodeGraph (for Gateway Mode)

CodeGraph adds source code search to the gateway. It's not a Python dependency — the gateway launches it through npm. Skip this section if you're using standalone Doc-RAG.

**Requirements:** Node.js 18+ with npm/npx on `PATH`.

When enabled, the gateway runs:

```bash
npx -y @colbymchenry/codegraph@0.9.9 serve --mcp
```

Version `0.9.9` is the current verified CodeGraph release. Update it deliberately and re-check the CodeGraph MCP contract when upgrading. The `-y` flag lets npx fetch the package automatically on first use. If CodeGraph is not configured or cannot start, the gateway degrades to doc-only search automatically.

## Quick Start

### 1. Run the Setup Wizard

```bash
python setup_config.py
```

This interactive script will:
- Create `config.yaml` from the template
- Help you add document source paths
- Optionally create `gateway.yaml` for CodeGraph gateway search
- Verify Ollama is running

Alternatively, copy and edit the template manually:

```bash
cp src/rag/config.example.yaml config.yaml
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
python -m rag query --source my_sdk "Initialize renderer"

# Exact symbol lookup
python -m rag symbol MySDK::Renderer::Initialize

# Build context block (prompt-ready)
python -m rag context "5-axis simulation setup"
```

### 4. Manage Sources

```bash
python -m rag source add my_new_docs <path-to-docs>
python -m rag source list
python -m rag source remove my_new_docs
```

### 5. Check Status

```bash
python -m rag status
# Output: total_chunks, per_source breakdown, collections, symbol count
```

### 6. Start the MCP Server

Start the standalone doc-rag MCP server and connect it to Claude Code. Create `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "mcp-doc-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "rag.server"],
      "cwd": "<absolute-path-to-mcp-doc-rag>",
      "env": {
        "RAG_CONFIG_PATH": "<absolute-path-to-mcp-doc-rag>/config.yaml"
      }
    }
  }
}
```

Restart Claude Code and use `/mcp` to verify the server is loaded. You'll have 11 doc search tools available.

### Adding CodeGraph (Optional)

To add source code search on top of doc search, switch to the gateway adapter (recommended) or direct gateway:

1. Copy and edit the gateway config template:

```bash
cp src/rag/gateway.example.yaml gateway.yaml
# Edit gateway.yaml: set doc_rag.config_path and codegraph.cwd (optional)
```

Minimal `gateway.yaml`:

```yaml
doc_rag:
  config_path: "<absolute-path-to-mcp-doc-rag>/config.yaml"
# Optional CodeGraph integration. Remove this section for doc-only gateway.
codegraph:
  command: "npx"
  args: ["-y", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"]
  cwd: "<absolute-path-to-your-code-project>"
daemon:
  autostart: true
  host: "127.0.0.1"
  port: 0
```

If `codegraph` is omitted or cannot start, the gateway degrades to doc-only search.

2. Update `.mcp.json` to use the gateway adapter (recommended) or direct gateway:

```json
{
  "mcpServers": {
    "mcp-doc-rag-gateway": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "rag", "adapter"],
      "cwd": "<absolute-path-to-mcp-doc-rag>",
      "env": {
        "GATEWAY_CONFIG_PATH": "<absolute-path-to-mcp-doc-rag>/gateway.yaml"
      }
    }
  }
}
```

3. Build the CodeGraph index (first time only):

```
In Claude Code: "Run codegraph_init to index my code project"
```

This initializes the code knowledge graph. After that, use `smart_search` to query both code and docs at once.

### 7. Evaluate Search Quality

```bash
# 1. Copy and annotate the template with your own chunk_ids
cp tests/eval/queries.template.jsonl tests/eval/queries.jsonl
# Edit queries.jsonl: fill relevant_chunk_ids for each query
# (search each query with "python -m rag query <query>" to find relevant chunks)

# 2. Run evaluation
python -m rag eval --queries tests/eval/queries.jsonl

# With source scope
python -m rag eval --queries tests/eval/queries.jsonl --source my_sdk

# 3. Record baseline for future comparison
python -m rag eval --queries tests/eval/queries.jsonl > tests/eval/baseline.txt

# 4. Compare rewrite modes
python -m rag eval --queries tests/eval/queries.jsonl --compare-rewrite

# 5. Bad case analysis only
python -m rag eval --queries tests/eval/queries.jsonl --bad-cases-only
```

Metrics: Recall@1/3/5/10, MRR, NDCG@5/10, latency p50/p95, per-stage Recall@5/10/MRR (bm25/vector/rrf/reranker/final), bad case classification (knowledge_gap, ranking_failure, rewrite_regression, reranker_regression).

The template contains 35 generic queries (12 API lookups + 23 natural language) —
replace the API symbols with ones from your SDK and adjust queries to your domain.

## Configuration Reference

Full `config.yaml` with defaults (see also `config.example.yaml`):

```yaml
# ---- Paths (relative to project root) ----
chroma_dir: ./output/chroma_db
symbol_index_path: ./output/symbol_index.json

# ---- Document Sources (label: path) ----
doc_sources:
  my_sdk: /absolute/path/to/your/docs

# ---- Ollama ----
ollama_host: http://localhost:11434
embed_model: nomic-embed-text
embed_dim: 768
embed_batch_size: 256

# ---- Reranker ----
reranker_model: jinaai/jina-reranker-v2-base-multilingual
reranker_max_length: 512

# ---- Chunking ----
chunk_max_chars: 2000

# ---- Retrieval ----
top_k_default: 10
candidate_expand_factor: 4
rrf_k: 30
rrf_bm25_weight: 2.0  # >1.0 = BM25 keyword matches weighted higher in RRF

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

# ---- Query Rewrite ----
query_rewrite_enabled: false            # off by default — no rewrite is best for most queries
query_rewrite_max_variants: 3
query_rewrite_llm_model: null           # optional local model for LLM rewrite
query_rewrite_llm_timeout_ms: 2000      # max wait for LLM response

# ---- Cache ----
cache_max_entries: 128
embedding_cache_dir: ./output/chroma_db/embedding_cache
bm25_cache_dir: ./output/chroma_db/bm25_cache

# ---- Reranker Optimization ----
reranker_score_gap_threshold: 0.15  # 0 = never skip reranker
reranker_max_candidates: 30          # max candidates for reranker (API prioritized)

# ---- Index ----
index_batch_size: 500
```

## MCP Server Integration

The `.mcp.json` configuration is covered in [Quick Start](#6-start-the-mcp-server). Here's a reference of all available tools and example usage.

### Doc-RAG Tools (Standalone & Gateway)

These tools are available in both standalone Doc-RAG and gateway mode:

| Tool | Description |
|------|-------------|
| `find_symbol` | O(1) exact symbol lookup by symbol_id |
| `search_docs` | Full hybrid search pipeline (BM25+vector→RRF→reranker→boost→expand) |
| `get_api_class` | Get complete class documentation with members |
| `get_api_function` | Get complete function documentation |
| `list_modules` | List all known sub-module names |
| `build_context` | Build token-bounded prompt-ready context block |
| `add_doc_source` | Register a new document source |
| `remove_doc_source` | Remove a source and its chunks |
| `list_doc_sources` | List all registered sources |
| `reindex` | Incremental reindex (per-source or all) |
| `index_status` | Aggregate index statistics |

### Gateway-Only Tools

These additional tools are available in gateway mode. CodeGraph lifecycle tools are present only when CodeGraph is configured in `gateway.yaml`.

| Tool | Description |
|------|-------------|
| `smart_search` | Flagship gateway tool: queries CodeGraph for code usages, extracts symbols, maps them to doc-rag API docs, returns merged code + doc results |
| `codegraph_init` | Initialize and build the first CodeGraph index for the configured project |
| `codegraph_reindex` | Full CodeGraph index rebuild (with optional `--force`) |
| `codegraph_sync` | Incremental CodeGraph sync — restarts the subprocess if changes detected |
| `codegraph_index_status` | CodeGraph index health + gateway subprocess health |
| `codegraph_restart` | Restart the CodeGraph MCP subprocess |

In gateway mode, CodeGraph's native tools are also dynamically exposed alongside these gateway tools.

### smart_search Flow (Gateway Only)

`smart_search` orchestrates a two-phase search:

```
User query: "How is Renderer::Initialize called in practice?"
    │
    ├─[1] Query CodeGraph for code usages
    │      └─ If CodeGraph unavailable: degrade to doc-only search
    ├─[2] Extract symbol names from CodeGraph results
    │      (recursive JSON scan + Markdown heading parse)
    ├─[3] Normalize and probe doc-rag SymbolIndex
    │      (exact match → template-strip → unqualified member)
    ├─[4] Fetch docs for matched API symbols via search_docs
    └─[5] Return merged result:
           code_usages + matched_api_symbols + unmatched_code_symbols + doc_results
```

The response includes `degraded: true` and a warning when CodeGraph is unavailable, so the client always knows what data source was used.

### Example Claude Code Usage

```
User: "How do I use MySDK::Renderer::Initialize?"

Claude internally calls:
  find_symbol("MySDK::Renderer::Initialize")
  get_api_function("Initialize", class_name="MySDK::Renderer")
  → Returns signature, params, remarks, example code

User gets a precise answer with code examples from the actual SDK docs.
```

With gateway + CodeGraph:

```
User: "Show me how Renderer::Initialize is used in the codebase,
       and what the docs say about its parameters."

Claude calls:
  smart_search("Renderer::Initialize usage patterns", top_k=10)
  → code_usages: [3 code locations calling Initialize]
  → matched_api_symbols: ["MySDK::Renderer::Initialize"]
  → doc_results: [signature, parameter docs, return type, examples]
```

### Configuration Scopes

| Scope | File Location | Use Case |
|-------|--------------|----------|
| **project** (recommended) | `.mcp.json` at project root | Team-shared, commit to version control |
| **user** | `~/.claude/mcp.json` | Personal tools available across all projects |
| **local** | `~/.config/claude/mcp.json` | Machine-specific configs, local credentials |

After configuration, restart Claude Code. Use `/mcp` to verify the server is loaded.

### Gateway Configuration Reference

Full `gateway.yaml` reference (see also `src/rag/gateway.example.yaml`):

```yaml
# Path to the doc-rag config.yaml (required)
# If omitted, falls back to RAG_CONFIG_PATH env var or ./config.yaml
doc_rag:
  config_path: "<absolute-path-to-mcp-doc-rag>/config.yaml"

# Optional CodeGraph MCP server. Omit this section for doc-only gateway mode.
codegraph:
  command: "npx"                                          # CodeGraph launcher
  args:                                                   # args passed to command
    - "-y"                                                # auto-fetch package
    - "@colbymchenry/codegraph@0.9.9"                     # pinned version
    - "serve"                                             # MCP serve mode
    - "--mcp"                                             # stdio JSON-RPC
  cwd: "<absolute-path-to-code-project>"                  # project to index

# Gateway daemon settings. The adapter autostarts a long-lived daemon
# that keeps indexes and CodeGraph subprocesses alive across sessions.
daemon:
  autostart: true                                         # auto-start daemon when adapter connects
  host: "127.0.0.1"                                       # loopback only
  port: 0                                                 # 0 = OS-assigned port
  runtime_dir: "<absolute-path-to-mcp-doc-rag>/output/runtime"  # runtime metadata + logs
```

If `codegraph` is omitted or the subprocess fails to start, the gateway operates in doc-only mode — all doc tools work normally, `smart_search` returns doc results with `degraded: true`.

Use `python -m rag daemon status` to check daemon health, `python -m rag daemon stop` to shut it down, and `python -m rag daemon reload` to pick up config changes without restarting.

## Document Format Support

| Format | Detection | Extracted |
|--------|-----------|-----------|
| Doxygen HTML (modern) | `<meta generator="Doxygen">`, `.memitem` | functions, classes, enums, macros, params, return types |
| Doxygen HTML (legacy) | `<TITLE>Function:...</TITLE>` | functions, signatures, params, descriptions |
| CHM-based HTML | `<meta name="generator" content="...">` | functions, classes, enums, params |
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
  ├─ [REWRITE] Query expansion with domain synonyms (configurable)
  │
  ├─ BM25 (field-weighted, per-collection, multi-variant if rewritten)
  ├─ Vector ANN (per-collection ChromaDB)
  │
  └─ RRF Fusion (k=30) → up to 80 candidates
      │
      ├─ [GAP CHECK] Skip reranker if top1-top2 RRF gap > threshold (configurable)
      ├─ [API PRIORITY] Select top candidates prioritizing API chunk types
      ├─ Reranker (jina-reranker-v2 cross-encoder, graceful fallback)
      ├─ Code Boost (+20% on trigger words)
      ├─ Reference Expansion (1-hop, max +5)
      │
      └─ Top-K Results
```

## Performance

Benchmarks on a mid-range NVIDIA GPU + Ollama `nomic-embed-text`:

| Stage | GPU (CUDA) | CPU (no GPU) |
|-------|-----------|--------------|
| Vector embedding (per query) | ~50ms | ~50ms |
| BM25 keyword search | ~30ms | ~30ms |
| RRF fusion + code boost + ref expand | < 5ms | < 5ms |
| **Reranker (30 candidates)** | **~15ms** | ~4,000ms |
| **Total search latency (p50)** | **~100ms** | ~5,200ms |
| First query (model load + JIT warmup) | ~10s | ~30s |

Key takeaways:
- With GPU, the reranker adds negligible overhead — full hybrid search in ~100ms.
- Without GPU, reranker dominates latency (~5s per query). Set up GPU PyTorch for production use, or temporarily skip reranker with `skip_rerank=True`.
- BM25 and vector search are independent of GPU and always fast.

### Evaluation Baseline

Measured on a production-scale C++ SDK documentation index with **108 annotated queries** (35 API symbol lookups + 73 natural language), covering 3 document sources across 7,834 indexed chunks. RRF weighting (`rrf_bm25_weight: 2.0`) significantly improves early-position recall by prioritizing exact keyword matches.

| Metric | No Rewrite | Rule Rewrite | LLM Rewrite (qwen2.5:3b) |
|--------|-----------|-------------|--------------------------|
| Recall@1 | 0.362 | 0.360 | 0.352 |
| Recall@3 | 0.513 | 0.486 | 0.492 |
| Recall@5 | 0.593 | 0.548 | 0.564 |
| Recall@10 | 0.722 | 0.678 | 0.659 |
| MRR | 0.644 | 0.614 | 0.616 |
| NDCG@5 | 0.676 | 0.635 | 0.642 |
| NDCG@10 | 0.695 | 0.659 | 0.662 |
| p50 latency | 288ms | 324ms | 2,874ms |
| p95 latency | 455ms | 646ms | 3,699ms |
| Zero-recall | 13/108 (12%) | 19/108 (18%) | 18/108 (17%) |

**Key findings:**
- **No rewrite is best** for this dataset — BM25-weighted RRF alone achieves Recall@10 0.722 with the fewest zero-recall queries (13)
- **LLM rewrite** (`qwen2.5:3b`) slightly outperforms rule-based rewrite on Recall@5/NDCG but adds ~2.5s per query. Best suited for offline/hard queries where latency isn't critical
- API symbol lookups (35 queries) maintain near-perfect Recall@1 regardless of rewrite mode
- Per-stage evals reveal that rewrite dilution hurts BM25 recall in both modes; focus future work on targeted expansion that preserves keyword precision

#### Per-Stage Breakdown (No Rewrite)

`python -m rag eval` now shows where each pipeline stage contributes:

| Stage | Recall@5 | Recall@10 | MRR |
|-------|----------|-----------|-----|
| BM25 | 0.535 | 0.700 | 0.641 |
| Vector | 0.433 | 0.534 | 0.456 |
| RRF | 0.829 | 0.896 | 0.863 |
| Reranker | 0.585 | 0.719 | 0.649 |
| **Final** | **0.593** | **0.722** | **0.644** |

Key insight: RRF fusion combines the best of both channels (0.896 Recall@10), the reranker trims noise at the cost of some recall, and final code boost + reference expansion recover slightly.

#### Bad Case Distribution (No Rewrite)

Zero-recall queries are now auto-classified:

| Category | Count | Description |
|----------|-------|-------------|
| knowledge_gap | 9 | Relevant docs not in any index |
| ranking_failure | 4 | Docs found but ranked out of top-10 |
| reranker_regression | 4 | Reranker demoted correct results |

Run your own baseline:

```bash
# Full eval with per-stage metrics and bad case analysis
python -m rag eval --queries tests/eval/queries.jsonl

# Compare all rewrite modes
python -m rag eval --queries tests/eval/queries.jsonl --compare-rewrite

# Bad case analysis only
python -m rag eval --queries tests/eval/queries.jsonl --bad-cases-only
```

## LLM Query Rewrite Setup

The LLM-based query rewriter uses a local small model via Ollama to complete partial queries, decompose complex questions, and generate semantic variants. This is **optional** — when disabled or unavailable, the built-in rule-based engine is used instead.

### 1. Pull the Model

```bash
# Recommended: qwen2.5:3b (~1.9 GB)
ollama pull qwen2.5:3b

# Alternatives: llama3.2:3b, gemma3:4b, or any Ollama chat model
ollama pull llama3.2:3b
```

Verify the model is available:

```bash
ollama list | grep qwen2.5
```

### 2. Configure

Add to `config.yaml`:

```yaml
# ---- Query Rewrite (LLM) ----
query_rewrite_enabled: true
query_rewrite_max_variants: 3
query_rewrite_llm_model: "qwen2.5:3b"    # set to null to disable
query_rewrite_llm_timeout_ms: 5000        # increase if using CPU-only
```

- `query_rewrite_llm_model` — Ollama model name. Set to `null` or omit to use rule-based only.
- `query_rewrite_llm_timeout_ms` — Max wait for the LLM response. GPU users can lower to 2000ms; CPU users should set 5000–10000ms.

### 3. Test

```bash
# Single query test
python -c "
from rag.config import load_config
from rag.retriever.query_rewriter import LLMQueryRewriter
c = load_config()
rw = LLMQueryRewriter(c.ollama_host, 'qwen2.5:3b', 5000)
result = rw.rewrite('how to init renderer')
if result:
    print('completed:', result.completed)
    print('variants:', result.variants)
    print('sub_queries:', result.sub_queries)
"
```

### 4. Evaluate Impact

```bash
# Compare all three modes
python -m rag eval --queries tests/eval/queries.jsonl --compare-rewrite
```

**Performance note:** LLM rewrite adds ~2.5s per query on CPU. On GPU (Ollama with CUDA), this drops to ~300–500ms. The no-rewrite path is always fastest and currently achieves the best metrics for this dataset — use LLM rewrite selectively for hard natural-language queries where recall matters more than latency.

### How It Works

```
User query: "how to init renderer"
     │
     ▼
LLMQueryRewriter.rewrite()
     │
     ▼
Ollama /api/chat → qwen2.5:3b
     │
     ▼
{
  "completed": "How do I initialize the renderer?",
  "sub_queries": ["What are the steps to initialize a renderer?"],
  "variants": ["How do I start up the renderer?"]
}
     │
     ├─ completed → used for vector search (single best query)
     ├─ sub_queries → BM25 expansion (independent searches, merged in RRF)
     └─ variants → BM25 expansion (synonym coverage)
```

Any failure (model not found, timeout, invalid JSON) transparently falls back to the rule-based `expand()` engine. The search pipeline never breaks.

## Important Tips

### Performance

- **First index.** Embedding runs in a single global batch via Ollama's batch API — ~10k chunks takes 1-2 minutes for embedding, not tens of minutes. Subsequent incremental indexes are fast (seconds — unchanged files are skipped via mtime/size pre-check).
- **Embedding cache.** Incremental reindex skips Ollama for unchanged texts via `sha256(embed_text + model)` disk cache. Second reindex embed phase is near-instant (< 5s) when cache is hot. Cache lives at `embedding_cache_dir`.
- **BM25 disk persistence.** BM25 tokenized corpora are persisted to disk (`bm25_cache_dir`). After MCP server restart, first query loads from disk instead of pulling full ChromaDB data, reducing cold-start latency from 1-5s to < 0.1s.
- **Pipeline phases.** `python -m rag reindex` prints per-phase timing (crawl, parse, chunk, embed, chroma) so you can see exactly where time is spent. Embedding is typically < 30% of total time.
- **RRF weighting.** BM25 keyword matches are weighted 2× in RRF fusion (`rrf_bm25_weight: 2.0`), significantly improving Recall@1 for API/symbol name queries without hurting natural-language search quality.
- **Reranker optimization.** The reranker is automatically skipped when the RRF top1-top2 gap is large (> `reranker_score_gap_threshold`), and only the top 30 candidates (API types prioritized) are sent to the cross-encoder. This reduces per-query latency without hurting quality. Set `reranker_score_gap_threshold: 0` to never skip.
- **Ollama must be running.** Start it with `ollama serve` or ensure the Windows service is running.
- **Reranker download.** The first `search_docs` call will download the jina-reranker model (~1.1GB). This is one-time. Pre-download by running a test search after indexing. The reranker includes an automatic compatibility patch for transformers >= 4.46.
- **ChromaDB storage.** The vector database defaults to `./output/chroma_db` inside the project directory. It can grow to several GB for large doc sets — configure `chroma_dir` in config.yaml if you need it elsewhere.

### Document Sources

- **Auto module detection.** Sub-modules are auto-detected from directory structure (first path component). For example, `rendering/opengl/public/renderer.h` → module `rendering`.
- **Add sources at any time.** Use `add_doc_source` MCP tool or CLI. Run `reindex --source <label>` after adding.
- **Supported extensions.** `.html`, `.htm`, `.pdf`, `.h`, `.hpp`, `.hxx`. Other files are skipped.

### Search Quality

- **Use exact symbol names** when possible. `find_symbol` → `get_api_class` is the fast path.
- **Code boost triggers** — queries containing "how to", "example", "create", "implement", etc. get a +20% boost on code-containing chunks.
- **Reference expansion** — top results automatically pull in referenced symbols (1-hop, +5 max). This is most effective when your source docs have `see_also` sections.
- **BM25 weights** — symbol_name (×10) and signature (×5) are weighted higher than remarks (×1) and examples (×0.5) for API-focused searches. Adjust in config.yaml for narrative-heavy docs.
- **Query rewrite** — natural language queries can be rewritten for better recall. Two modes: rule-based (default, always available) and LLM-based (optional). See [LLM Query Rewrite Setup](#llm-query-rewrite-setup) for deployment instructions.
- **Evaluate quality** — measure retrieval quality with `python -m rag eval`. Requires annotated queries in `tests/eval/queries.jsonl`. Track Recall@K, MRR, and NDCG@K changes as you tune the pipeline.

### Troubleshooting

- **"0 chunks" after indexing.** Check that your document paths in config.yaml exist and contain supported file types. Run `python -m rag status` to verify.
- **Ollama connection errors.** Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- **Import errors.** Run `pip install -e .` from the project directory to ensure all dependencies are installed.
- **Symbol index is empty.** The symbol index is built after `reindex` completes. If indexing was interrupted, run `python -m rag reindex` again.

## Step-by-Step Verification (Tests)

The test suite is organized into numbered stages — run them in order to verify each layer of the system. Each stage builds on the previous one.

### Quick Run

```bash
# All tests (skip slow E2E)
pytest tests/ -q -k "not slow"

# Everything including slow E2E
pytest tests/ -q

# Run a single stage
pytest tests/test_09_search.py -v

# Gateway-specific stages
pytest tests/test_14_gateway_config.py tests/test_15_gateway_tools.py \
       tests/test_16_gateway_server.py tests/test_17_gateway_cli.py \
       tests/test_18_gateway_lifecycle.py -v
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
| 8 | `test_08_embedder.py` | Embedding dimension, batch count, offline error handling, embedding cache | Ollama + `nomic-embed-text` |
| 9 | `test_09_search.py` | Vector ANN, BM25 keyword, hybrid pipeline, symbol lookup, weighted RRF, BM25 disk persistence | Stage 8 + indexed ChromaDB data |
| 10 | `test_10_query_rewriter.py` | Query rewrite synonym expansion | Stage 8 |
| 11 | `test_11_e2e.py` | Full pipeline: index small doc set → search → verify | Stage 8 + document files |
| 12 | `test_12_llm_rewriter.py` | LLM rewriter JSON parsing, fallback on failure, symbol skip | None |
| 13 | `test_13_eval_trace.py` | PipelineTrace recall, bad case classification | None |
| 14 | `test_14_gateway_config.py` | Gateway config loading and optional CodeGraph defaults | None |
| 15 | `test_15_gateway_tools.py` | Gateway doc backend, CodeGraph client fakes, smart search routing | None |
| 16 | `test_16_gateway_server.py` | Gateway MCP stdio request handling and tool list assembly | None |
| 17 | `test_17_gateway_cli.py` | `rag gateway` CLI dispatch and existing CLI path preservation | None |
| 18 | `test_18_gateway_lifecycle.py` | CodeGraph lifecycle CLI command construction, status, init, reindex, sync, restart | None |
| 19 | `test_19_pytest_config.py` | Pytest basetemp/cache_dir project configuration | None |

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

## License

MIT — see [LICENSE](LICENSE) for details.
