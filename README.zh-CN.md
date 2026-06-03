# mcp-doc-rag

[![CI](https://github.com/Lang-sl/mcp-doc-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Lang-sl/mcp-doc-rag/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-native-purple)](https://modelcontextprotocol.io/)
[![English Docs](https://img.shields.io/badge/docs-English-blue)](README.md)

**完全本地化、MCP 集成的 C/C++ SDK 文档 RAG 检索系统。**

一套检索增强生成（RAG）引擎，可索引 C++ SDK 文档（Doxygen HTML、PDF、C++ 头文件），并通过 MCP（Model Context Protocol）服务器暴露混合搜索能力——让 Claude Code 等 AI 编程助手能够按需检索精确的 API 文档。

## 为什么选择 mcp-doc-rag

- **100% 本地** — 无云端 API 调用。嵌入用 Ollama，向量数据库用 ChromaDB，重排序器来自 HuggingFace。所有数据留存在你的机器上。
- **MCP 原生** — 优先作为 MCP 服务器设计。Claude Code（及其他 MCP 客户端）可在编码时自动调用 RAG 工具。
- **混合搜索** — 结合字段加权 BM25（符号×10、签名×5）+ 向量 ANN → RRF 融合 → 条件性 jina-reranker 跨编码器 → 代码加权 → 引用扩展。对于符号/API 标识符查询（如 `MwMultiAxis::CalculateToolpath`）自动跳过 reranker，保持低延迟。
- **结构化分块** — Doxygen 感知的 HTML 解析器提取 symbol_id、类、函数、签名、参数、返回类型、备注和代码示例到结构化 JSON chunk 中。
- **O(1) 符号查找** — 通过内存哈希索引精确定位符号，对已知 API 名称绕过全量搜索。
- **增量索引** — SHA1 内容哈希 + mtime/size 预过滤。仅重新索引变更文件。
- **可定制** — 通过 MCP 工具或 CLI 在运行时添加/移除文档源。

## 技术栈

| 组件 | 技术 |
|------|------|
| 嵌入 | Ollama `nomic-embed-text`（768 维，~275MB） |
| 向量数据库 | ChromaDB（按 source.module 分 collection） |
| 重排序器 | `jinaai/jina-reranker-v2-base-multilingual`（HuggingFace） |
| 关键词搜索 | `rank-bm25`（字段加权多索引） |
| 跨编码器运行时 | `transformers` + `torch` + `einops` |
| PDF 提取 | `pdfplumber` |
| HTML 解析 | `BeautifulSoup4`（Doxygen 结构感知） |
| 集成 | MCP Server（stdio JSON-RPC） |
| 配置 | YAML |

## 前置条件

- **Python** >= 3.11
- **Ollama**（用于嵌入）

### 安装 Ollama

```bash
# Windows
winget install Ollama.Ollama

# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

设置模型存储路径（可选，默认使用 Ollama 的默认位置）：

```bash
# Windows
setx OLLAMA_MODELS "C:\path\to\models"

# macOS / Linux
export OLLAMA_MODELS=/path/to/models
```

拉取嵌入模型：

```bash
ollama pull nomic-embed-text
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/Lang-sl/mcp-doc-rag.git
cd mcp-doc-rag

# 开发模式安装
pip install -e .
```

**注意：** 首次使用重排序器时，会自动从 HuggingFace 下载 jina-reranker 模型（~1.1GB）。这是一次性下载。如果重排序器不可用（例如 transformers 版本不兼容），搜索会平滑降级——直接使用 RRF 融合分数。

## 快速开始

### 1. 运行配置向导

```bash
python setup_config.py
```

该交互式脚本将：
- 从模板创建 `config.yaml`
- 帮助你添加文档源路径
- 验证 Ollama 是否在运行

也可以手动复制并编辑模板：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml：设置 doc_sources 路径
```

### 2. 索引文档

```bash
# 全量索引（增量模式——跳过未变更文件）
python -m rag reindex

# 索引单个源
python -m rag reindex --source my_sdk

# 强制完全重建（忽略缓存的哈希）
python -m rag reindex --full
```

### 3. 搜索

```bash
# 混合搜索
python -m rag query "How to initialize the rendering kernel"

# 按源过滤
python -m rag query --source my_sdk "Initialize renderer"

# 精确符号查找
python -m rag symbol MySDK::Renderer::Initialize

# 构建上下文块（可直接作为提示词）
python -m rag context "5-axis simulation setup"
```

### 4. 管理文档源

```bash
python -m rag source add my_new_docs D:/path/to/docs
python -m rag source list
python -m rag source remove my_new_docs
```

### 5. 查看状态

```bash
python -m rag status
# 输出：total_chunks, 按源统计, collections 数, 符号数量
```

## 配置参考

完整的 `config.yaml` 及默认值（另见 `config.example.yaml`）：

```yaml
# ---- 路径（相对于项目根目录） ----
chroma_dir: ./chroma_db
symbol_index_path: ./symbol_index.json

# ---- 文档源（标签: 路径） ----
doc_sources:
  my_sdk: /absolute/path/to/your/docs

# ---- Ollama ----
ollama_host: http://localhost:11434
embed_model: nomic-embed-text
embed_dim: 768
embed_batch_size: 32

# ---- 重排序器 ----
reranker_model: jinaai/jina-reranker-v2-base-multilingual
reranker_max_length: 512

# ---- 分块 ----
chunk_max_chars: 2000

# ---- 检索 ----
top_k_default: 10
candidate_expand_factor: 4
rrf_k: 30

# ---- BM25 字段权重 ----
bm25_weights:
  symbol_name: 10.0
  signature: 5.0
  remarks: 1.0
  example: 0.5

# ---- 代码加权 ----
code_boost_ratio: 1.20
code_boost_triggers:
  - "how to"
  - "create"
  - "example"
  - "sample"
  - "write"
  - "implement"
  - "setup"

# ---- 引用扩展 ----
ref_expansion_max: 5

# ---- 上下文构建 ----
context_max_tokens: 6000

# ---- 缓存 ----
cache_max_entries: 128

# ---- 索引 ----
index_batch_size: 100
```

## MCP 服务器集成

mcp-doc-rag 是一个 MCP 服务器——AI 编程助手可以直接调用其工具。

### Claude Code 配置

在**项目根目录**（运行 `claude` 的目录）创建 `.mcp.json`。这是推荐的配置方式——`.mcp.json` 是专用的 MCP 配置文件，`settings.local.json` 已不再支持 `mcpServers` 字段。

```json
{
  "mcpServers": {
    "mcp-doc-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "rag.server"],
      "cwd": "D:/rag/mcp-doc-rag",
      "env": {
        "RAG_CONFIG_PATH": "D:/rag/mcp-doc-rag/config.yaml"
      }
    }
  }
}
```

将路径替换为你的仓库绝对路径。即使在 Windows 上也请使用正斜杠。

**其他作用域**（根据需求选择）：

| 作用域 | 文件位置 | 适用场景 |
|--------|---------|----------|
| **project**（推荐） | 项目根目录 `.mcp.json` | 团队共享，可提交到版本控制 |
| **user** | `~/.claude/mcp.json` | 个人工具，在所有项目中可用 |
| **local** | `~/.config/claude/mcp.json` | 本机专属配置、本地凭据 |

配置完成后，重启 Claude Code。使用 `/mcp` 验证服务器是否已加载。

### 可用 MCP 工具

| 工具 | 说明 |
|------|------|
| `find_symbol` | 按 symbol_id 精确定位（O(1)） |
| `search_docs` | 完整混合搜索流程 |
| `get_api_class` | 获取完整的类文档及其成员 |
| `get_api_function` | 获取完整的函数文档 |
| `list_modules` | 列出所有已知的子模块名称 |
| `build_context` | 构建受 token 限制的提示词上下文块 |
| `add_doc_source` | 注册新的文档源 |
| `remove_doc_source` | 移除文档源及其 chunk |
| `list_doc_sources` | 列出所有已注册的文档源 |
| `reindex` | 增量重建索引（指定源或全部） |
| `index_status` | 汇总索引统计信息 |

### Claude Code 使用示例

```
用户："How do I use MySDK::Renderer::Initialize?"

Claude 内部调用：
  find_symbol("MySDK::Renderer::Initialize")
  get_api_function("Initialize", class_name="MySDK::Renderer")
  → 返回签名、参数、备注、示例代码

用户得到来自实际 SDK 文档的精确答案及代码示例。
```

## 文档格式支持

| 格式 | 检测方式 | 提取内容 |
|------|----------|----------|
| Doxygen HTML（现代） | `<meta generator="Doxygen">`、`.memitem` | 函数、类、枚举、宏、参数、返回类型 |
| Doxygen HTML（旧版） | `<TITLE>Function:...</TITLE>` | 函数、签名、参数、描述 |
| CHM 格式 HTML | `<meta name="generator" content="...">` | 函数、类、枚举、参数 |
| PDF | `.pdf` 扩展名 | 页面段落 |
| C++ 头文件 | `.h`、`.hpp`、`.hxx` | 函数/类/枚举签名及注释 |

### 添加自定义解析器

解析器采用装饰器注册机制——无需改动 orchestrator 或 crawler 即可添加新格式：

```python
# parser_markdown.py
from rag.indexer.parser_registry import register_parser

@register_parser(file_type="markdown", extensions=[".md", ".mdx"])
def parse_markdown(file_path, source_label, source_module):
    # 解析 .md 文件 → 返回 list[dict]
    ...
```

然后在 `src/rag/indexer/__init__.py` 中导入即可触发注册。crawler 会自动发现 `.md` 文件，orchestrator 会自动选用正确的解析器。

## 检索流程

```
Query
  │
  ├─ [CACHE] LRU 检查（128 条目）
  │
  ├─ BM25（字段加权，按 collection）
  ├─ 向量 ANN（按 collection ChromaDB）
  │
  └─ RRF 融合（k=30） → 80 个候选
      │
      ├─ 重排序器（jina-reranker-v2 跨编码器，平滑降级）
      ├─ 代码加权（触发词 +20%）
      ├─ 引用扩展（1 跳，最多 +5）
      │
      └─ Top-K 结果
```

## 重要提示

### 性能

- **首次索引。** 嵌入通过 Ollama 批量 API 一次性全局批量执行——~10k chunk 的嵌入只需 1-2 分钟，而非几十分钟。后续增量索引很快（秒级——不变更的文件通过 mtime/size 预检查跳过）。
- **流水线阶段。** `python -m rag reindex` 会打印每个阶段的耗时（crawl, parse, chunk, embed, chroma），便于精确定位时间消耗。嵌入通常占总时间不到 30%。
- **Ollama 必须在运行。** 使用 `ollama serve` 启动或确保 Windows 服务正在运行。
- **重排序器下载。** 首次 `search_docs` 调用会下载 jina-reranker 模型（~1.1GB），仅此一次。索引后运行一次测试搜索可提前下载。重排序器包含 transformers >= 4.46 的自动兼容性补丁。
- **ChromaDB 存储。** 向量数据库默认存放在项目目录下的 `./chroma_db`。对于大型文档集可能增长到数 GB——如需其他位置，请在 config.yaml 中配置 `chroma_dir`。

### 文档源

- **自动模块检测。** 子模块通过目录结构自动检测（以路径首层目录为模块名）。例如 `rendering/opengl/public/renderer.h` → 模块 `rendering`。
- **随时添加源。** 使用 `add_doc_source` MCP 工具或 CLI。添加后运行 `reindex --source <label>`。
- **支持的扩展名。** `.html`、`.htm`、`.pdf`、`.h`、`.hpp`、`.hxx`。其他文件会被跳过。

### 搜索质量

- **尽可能使用精确符号名。** `find_symbol` → `get_api_class` 是最快路径。
- **代码加权触发词**——包含 "how to"、"example"、"create"、"implement" 等词的查询会对含代码的 chunk 增加 +20% 权重。
- **引用扩展**——Top 结果自动拉取引用的符号（1 跳，最多 +5）。当文档源包含 `see_also` 章节时效果最佳。
- **BM25 权重**——面向 API 搜索时，symbol_name（×10）和 signature（×5）的权重高于 remarks（×1）和 example（×0.5）。对于以叙述为主的文档，可在 config.yaml 中调整。

### 故障排除

- **索引后显示"0 chunks"。** 检查 config.yaml 中的文档路径是否存在且包含支持的文件类型。运行 `python -m rag status` 验证。
- **Ollama 连接错误。** 确保 Ollama 正在运行：`curl http://localhost:11434/api/tags`
- **导入错误。** 在项目目录中运行 `pip install -e .` 确保所有依赖已安装。
- **符号索引为空。** 符号索引在 `reindex` 完成后构建。如果索引过程中断，重新运行 `python -m rag reindex`。

## 分步验证（测试）

测试套件按 10 个编号阶段组织——按顺序运行以逐层验证系统的每个部分。每个阶段都建立在前一个阶段之上。

### 快速运行

```bash
# 阶段 1-7：纯单元 + 文件爬虫测试（无需 Ollama）
pytest tests/test_01_config.py tests/test_02_source_manager.py \
       tests/test_03_symbol_index.py tests/test_04_parser.py \
       tests/test_05_chunker.py tests/test_06_context_builder.py \
       tests/test_07_crawler.py -v

# 阶段 8：嵌入（需要 Ollama 运行）
pytest tests/test_08_embedder.py -v

# 阶段 9：搜索流水线（需要 Ollama + 已索引数据）
pytest tests/test_09_search.py -v

# 阶段 10：完整端到端（较慢——需要所有环境）
pytest tests/test_10_e2e.py -v -m slow

# 运行除慢速端到端测试外的全部测试
pytest tests/ -v -k "not slow"
```

### 阶段参考

| 阶段 | 文件 | 验证内容 | 前置条件 |
|------|------|----------|----------|
| 1 | `test_01_config.py` | 默认值、YAML 加载、环境变量覆盖、BM25 权重 | 无 |
| 2 | `test_02_source_manager.py` | 文档源 CRUD：增、删、列、重复检测 | 无 |
| 3 | `test_03_symbol_index.py` | O(1) 哈希映射查找、源范围删除 | 无 |
| 4 | `test_04_parser.py` | 类型名提取、叙述型 HTML、Doxygen 函数解析 | 无 |
| 5 | `test_05_chunker.py` | Chunk 组装、BM25 字段、嵌入文本、丢弃规则 | 无 |
| 6 | `test_06_context_builder.py` | 上下文格式化、token 上限、分数排序 | 无 |
| 7 | `test_07_crawler.py` | 文件发现、SHA1 增量检查、二次跳过 | 已配置路径的真实文档文件 |
| 8 | `test_08_embedder.py` | 嵌入维度、批次数、离线错误处理 | Ollama + `nomic-embed-text` |
| 9 | `test_09_search.py` | 向量 ANN、BM25 关键词、混合流水线、符号查找 | 阶段 8 + 已索引 ChromaDB 数据 |
| 10 | `test_10_e2e.py` | 完整流水线：索引小文档集 → 搜索 → 验证 | 阶段 8 + 文档文件 |

**阶段 1-6** 即时运行（无网络，除临时文件外无磁盘 I/O）。如果有任何失败，说明存在代码或依赖问题。

**阶段 7** 验证文件爬虫对真实文档的表现。设置 `RAG_TEST_DOC_DIR` 指向包含 Doxygen HTML 或 PDF 文件的目录。未设置则自动跳过。

**阶段 8** 是第一个依赖 Ollama 的测试。如果测试跳过并提示 "Ollama not running"，用 `ollama serve` 启动并确保 `nomic-embed-text` 已拉取。

**阶段 9** 需要已索引的数据。运行这些测试前先执行 `python -m rag reindex`。

**阶段 10** 标记为 `@pytest.mark.slow`——它会索引一小部分文档并运行完整搜索流程。运行前设置 `RAG_TEST_DOC_DIR`。适合作为发布前的冒烟测试。

### 结果解读

```
53 passed, 1 deselected  ← ✅ 所有系统正常运作
40 passed, 13 skipped    ← ⚠️ 阶段 8+ 被跳过。检查 Ollama 和索引。
3 failed, 50 passed      ← ❌ 失败表明特定组件有问题。
                            逐阶段运行以隔离问题。
```

## 项目结构

```
mcp-doc-rag/
├── pyproject.toml
├── setup_config.py            # 交互式配置向导
├── config.example.yaml        # 配置模板
├── .gitignore
├── LICENSE
├── README.md
├── tests/
│   ├── conftest.py              # 共享 fixture、Ollama 检测
│   ├── test_01_config.py        # 阶段 1：配置加载
│   ├── test_02_source_manager.py # 阶段 2：文档源 CRUD
│   ├── test_03_symbol_index.py  # 阶段 3：符号索引
│   ├── test_04_parser.py        # 阶段 4：HTML 解析器
│   ├── test_05_chunker.py       # 阶段 5：Chunk 组装
│   ├── test_06_context_builder.py # 阶段 6：上下文构建器
│   ├── test_07_crawler.py       # 阶段 7：文件爬虫
│   ├── test_08_embedder.py      # 阶段 8：嵌入
│   ├── test_09_search.py        # 阶段 9：搜索流水线
│   └── test_10_e2e.py           # 阶段 10：完整端到端（慢速）
└── src/rag/
    ├── server.py              # MCP 服务器（11 个工具，stdio JSON-RPC）
    ├── cli.py                 # CLI 入口
    ├── config.py              # YAML 配置加载器
    ├── models.py              # Chunk、SearchResult、IndexStats 数据类
    ├── symbol_index.py        # O(1) 符号哈希映射
    ├── source_manager.py      # 文档源 CRUD
    ├── context_builder.py     # token 受限的上下文格式化器
    ├── indexer/
    │   ├── crawler.py           # 文件遍历器带 SHA1 增量检查
    │   ├── parser_registry.py   # 基于装饰器的解析器注册
    │   ├── parser_html.py       # Doxygen HTML 解析器（4 种格式）
    │   ├── parser_pdf.py        # PDF 文本提取器
    │   ├── parser_header.py     # C++ 头文件签名提取器
    │   ├── chunker.py           # 结构化 chunk 组装器
    │   ├── embedder.py          # Ollama 批量嵌入封装
    │   └── orchestrator.py      # 完整索引流水线
    └── retriever/
        ├── vector_search.py   # ChromaDB ANN 按 collection
        ├── bm25_search.py     # 字段加权 BM25
        ├── hybrid.py          # 完整流水线编排
        └── reranker.py        # jina-reranker-v2 跨编码器
```

## 开源协议

MIT — 详见 [LICENSE](LICENSE)。
