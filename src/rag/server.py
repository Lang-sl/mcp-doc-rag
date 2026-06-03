"""MCP Server — exposes 12 tools via stdio JSON-RPC."""
from __future__ import annotations

import json
import sys
from typing import Any

from rag.config import load_config
from rag.models import SearchResult
from rag.source_manager import add_source, remove_source, list_sources
from rag.symbol_index import SymbolIndex
from rag.retriever.hybrid import HybridRetriever
from rag.context_builder import build_context
from rag.indexer.orchestrator import index_all, index_source


# Global state, initialized on startup
_config = None
_retriever = None
_symbol_index = None


def _init() -> None:
    global _config, _retriever, _symbol_index
    _config = load_config()
    _retriever = HybridRetriever(_config)
    _symbol_index = SymbolIndex(_config.symbol_index_path)


def _result_to_dict(r: SearchResult) -> dict:
    return {
        "chunk_id": r.chunk.chunk_id,
        "type": r.chunk.type,
        "symbol_id": r.chunk.symbol_id,
        "class_name": r.chunk.class_name,
        "function_name": r.chunk.function_name,
        "signature": r.chunk.signature,
        "source_label": r.chunk.source_label,
        "source_module": r.chunk.source_module,
        "source_file": r.chunk.source_file,
        "contains_code": r.chunk.contains_code,
        "remarks": (r.chunk.remarks or "")[:500],
        "score": round(r.score, 4),
    }


# === Tool Handlers ===

def handle_find_symbol(symbol: str) -> dict | None:
    """O(1) symbol lookup."""
    result = _symbol_index.lookup(symbol)
    return result


def handle_search_docs(
    query: str,
    top_k: int = 10,
    source_label: str | None = None,
    module: str | None = None,
) -> list[dict]:
    """Full hybrid search pipeline."""
    results = _retriever.search(query, top_k, source_label, module)
    return [_result_to_dict(r) for r in results]


def handle_get_api_class(class_name: str) -> dict | None:
    """Get full documentation for a class."""
    # Try symbol lookup first
    symbol = _symbol_index.lookup(class_name)
    if symbol and symbol.get("type") == "class":
        # Search for all chunks belonging to this class
        results = _retriever.search(class_name, top_k=20, source_label=symbol.get("source_label"))
        class_chunks = [
            r for r in results
            if r.chunk.class_name == class_name and r.chunk.type != "narrative"
        ]
        if class_chunks:
            return {
                "class_name": class_name,
                "source_label": symbol["source_label"],
                "source_file": symbol["file_path"],
                "members": [_result_to_dict(r) for r in class_chunks],
            }

    # Fallback: pure search
    results = _retriever.search(f"class {class_name}", top_k=15)
    if results:
        return {
            "class_name": class_name,
            "found_by": "semantic_search",
            "results": [_result_to_dict(r) for r in results[:10]],
        }
    return None


def handle_get_api_function(func_name: str, class_name: str | None = None) -> dict | None:
    """Get full documentation for a function."""
    symbol_id = f"{class_name}::{func_name}" if class_name else func_name

    # Try symbol lookup
    symbol = _symbol_index.lookup(symbol_id)
    if not symbol and class_name:
        # Try without class prefix
        symbol = _symbol_index.lookup(func_name)

    if symbol:
        results = _retriever.search(
            func_name,
            top_k=5,
            source_label=symbol.get("source_label"),
        )
        return {
            "function_name": func_name,
            "class_name": class_name,
            "source_label": symbol["source_label"],
            "source_file": symbol["file_path"],
            "results": [_result_to_dict(r) for r in results],
        }

    # Fallback
    query = f"{class_name}::{func_name}" if class_name else func_name
    results = _retriever.search(query, top_k=10)
    if results:
        return {
            "function_name": func_name,
            "class_name": class_name,
            "found_by": "semantic_search",
            "results": [_result_to_dict(r) for r in results],
        }
    return None


def handle_list_modules(source_label: str | None = None) -> list[str]:
    """List all module names."""
    import chromadb
    client = chromadb.PersistentClient(path=_config.chroma_dir)
    collections = [c.name for c in client.list_collections()]

    if source_label:
        prefix = f"{source_label}."
        collections = [c[len(prefix):] for c in collections if c.startswith(prefix)]
    else:
        # Extract unique module parts
        modules = set()
        for c in collections:
            parts = c.split(".", 1)
            if len(parts) > 1:
                modules.add(parts[1])
        collections = sorted(modules)

    return collections


def handle_build_context(
    query: str,
    top_k: int = 10,
    context_max_tokens: int = 6000,
    source_label: str | None = None,
) -> str:
    """Build a prompt-ready context block."""
    results = _retriever.search(query, top_k, source_label)
    return build_context(results, query, max_tokens=context_max_tokens)


def handle_add_doc_source(path: str, label: str) -> dict:
    """Register a new doc source."""
    return add_source(_config, label, path)


def handle_remove_doc_source(label: str) -> dict:
    """Remove a doc source."""
    # Remove from symbol index
    count = _symbol_index.remove_source(label)

    # Remove ChromaDB collections for this source
    import chromadb
    client = chromadb.PersistentClient(path=_config.chroma_dir)
    for coll in client.list_collections():
        if coll.name.startswith(f"{label}."):
            try:
                client.delete_collection(name=coll.name)
            except Exception:
                pass

    return remove_source(_config, label)


def handle_list_doc_sources() -> list[dict]:
    """List all registered sources."""
    return list_sources(_config)


def handle_reindex(source_label: str | None = None) -> dict:
    """Rebuild index."""
    if source_label:
        result = index_source(_config, source_label)
    else:
        result = index_all(_config)

    # Rebuild symbol index after indexing
    _build_symbol_index_from_db()

    return result


def handle_index_status() -> dict:
    """Return index statistics."""
    import chromadb
    client = chromadb.PersistentClient(path=_config.chroma_dir)

    collections = client.list_collections()
    total_chunks = 0
    per_source: dict[str, int] = {}

    for coll in collections:
        try:
            count = coll.count()
        except Exception:
            count = 0
        total_chunks += count

        source = coll.name.split(".")[0] if "." in coll.name else coll.name
        per_source[source] = per_source.get(source, 0) + count

    return {
        "total_chunks": total_chunks,
        "total_sources": len(_config.doc_sources),
        "total_collections": len(collections),
        "total_symbols": len(_symbol_index),
        "per_source": per_source,
    }


def _build_symbol_index_from_db() -> None:
    """Rebuild symbol index from ChromaDB collections."""
    import chromadb
    client = chromadb.PersistentClient(path=_config.chroma_dir)

    # Clear existing index
    _symbol_index._index.clear()

    for coll in client.list_collections():
        try:
            response = coll.get(include=["metadatas"])
        except Exception:
            continue

        for metadata in response.get("metadatas", []):
            symbol_id = metadata.get("symbol_id", "")
            if not symbol_id:
                continue

            if symbol_id in _symbol_index._index:
                continue

            _symbol_index._index[symbol_id] = {
                "type": metadata.get("type", ""),
                "symbol_id": symbol_id,
                "class_name": metadata.get("class_name") or None,
                "function_name": metadata.get("function_name") or None,
                "source_label": metadata.get("source_label", ""),
                "source_module": metadata.get("source_module", ""),
                "file_path": metadata.get("source_file", ""),
            }

    _symbol_index.flush()


# === Tool Registry (MCP format) ===

TOOLS = [
    {"name": "find_symbol", "description": "Exact symbol lookup by symbol_id (O(1) hash). First step for any API query.", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "search_docs", "description": "Full hybrid search: BM25 + embedding -> RRF -> reranker -> boost -> expand.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}, "source_label": {"type": "string"}, "module": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_api_class", "description": "Get full documentation for a specific class.", "inputSchema": {"type": "object", "properties": {"class_name": {"type": "string"}}, "required": ["class_name"]}},
    {"name": "get_api_function", "description": "Get full documentation for a specific function.", "inputSchema": {"type": "object", "properties": {"func_name": {"type": "string"}, "class_name": {"type": "string"}}, "required": ["func_name"]}},
    {"name": "list_modules", "description": "List all known sub-module names.", "inputSchema": {"type": "object", "properties": {"source_label": {"type": "string"}}}},
    {"name": "build_context", "description": "Build a token-bounded, formatted context block for prompt injection.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}, "context_max_tokens": {"type": "integer", "default": 6000}, "source_label": {"type": "string"}}, "required": ["query"]}},
    {"name": "add_doc_source", "description": "Register a new document source directory.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "label": {"type": "string"}}, "required": ["path", "label"]}},
    {"name": "remove_doc_source", "description": "Remove a registered document source.", "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}},
    {"name": "list_doc_sources", "description": "List all registered document sources.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "reindex", "description": "Rebuild the search index (incremental by default).", "inputSchema": {"type": "object", "properties": {"source_label": {"type": "string"}}}},
    {"name": "index_status", "description": "Return index statistics and breakdown.", "inputSchema": {"type": "object", "properties": {}}},
]

TOOL_HANDLERS = {
    "find_symbol": lambda args: handle_find_symbol(args["symbol"]),
    "search_docs": lambda args: handle_search_docs(args["query"], args.get("top_k", 10), args.get("source_label"), args.get("module")),
    "get_api_class": lambda args: handle_get_api_class(args["class_name"]),
    "get_api_function": lambda args: handle_get_api_function(args["func_name"], args.get("class_name")),
    "list_modules": lambda args: handle_list_modules(args.get("source_label")),
    "build_context": lambda args: handle_build_context(args["query"], args.get("top_k", 10), args.get("context_max_tokens", 6000), args.get("source_label")),
    "add_doc_source": lambda args: handle_add_doc_source(args["path"], args["label"]),
    "remove_doc_source": lambda args: handle_remove_doc_source(args["label"]),
    "list_doc_sources": lambda args: handle_list_doc_sources(),
    "reindex": lambda args: handle_reindex(args.get("source_label")),
    "index_status": lambda args: handle_index_status(),
}


def main() -> None:
    """MCP Server main loop — stdio JSON-RPC."""
    _init()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        method = request.get("method")

        if method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            handler = TOOL_HANDLERS.get(tool_name)
            if handler:
                try:
                    result = handler(arguments)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]},
                    }
                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": str(e)},
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                }

        elif method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "mcp-doc-rag", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            }

        elif method == "initialized":
            # No response needed for notifications
            continue

        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
