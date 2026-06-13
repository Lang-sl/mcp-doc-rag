from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

import yaml

from rag.config import _YAML_SKIP_FIELDS, load_config
from rag.context_builder import build_context
from rag.indexer.orchestrator import index_all, index_source
from rag.models import SearchResult
from rag.retriever.hybrid import HybridRetriever
from rag.source_manager import add_source, list_sources
from rag.symbol_index import SymbolIndex


class DocRagBackend:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.retriever = HybridRetriever(self.config)
        self.symbol_index = SymbolIndex(self.config.symbol_index_path)

    def find_symbol(self, symbol: str) -> dict | None:
        result = self.symbol_index.lookup(symbol)
        if result is None:
            return None
        source_label = result.get("source_label")
        if source_label and hasattr(self, "config") and source_label not in self.config.doc_sources:
            return None
        return result

    def search_docs(
        self,
        query: str,
        top_k: int = 10,
        source_label: str | None = None,
        module: str | None = None,
    ) -> list[dict]:
        results = self._active_results(self.retriever.search(query, top_k, source_label, module))
        return [self._result_to_dict(result) for result in results]

    def get_api_class(self, class_name: str) -> dict | None:
        symbol = self.find_symbol(class_name)
        if symbol and symbol.get("type") == "class":
            results = self.retriever.search(
                class_name,
                top_k=20,
                source_label=symbol.get("source_label"),
                skip_rerank=True,
            )
            class_chunks = [
                result for result in self._active_results(results)
                if result.chunk.class_name == class_name and result.chunk.type != "narrative"
            ]
            if class_chunks:
                return {
                    "class_name": class_name,
                    "source_label": symbol["source_label"],
                    "source_file": symbol["file_path"],
                    "members": [self._result_to_dict(result) for result in class_chunks],
                }

        results = self._active_results(self.retriever.search(f"class {class_name}", top_k=15, skip_rerank=True))
        if results:
            return {
                "class_name": class_name,
                "found_by": "semantic_search",
                "results": [self._result_to_dict(result) for result in results[:10]],
            }
        return None

    def get_api_function(self, func_name: str, class_name: str | None = None) -> dict | None:
        symbol_id = f"{class_name}::{func_name}" if class_name else func_name

        symbol = self.find_symbol(symbol_id)
        if not symbol and class_name:
            symbol = self.find_symbol(func_name)

        if symbol:
            results = self._active_results(self.retriever.search(
                func_name,
                top_k=5,
                source_label=symbol.get("source_label"),
                skip_rerank=True,
            ))
            return {
                "function_name": func_name,
                "class_name": class_name,
                "source_label": symbol["source_label"],
                "source_file": symbol["file_path"],
                "results": [self._result_to_dict(result) for result in results],
            }

        query = f"{class_name}::{func_name}" if class_name else func_name
        results = self._active_results(self.retriever.search(query, top_k=10, skip_rerank=True))
        if results:
            return {
                "function_name": func_name,
                "class_name": class_name,
                "found_by": "semantic_search",
                "results": [self._result_to_dict(result) for result in results],
            }
        return None

    def list_modules(self, source_label: str | None = None) -> list[str]:
        import chromadb

        client = chromadb.PersistentClient(path=self.config.chroma_dir)
        collections = [collection.name for collection in client.list_collections()]

        if source_label:
            prefix = f"{source_label}."
            return [name[len(prefix):] for name in collections if name.startswith(prefix)]

        modules = set()
        for name in collections:
            parts = name.split(".", 1)
            if len(parts) > 1:
                modules.add(parts[1])
        return sorted(modules)

    def build_context(
        self,
        query: str,
        top_k: int = 10,
        context_max_tokens: int = 6000,
        source_label: str | None = None,
    ) -> str:
        results = self._active_results(self.retriever.search(query, top_k, source_label))
        return build_context(results, query, max_tokens=context_max_tokens)

    def add_doc_source(self, path: str, label: str) -> dict:
        with self._config_env():
            return add_source(self.config, label, path)

    def remove_doc_source(self, label: str) -> dict:
        if label not in self.config.doc_sources:
            return {"ok": False, "error": f"Source with label '{label}' not found."}

        source_path = self.config.doc_sources[label]
        self.config.doc_sources.pop(label)
        try:
            self._save_config()
        except Exception as exc:
            self.config.doc_sources[label] = source_path
            return {"ok": False, "error": str(exc)}

        import chromadb

        client = chromadb.PersistentClient(path=self.config.chroma_dir)
        collections_to_delete = [
            collection.name for collection in client.list_collections()
            if collection.name.startswith(f"{label}.")
        ]
        failed_collections: list[str] = []
        for collection_name in collections_to_delete:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                failed_collections.append(collection_name)

        remaining_collections = [
            collection.name for collection in client.list_collections()
            if collection.name.startswith(f"{label}.")
        ]

        symbol_error = None
        self.symbol_index.remove_source(label)
        try:
            self.symbol_index.flush()
        except Exception as exc:
            symbol_error = str(exc)
        self.retriever.invalidate_cache()

        cleanup_errors = failed_collections + [
            name for name in remaining_collections if name not in failed_collections
        ]
        if cleanup_errors or symbol_error:
            result = {
                "ok": False,
                "label": label,
                "path": source_path,
                "source_removed": True,
                "error": "Source removed, but cleanup was incomplete.",
            }
            if cleanup_errors:
                result["collections"] = cleanup_errors
            if symbol_error:
                result["symbol_index_error"] = symbol_error
            return result
        return {"ok": True, "label": label, "path": source_path}

    def list_doc_sources(self) -> list[dict]:
        return list_sources(self.config)

    def reindex(self, source_label: str | None = None) -> dict:
        if source_label:
            result = index_source(self.config, source_label)
        else:
            result = index_all(self.config)

        self._build_symbol_index_from_db()
        self.retriever.invalidate_cache()

        return result

    def index_status(self) -> dict:
        import chromadb

        client = chromadb.PersistentClient(path=self.config.chroma_dir)
        collections = client.list_collections()
        total_chunks = 0
        per_source: dict[str, int] = {}

        for collection in collections:
            try:
                count = collection.count()
            except Exception:
                count = 0
            total_chunks += count

            source = collection.name.split(".")[0] if "." in collection.name else collection.name
            per_source[source] = per_source.get(source, 0) + count

        return {
            "total_chunks": total_chunks,
            "total_sources": len(self.config.doc_sources),
            "total_collections": len(collections),
            "total_symbols": len(self.symbol_index),
            "per_source": per_source,
        }

    def health(self) -> dict:
        return {
            "ok": True,
            "config_path": self.config_path,
            "source_count": len(self.config.doc_sources),
        }

    @staticmethod
    def _result_to_dict(result: SearchResult) -> dict:
        chunk = result.chunk
        return {
            "chunk_id": chunk.chunk_id,
            "type": chunk.type,
            "symbol_id": chunk.symbol_id,
            "class_name": chunk.class_name,
            "function_name": chunk.function_name,
            "signature": chunk.signature,
            "source_label": chunk.source_label,
            "source_module": chunk.source_module,
            "source_file": chunk.source_file,
            "contains_code": chunk.contains_code,
            "remarks": (chunk.remarks or "")[:500],
            "score": round(result.score, 4),
        }

    def _build_symbol_index_from_db(self) -> None:
        from rag.indexer.orchestrator import _rebuild_symbol_index

        _rebuild_symbol_index(self.config)
        self.symbol_index._load()

    def _save_config(self) -> None:
        path = self._config_save_path()
        data = asdict(self.config)
        for skip_field in _YAML_SKIP_FIELDS:
            data.pop(skip_field, None)

        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        temp_path = f"{abs_path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
            os.replace(temp_path, abs_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _config_save_path(self) -> str:
        if self.config_path is not None:
            return self.config_path
        return os.environ.get("RAG_CONFIG_PATH", "./config.yaml")

    def _active_results(self, results: list[SearchResult]) -> list[SearchResult]:
        return [
            result for result in results
            if self._is_active_source(result.chunk.source_label)
        ]

    def _is_active_source(self, source_label: str) -> bool:
        if not hasattr(self, "config"):
            return True
        return source_label in self.config.doc_sources

    @contextmanager
    def _config_env(self) -> Iterator[None]:
        if self.config_path is None:
            yield
            return

        previous = os.environ.get("RAG_CONFIG_PATH")
        os.environ["RAG_CONFIG_PATH"] = self.config_path
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("RAG_CONFIG_PATH", None)
            else:
                os.environ["RAG_CONFIG_PATH"] = previous
