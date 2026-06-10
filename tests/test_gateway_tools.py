from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.config import Config
from rag.gateway.doc_backend import DocRagBackend
from rag.symbol_index import SymbolIndex


@dataclass
class FakeChunk:
    chunk_id: str = "chunk-1"
    type: str = "function"
    symbol_id: str = "Sdk::Render"
    class_name: str | None = "Sdk"
    function_name: str | None = "Render"
    signature: str | None = "void Render()"
    source_label: str = "sdk"
    source_module: str = "render"
    source_file: str = "render.h"
    contains_code: bool = True
    remarks: str | None = "Render the mesh."


@dataclass
class FakeSearchResult:
    chunk: FakeChunk = field(default_factory=FakeChunk)
    score: float = 1.23456


class FakeSymbolIndex:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def __len__(self) -> int:
        return 1

    def lookup(self, symbol: str) -> dict | None:
        if symbol == "Sdk::Render":
            return {"symbol_id": "Sdk::Render", "type": "function", "source_label": "sdk", "file_path": "render.h"}
        if symbol == "Sdk":
            return {"symbol_id": "Sdk", "type": "class", "source_label": "sdk", "file_path": "sdk.h"}
        return None

    def remove_source(self, label: str) -> int:
        self.removed.append(label)
        return 1

    def flush(self) -> None:
        pass


class FakeRetriever:
    def __init__(self) -> None:
        self.invalidated = False

    def search(self, query: str, top_k: int = 10, source_label=None, module=None, skip_rerank: bool = False):
        return [FakeSearchResult()]

    def invalidate_cache(self) -> None:
        self.invalidated = True


def test_doc_backend_find_symbol_uses_symbol_index():
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.symbol_index = FakeSymbolIndex()

    assert backend.find_symbol("Sdk::Render")["symbol_id"] == "Sdk::Render"
    assert backend.find_symbol("Missing") is None


def test_doc_backend_search_docs_serializes_search_results():
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.retriever = FakeRetriever()

    results = backend.search_docs("render mesh", top_k=1)

    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["symbol_id"] == "Sdk::Render"
    assert results[0]["score"] == 1.2346


def test_doc_backend_result_to_dict_truncates_remarks():
    long_remarks = "x" * 600
    result = FakeSearchResult(FakeChunk(remarks=long_remarks))

    serialized = DocRagBackend._result_to_dict(result)

    assert serialized["remarks"] == "x" * 500


def test_doc_backend_list_doc_sources_uses_config():
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config = Config(doc_sources={"sdk": "sdk/docs"})

    assert backend.list_doc_sources() == [{"label": "sdk", "path": "sdk/docs"}]


def test_doc_backend_add_doc_source_saves_explicit_config_path(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "backend.yaml"
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = str(config_path)
    backend.config = Config(doc_sources={})
    monkeypatch.setenv("RAG_CONFIG_PATH", str(tmp_path / "wrong.yaml"))

    result = backend.add_doc_source(str(source_dir), "sdk")

    assert result == {"ok": True, "label": "sdk", "path": os.path.normpath(str(source_dir))}
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["doc_sources"]["sdk"] == os.path.normpath(str(source_dir))
    assert not (tmp_path / "wrong.yaml").exists()
    assert os.environ["RAG_CONFIG_PATH"] == str(tmp_path / "wrong.yaml")


def test_doc_backend_remove_doc_source_saves_explicit_config_path(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "backend.yaml"
    wrong_path = tmp_path / "wrong.yaml"
    symbol_index_path = tmp_path / "symbols.json"
    symbol_index = SymbolIndex(str(symbol_index_path))
    symbol_index._index = {
        "Sdk::Render": {"symbol_id": "Sdk::Render", "source_label": "sdk"},
        "Other::Call": {"symbol_id": "Other::Call", "source_label": "other"},
    }
    symbol_index.flush()
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = str(config_path)
    backend.config = Config(
        chroma_dir=str(tmp_path / "chroma"),
        symbol_index_path=str(symbol_index_path),
        doc_sources={"sdk": "sdk/docs"},
    )
    backend.symbol_index = symbol_index
    backend.retriever = FakeRetriever()
    fake_chromadb = types.SimpleNamespace(
        PersistentClient=lambda path: types.SimpleNamespace(list_collections=lambda: [])
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setenv("RAG_CONFIG_PATH", str(wrong_path))

    result = backend.remove_doc_source("sdk")

    assert result == {"ok": True, "label": "sdk", "path": "sdk/docs"}
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["doc_sources"] == {}
    assert not wrong_path.exists()
    assert os.environ["RAG_CONFIG_PATH"] == str(wrong_path)
    reloaded = SymbolIndex(str(symbol_index_path))
    assert reloaded.lookup("Sdk::Render") is None
    assert reloaded.lookup("Other::Call")["source_label"] == "other"
    assert backend.retriever.invalidated is True


def test_doc_backend_remove_missing_source_has_no_side_effects(tmp_path: Path, monkeypatch):
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = None
    backend.config = Config(chroma_dir=str(tmp_path / "chroma"), doc_sources={})
    backend.symbol_index = FakeSymbolIndex()
    backend.retriever = FakeRetriever()
    fake_chromadb = types.SimpleNamespace(
        PersistentClient=lambda path: (_ for _ in ()).throw(AssertionError("Chroma should not be touched"))
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    result = backend.remove_doc_source("missing")

    assert result == {"ok": False, "error": "Source with label 'missing' not found."}
    assert backend.symbol_index.removed == []
    assert backend.retriever.invalidated is False


def test_doc_backend_remove_source_reports_chroma_delete_failure_without_config_mutation(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "backend.yaml"
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = str(config_path)
    backend.config = Config(chroma_dir=str(tmp_path / "chroma"), doc_sources={"sdk": "sdk/docs"})
    backend.symbol_index = FakeSymbolIndex()
    backend.retriever = FakeRetriever()

    class FakeCollection:
        name = "sdk.render"

    class FailingClient:
        def list_collections(self):
            return [FakeCollection()]

        def delete_collection(self, name: str):
            raise RuntimeError("delete failed")

    fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: FailingClient())
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    result = backend.remove_doc_source("sdk")

    assert result == {
        "ok": False,
        "label": "sdk",
        "path": "sdk/docs",
        "source_removed": True,
        "error": "Source removed, but cleanup was incomplete.",
        "collections": ["sdk.render"],
    }
    assert backend.config.doc_sources == {}
    assert backend.symbol_index.removed == ["sdk"]
    assert backend.retriever.invalidated is True
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["doc_sources"] == {}


def test_doc_backend_filters_results_from_removed_sources():
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config = Config(doc_sources={})
    backend.retriever = FakeRetriever()
    backend.symbol_index = FakeSymbolIndex()

    assert backend.find_symbol("Sdk::Render") is None
    assert backend.search_docs("render mesh") == []


def test_doc_backend_filters_removed_sources_from_api_helpers(monkeypatch):
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config = Config(doc_sources={})
    backend.retriever = FakeRetriever()
    backend.symbol_index = FakeSymbolIndex()
    monkeypatch.setattr("rag.gateway.doc_backend.build_context", lambda results, query, max_tokens: results)

    assert backend.get_api_class("Sdk") is None
    assert backend.get_api_function("Render", "Sdk") is None
    assert backend.build_context("render mesh") == []


def test_doc_backend_remove_source_reports_symbol_flush_failure_after_source_removed(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "backend.yaml"
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = str(config_path)
    backend.config = Config(chroma_dir=str(tmp_path / "chroma"), doc_sources={"sdk": "sdk/docs"})
    backend.retriever = FakeRetriever()

    class FailingFlushSymbolIndex(FakeSymbolIndex):
        def flush(self):
            raise RuntimeError("flush failed")

    backend.symbol_index = FailingFlushSymbolIndex()
    fake_chromadb = types.SimpleNamespace(
        PersistentClient=lambda path: types.SimpleNamespace(list_collections=lambda: [])
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    result = backend.remove_doc_source("sdk")

    assert result == {
        "ok": False,
        "label": "sdk",
        "path": "sdk/docs",
        "source_removed": True,
        "error": "Source removed, but cleanup was incomplete.",
        "symbol_index_error": "flush failed",
    }
    assert backend.config.doc_sources == {}
    assert backend.symbol_index.removed == ["sdk"]
    assert backend.retriever.invalidated is True
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["doc_sources"] == {}

def test_doc_backend_remove_source_preserves_symbols_when_config_remove_fails(tmp_path: Path, monkeypatch):
    symbol_index_path = tmp_path / "symbols.json"
    symbol_index = SymbolIndex(str(symbol_index_path))
    symbol_index._index = {
        "Sdk::Render": {"symbol_id": "Sdk::Render", "source_label": "sdk"},
        "Other::Call": {"symbol_id": "Other::Call", "source_label": "other"},
    }
    symbol_index.flush()
    backend = DocRagBackend.__new__(DocRagBackend)
    backend.config_path = str(tmp_path / "backend.yaml")
    backend.config = Config(chroma_dir=str(tmp_path / "chroma"), doc_sources={"sdk": "sdk/docs"})
    backend.symbol_index = symbol_index
    backend.retriever = FakeRetriever()

    class FakeCollection:
        name = "sdk.render"

    class DeletingClient:
        def __init__(self):
            self.collections = [FakeCollection()]
            self.touched = False

        def list_collections(self):
            self.touched = True
            return self.collections

        def delete_collection(self, name: str):
            self.touched = True
            self.collections = []

    client = DeletingClient()
    fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    def failing_save_config():
        raise RuntimeError("config save failed")

    monkeypatch.setattr(backend, "_save_config", failing_save_config)

    result = backend.remove_doc_source("sdk")

    assert result == {"ok": False, "error": "config save failed"}
    assert backend.config.doc_sources == {"sdk": "sdk/docs"}
    assert SymbolIndex(str(symbol_index_path)).lookup("Sdk::Render")["source_label"] == "sdk"
    assert backend.symbol_index.lookup("Sdk::Render")["source_label"] == "sdk"
    assert backend.retriever.invalidated is False
    assert client.touched is False
