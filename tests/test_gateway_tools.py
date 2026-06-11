from __future__ import annotations

import io
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.config import Config
from rag.gateway.codegraph_client import CodeGraphClient
from rag.gateway.config import CodeGraphConfig
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


class FakeProcess:
    def __init__(self, responses: list[dict]):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(json.dumps(item) for item in responses) + "\n")
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = 1


def test_codegraph_client_lists_tools_with_fake_process():
    responses = [
        {"jsonrpc": "2.0", "method": "log", "params": {"message": "starting"}},
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "codegraph_search"}]}},
    ]
    process = FakeProcess(responses)
    client = CodeGraphClient(CodeGraphConfig(), process_factory=lambda command, cwd: process)

    assert client.start() is True
    assert client.available is True
    assert client.tool_names == ["codegraph_search"]
    assert client.tools[0]["name"] == "codegraph_search"

    written = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert [item["method"] for item in written] == ["initialize", "initialized", "tools/list"]


def test_codegraph_client_ignores_unrelated_response_ids():
    responses = [
        {"jsonrpc": "2.0", "id": 99, "result": {"ignored": True}},
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "codegraph_search"}]}},
    ]
    process = FakeProcess(responses)
    client = CodeGraphClient(CodeGraphConfig(), process_factory=lambda command, cwd: process)

    assert client.start() is True
    assert client.tool_names == ["codegraph_search"]


def test_codegraph_client_call_tool_returns_result_content():
    responses = [
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "codegraph_search"}]}},
        {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "result"}]}},
    ]
    process = FakeProcess(responses)
    client = CodeGraphClient(CodeGraphConfig(), process_factory=lambda command, cwd: process)
    client.start()

    result = client.call_tool("codegraph_search", {"query": "mesh"})

    assert result == {"content": [{"type": "text", "text": "result"}]}


def test_codegraph_client_call_tool_returns_unavailable_when_process_exited():
    responses = [
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "codegraph_search"}]}},
    ]
    process = FakeProcess(responses)
    client = CodeGraphClient(CodeGraphConfig(), process_factory=lambda command, cwd: process)
    client.start()
    process.returncode = 1

    result = client.call_tool("codegraph_search", {"query": "mesh"})

    assert result == {"error": "CodeGraph unavailable"}
    assert client.available is False


def test_codegraph_client_tool_error_does_not_disable_client():
    responses = [
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "codegraph_search"}]}},
        {"jsonrpc": "2.0", "id": 3, "error": {"code": -32602, "message": "bad arguments"}},
    ]
    process = FakeProcess(responses)
    client = CodeGraphClient(CodeGraphConfig(), process_factory=lambda command, cwd: process)
    client.start()

    result = client.call_tool("codegraph_search", {"bad": True})

    assert result == {"error": "bad arguments"}
    assert client.available is True


class HangingStdout:
    def readline(self):
        import time
        time.sleep(0.2)
        return ""


class HangingProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = HangingStdout()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = 1


def test_codegraph_client_start_times_out_and_degrades():
    process = HangingProcess()
    client = CodeGraphClient(
        CodeGraphConfig(),
        process_factory=lambda command, cwd: process,
        response_timeout_seconds=0.01,
    )

    assert client.start() is False
    assert client.available is False
    assert client.tool_names == []
    assert process.terminated is True


class FakeGatewayDocBackend:
    def __init__(self) -> None:
        self.searches: list[tuple[str, int]] = []
        self.lookups: list[str] = []

    def find_symbol(self, symbol: str) -> dict | None:
        self.lookups.append(symbol)
        if symbol == "Sdk::Render":
            return {"symbol_id": "api-render", "name": symbol}
        if symbol == "ProjectOnly":
            return {"symbol_id": "api-project", "name": symbol}
        if symbol == "Open":
            return {"symbol_id": "api-open", "name": symbol}
        return None

    def search_docs(self, query: str, top_k: int = 10) -> list[dict]:
        self.searches.append((query, top_k))
        return [{"query": query, "top_k": top_k}]

    def get_api_class(self, class_name: str) -> dict:
        return {"class_name": class_name}

    def get_api_function(self, func_name: str, class_name: str | None = None) -> dict:
        return {"func_name": func_name, "class_name": class_name}

    def list_modules(self, source_label: str | None = None) -> list[str]:
        return [source_label or "all"]


class FakeGatewayCodeGraphClient:
    def __init__(self, result: dict | None = None, available: bool = True) -> None:
        self.available = available
        self.tool_names = ["codegraph_search", "codegraph_extra"]
        self.result = result or {}
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self.result


def test_normalize_symbol_candidates_preserves_exact_template_stripped_and_final_member():
    from rag.gateway.tools import normalize_symbol_candidates

    assert normalize_symbol_candidates("Sdk::Render<Mesh>") == ["Sdk::Render<Mesh>", "Sdk::Render", "Render"]


def test_extract_symbol_names_reads_markdown_headings_from_text_dict():
    from rag.gateway.tools import extract_symbol_names

    value = {
        "text": "## Search Results...\n\n### Sdk::Render<Mesh> (method)\n...\n\n### ProjectOnly (function)\n..."
    }

    assert extract_symbol_names(value) == ["Sdk::Render<Mesh>", "ProjectOnly"]


def test_extract_symbol_names_ignores_generic_name_metadata():
    from rag.gateway.tools import extract_symbol_names

    value = {
        "name": "search results",
        "text": "### Sdk::Render<Mesh> (method)\n...",
    }

    assert extract_symbol_names(value) == ["Sdk::Render<Mesh>"]


def test_smart_search_degrades_to_doc_search_when_codegraph_unavailable():
    from rag.gateway.tools import GatewayTools

    doc_backend = FakeGatewayDocBackend()
    codegraph_client = FakeGatewayCodeGraphClient(available=False)
    tools = GatewayTools(doc_backend, codegraph_client)

    result = tools.smart_search("render mesh", top_k=3)

    assert result["degraded"] is True
    assert result["code_usages"] == []
    assert result["matched_api_symbols"] == []
    assert result["unmatched_code_symbols"] == []
    assert result["doc_results"] == [{"query": "render mesh", "top_k": 3}]
    assert doc_backend.searches == [("render mesh", 3)]
    assert codegraph_client.calls == []


def test_smart_search_keeps_matched_symbols_and_unmatched_symbols_when_codegraph_available():
    from rag.gateway.tools import GatewayTools

    codegraph_result = {
        "content": [
            {
                "type": "text",
                "text": "### Sdk::Render<Mesh> (method)\n...\n\n### Missing::Call (function)\n...",
            }
        ]
    }
    doc_backend = FakeGatewayDocBackend()
    codegraph_client = FakeGatewayCodeGraphClient(codegraph_result)
    tools = GatewayTools(doc_backend, codegraph_client)

    result = tools.smart_search("render mesh", top_k=2)

    assert result["degraded"] is False
    assert result["code_usages"] == {"text": codegraph_result["content"][0]["text"]}
    assert result["matched_api_symbols"] == [{"symbol_id": "api-render", "name": "Sdk::Render"}]
    assert result["unmatched_code_symbols"] == ["Missing::Call"]
    assert doc_backend.searches == [("api-render", 2)]
    assert codegraph_client.calls == [("codegraph_search", {"query": "render mesh"})]


def test_smart_search_does_not_match_unrelated_unqualified_member_fallback():
    from rag.gateway.tools import GatewayTools

    codegraph_result = {
        "content": [
            {
                "type": "text",
                "text": "### Foo::Open (method)\n...",
            }
        ]
    }
    doc_backend = FakeGatewayDocBackend()
    codegraph_client = FakeGatewayCodeGraphClient(codegraph_result)
    tools = GatewayTools(doc_backend, codegraph_client)

    result = tools.smart_search("open file", top_k=2)

    assert result["matched_api_symbols"] == []
    assert result["unmatched_code_symbols"] == ["Foo::Open"]
    assert doc_backend.searches == [("open file", 2)]
    assert "Open" not in doc_backend.lookups


def test_smart_search_degrades_when_codegraph_call_returns_error():
    from rag.gateway.tools import GatewayTools

    doc_backend = FakeGatewayDocBackend()
    codegraph_client = FakeGatewayCodeGraphClient({"error": "bad request"})
    tools = GatewayTools(doc_backend, codegraph_client)

    result = tools.smart_search("render mesh", top_k=4)

    assert result["degraded"] is True
    assert result["matched_api_symbols"] == []
    assert result["unmatched_code_symbols"] == []
    assert result["doc_results"] == [{"query": "render mesh", "top_k": 4}]
    assert result["warnings"] == ["bad request"]


def test_gateway_tools_call_tool_routes_missing_doc_passthrough_methods():
    from rag.gateway.tools import GatewayTools

    tools = GatewayTools(FakeGatewayDocBackend(), FakeGatewayCodeGraphClient())

    assert tools.call_tool("get_api_class", {"class_name": "Sdk"}) == {"class_name": "Sdk"}
    assert tools.call_tool("get_api_function", {"func_name": "Render", "class_name": "Sdk"}) == {
        "func_name": "Render",
        "class_name": "Sdk",
    }
    assert tools.call_tool("list_modules", {"source_label": "sdk"}) == ["sdk"]
