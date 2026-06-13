from __future__ import annotations

from types import SimpleNamespace


def test_gateway_tool_service_without_codegraph_lists_doc_tools(monkeypatch):
    from rag.gateway.config import GatewayConfig
    from rag.gateway.service import GatewayToolService
    from rag.gateway.tools import DOC_TOOL_NAMES

    class FakeDocBackend:
        def health(self):
            return {"ok": True}

    service = GatewayToolService(
        GatewayConfig(),
        doc_backend_factory=lambda config_path: FakeDocBackend(),
        codegraph_client_factory=lambda config: None,
        lifecycle_factory=lambda config, client: None,
    )

    tool_names = {tool["name"] for tool in service.list_tools()}

    assert "smart_search" in tool_names
    assert DOC_TOOL_NAMES.issubset(tool_names)
    assert service.health()["codegraph"]["state"] == "not_configured"


def test_gateway_tool_service_with_codegraph_lists_lifecycle_and_passthrough_tools():
    from rag.gateway.config import CodeGraphConfig, GatewayConfig
    from rag.gateway.service import GatewayToolService

    class FakeDocBackend:
        def health(self):
            return {"ok": True}

    class FakeCodeGraphClient:
        tools = [{"name": "codegraph_search", "description": "search", "inputSchema": {"type": "object"}}]
        tool_names = ["codegraph_search"]
        available = True
        last_startup_error = None

        def start(self):
            return True

        def health(self):
            return {"available": True, "process_running": True, "tool_names": self.tool_names}

        def shutdown(self):
            self.available = False

    service = GatewayToolService(
        GatewayConfig(codegraph=CodeGraphConfig()),
        doc_backend_factory=lambda config_path: FakeDocBackend(),
        codegraph_client_factory=lambda config: FakeCodeGraphClient(),
        lifecycle_factory=lambda config, client: SimpleNamespace(index_status=lambda: {"ok": True}),
    )

    tool_names = {tool["name"] for tool in service.list_tools()}

    assert "codegraph_index_status" in tool_names
    assert "codegraph_search" in tool_names
    assert service.health()["codegraph"]["state"] == "configured_running"


def test_gateway_tool_service_call_tool_delegates_to_gateway_tools():
    from rag.gateway.config import GatewayConfig
    from rag.gateway.service import GatewayToolService

    class FakeDocBackend:
        def search_docs(self, query, top_k=10, source_label=None, module=None):
            return [{"query": query, "top_k": top_k}]

        def health(self):
            return {"ok": True}

    service = GatewayToolService(
        GatewayConfig(),
        doc_backend_factory=lambda config_path: FakeDocBackend(),
        codegraph_client_factory=lambda config: None,
        lifecycle_factory=lambda config, client: None,
    )

    result = service.call_tool("search_docs", {"query": "render", "top_k": 3})

    assert result == [{"query": "render", "top_k": 3}]
