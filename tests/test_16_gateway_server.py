from __future__ import annotations

import json
from types import SimpleNamespace

from rag.gateway.tools import DOC_TOOL_NAMES


def test_build_tools_list_includes_smart_search_doc_tools_and_codegraph_tools():
    from rag.gateway.server import build_tools_list

    codegraph_tools = [
        {
            "name": "codegraph_search",
            "description": "Search code graph",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    tools = build_tools_list(codegraph_tools)
    tool_names = [tool["name"] for tool in tools]

    assert tool_names[0] == "smart_search"
    assert set(DOC_TOOL_NAMES).issubset(set(tool_names))
    assert "codegraph_search" in tool_names

    smart_search_tool = tools[0]
    assert smart_search_tool["inputSchema"] == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }


def test_handle_request_initialize_returns_gateway_server_metadata():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        SimpleNamespace(),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mcp-doc-rag-gateway", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        },
    }


def test_handle_request_initialized_notification_returns_none():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        SimpleNamespace(),
    )

    assert response is None


def test_handle_request_mcp_initialized_notification_returns_none():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        SimpleNamespace(),
    )

    assert response is None


def test_handle_request_tools_list_handles_missing_codegraph_client():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        SimpleNamespace(),
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    tools = response["result"]["tools"]
    assert tools[0]["name"] == "smart_search"
    assert set(DOC_TOOL_NAMES).issubset({tool["name"] for tool in tools})


def test_handle_request_tools_list_includes_runtime_codegraph_tools():
    from rag.gateway.server import handle_request

    codegraph_tool = {
        "name": "codegraph_search",
        "description": "Search code graph",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    tools_handler = SimpleNamespace(codegraph_client=SimpleNamespace(tools=[codegraph_tool]))

    response = handle_request(
        {"jsonrpc": "2.0", "id": 22, "method": "tools/list", "params": {}},
        tools_handler,
    )

    assert codegraph_tool in response["result"]["tools"]


def test_handle_request_tools_call_serializes_result_as_mcp_text():
    from rag.gateway.server import handle_request

    result = {"message": "你好", "items": [1, 2]}
    tools_handler = SimpleNamespace(
        call_tool=lambda name, arguments: {
            "tool": name,
            "arguments": arguments,
            "result": result,
        }
    )

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "smart_search", "arguments": {"query": "render"}},
        },
        tools_handler,
    )

    expected_payload = {
        "tool": "smart_search",
        "arguments": {"query": "render"},
        "result": result,
    }
    assert response == {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(expected_payload, ensure_ascii=False, indent=2),
                }
            ]
        },
    }


def test_handle_request_unknown_method_returns_jsonrpc_error():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}},
        SimpleNamespace(),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 4,
        "error": {"code": -32601, "message": "Method not found: unknown/method"},
    }


def test_handle_request_unknown_tool_returns_jsonrpc_error():
    from rag.gateway.server import handle_request

    def call_tool(name: str, arguments: dict):
        raise KeyError(name)

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        },
        SimpleNamespace(call_tool=call_tool),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {"code": -32601, "message": "Tool not found: missing_tool"},
    }


def test_handle_request_tool_failure_returns_jsonrpc_error():
    from rag.gateway.server import handle_request

    def call_tool(name: str, arguments: dict):
        raise RuntimeError("tool failed")

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "smart_search", "arguments": {"query": "render"}},
        },
        SimpleNamespace(call_tool=call_tool),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 6,
        "error": {"code": -32000, "message": "tool failed"},
    }


def test_handle_request_codegraph_error_result_returns_jsonrpc_error():
    from rag.gateway.server import handle_request

    tools_handler = SimpleNamespace(
        codegraph_client=SimpleNamespace(tool_names=["codegraph_search"]),
        call_tool=lambda name, arguments: {"error": "bad codegraph request"},
    )

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 66,
            "method": "tools/call",
            "params": {"name": "codegraph_search", "arguments": {"query": "render"}},
        },
        tools_handler,
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 66,
        "error": {"code": -32000, "message": "bad codegraph request"},
    }


def test_handle_request_doc_error_result_remains_successful_payload():
    from rag.gateway.server import handle_request

    tools_handler = SimpleNamespace(
        codegraph_client=SimpleNamespace(tool_names=["codegraph_search"]),
        call_tool=lambda name, arguments: {"ok": False, "error": "source missing"},
    )

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 67,
            "method": "tools/call",
            "params": {"name": "remove_doc_source", "arguments": {"label": "sdk"}},
        },
        tools_handler,
    )

    assert "result" in response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload == {"ok": False, "error": "source missing"}


def test_handle_request_rejects_non_object_tool_params():
    from rag.gateway.server import handle_request

    response = handle_request(
        {"jsonrpc": "2.0", "id": 68, "method": "tools/call", "params": []},
        SimpleNamespace(),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 68,
        "error": {"code": -32602, "message": "tools/call params must be an object"},
    }


def test_handle_request_rejects_non_object_tool_arguments():
    from rag.gateway.server import handle_request

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 69,
            "method": "tools/call",
            "params": {"name": "list_doc_sources", "arguments": []},
        },
        SimpleNamespace(),
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 69,
        "error": {"code": -32602, "message": "tools/call arguments must be an object"},
    }


def test_create_tools_wires_gateway_dependencies(monkeypatch):
    from rag.gateway import server

    created: dict[str, object] = {}

    config = SimpleNamespace(codegraph="codegraph-config", doc_rag_config_path="doc-config.yaml")

    class FakeDocRagBackend:
        def __init__(self, config_path=None):
            created["doc_rag_config_path"] = config_path

    class FakeCodeGraphClient:
        def __init__(self, config_value):
            created["codegraph_config"] = config_value
            self.tools = [{"name": "codegraph_search", "inputSchema": {}}]
            self.tool_names = ["codegraph_search"]

        def start(self):
            created["codegraph_started"] = True
            return True

    class FakeGatewayTools:
        def __init__(self, doc_backend, codegraph_client, codegraph_lifecycle=None):
            created["gateway_tools_args"] = (doc_backend, codegraph_client)
            self.doc_backend = doc_backend
            self.codegraph_client = codegraph_client

    monkeypatch.setattr(server, "load_gateway_config", lambda config_path=None: config)
    monkeypatch.setattr(server, "DocRagBackend", FakeDocRagBackend)
    monkeypatch.setattr(server, "CodeGraphClient", FakeCodeGraphClient)
    monkeypatch.setattr(server, "GatewayTools", FakeGatewayTools)

    tools = server.create_tools("gateway.yaml")

    assert created["doc_rag_config_path"] == "doc-config.yaml"
    assert created["codegraph_config"] == "codegraph-config"
    assert created["codegraph_started"] is True
    assert created["gateway_tools_args"] == (tools.doc_backend, tools.codegraph_client)


def test_create_tools_swallows_codegraph_start_failure(monkeypatch):
    from rag.gateway import server

    config = SimpleNamespace(codegraph="codegraph-config", doc_rag_config_path=None)

    class FakeDocRagBackend:
        def __init__(self, config_path=None):
            self.config_path = config_path

    class FakeCodeGraphClient:
        def __init__(self, config_value):
            self.config = config_value

        def start(self):
            raise RuntimeError("boom")

    class FakeGatewayTools:
        def __init__(self, doc_backend, codegraph_client, codegraph_lifecycle=None):
            self.doc_backend = doc_backend
            self.codegraph_client = codegraph_client

    monkeypatch.setattr(server, "load_gateway_config", lambda config_path=None: config)
    monkeypatch.setattr(server, "DocRagBackend", FakeDocRagBackend)
    monkeypatch.setattr(server, "CodeGraphClient", FakeCodeGraphClient)
    monkeypatch.setattr(server, "GatewayTools", FakeGatewayTools)

    tools = server.create_tools()

    assert isinstance(tools.doc_backend, FakeDocRagBackend)
    assert isinstance(tools.codegraph_client, FakeCodeGraphClient)


def test_build_tools_list_includes_lifecycle_tools_when_configured():
    from rag.gateway.server import build_tools_list

    tools = build_tools_list([], include_codegraph_lifecycle=True)
    tool_names = [tool["name"] for tool in tools]

    assert "codegraph_init" in tool_names
    assert "codegraph_reindex" in tool_names
    assert "codegraph_sync" in tool_names
    assert "codegraph_index_status" in tool_names
    assert "codegraph_restart" in tool_names

    reindex_tool = next(tool for tool in tools if tool["name"] == "codegraph_reindex")
    assert reindex_tool["inputSchema"]["properties"]["force"] == {"type": "boolean", "default": False}


def test_build_tools_list_omits_lifecycle_tools_without_codegraph_config():
    from rag.gateway.server import build_tools_list

    tools = build_tools_list([], include_codegraph_lifecycle=False)
    tool_names = [tool["name"] for tool in tools]

    assert "codegraph_init" not in tool_names
    assert "codegraph_index_status" not in tool_names


def test_create_tools_wires_codegraph_lifecycle(monkeypatch):
    from rag.gateway import server

    created: dict[str, object] = {}
    config = SimpleNamespace(codegraph="codegraph-config", doc_rag_config_path=None)

    class FakeDocRagBackend:
        def __init__(self, config_path=None):
            self.config_path = config_path

    class FakeCodeGraphClient:
        def __init__(self, config_value):
            self.config = config_value
            self.tools = []
            self.tool_names = []

        def start(self):
            return False

    class FakeLifecycle:
        def __init__(self, config_value, client):
            created["lifecycle_args"] = (config_value, client)

    class FakeGatewayTools:
        def __init__(self, doc_backend, codegraph_client, codegraph_lifecycle):
            self.doc_backend = doc_backend
            self.codegraph_client = codegraph_client
            self.codegraph_lifecycle = codegraph_lifecycle
            created["gateway_tools_args"] = (doc_backend, codegraph_client, codegraph_lifecycle)

    monkeypatch.setattr(server, "load_gateway_config", lambda config_path=None: config)
    monkeypatch.setattr(server, "DocRagBackend", FakeDocRagBackend)
    monkeypatch.setattr(server, "CodeGraphClient", FakeCodeGraphClient)
    monkeypatch.setattr(server, "CodeGraphLifecycle", FakeLifecycle)
    monkeypatch.setattr(server, "GatewayTools", FakeGatewayTools)

    tools = server.create_tools()

    assert created["lifecycle_args"][0] == "codegraph-config"
    assert created["gateway_tools_args"] == (tools.doc_backend, tools.codegraph_client, tools.codegraph_lifecycle)


def test_main_ignores_blank_and_invalid_json_lines(monkeypatch, capsys):
    from rag.gateway import server

    requests = [
        "\n",
        "not-json\n",
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}}) + "\n",
    ]

    monkeypatch.setattr(server.sys, "stdin", requests)
    monkeypatch.setattr(server, "create_tools", lambda config_path=None: SimpleNamespace())

    server.main()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "mcp-doc-rag-gateway", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            },
            ensure_ascii=False,
        )
    ]


def test_main_shuts_down_codegraph_client_when_stdin_ends(monkeypatch):
    from rag.gateway import server

    class FakeCodeGraphClient:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    codegraph_client = FakeCodeGraphClient()
    tools_handler = SimpleNamespace(codegraph_client=codegraph_client)

    monkeypatch.setattr(server.sys, "stdin", [])
    monkeypatch.setattr(server, "create_tools", lambda config_path=None: tools_handler)

    server.main()

    assert codegraph_client.shutdown_called is True
