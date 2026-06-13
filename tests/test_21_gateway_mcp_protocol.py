from __future__ import annotations

import json


def test_mcp_protocol_ignores_notifications_without_id():
    from rag.gateway.mcp_protocol import handle_mcp_request

    response = handle_mcp_request(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        service=object(),
        server_name="mcp-doc-rag-gateway",
    )

    assert response is None


def test_mcp_protocol_serializes_tool_result_as_text_json():
    from rag.gateway.mcp_protocol import handle_mcp_request

    class Service:
        def call_tool(self, name, arguments):
            return {"name": name, "arguments": arguments}

    response = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "search_docs", "arguments": {"query": "render"}},
        },
        service=Service(),
        server_name="mcp-doc-rag-gateway",
    )

    payload = json.loads(response["result"]["content"][0]["text"])

    assert payload == {"name": "search_docs", "arguments": {"query": "render"}}


def test_mcp_protocol_returns_tools_list():
    from rag.gateway.mcp_protocol import handle_mcp_request

    class Service:
        def list_tools(self):
            return [{"name": "search_docs"}]

    response = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 8, "method": "tools/list", "params": {}},
        service=Service(),
        server_name="mcp-doc-rag-gateway",
    )

    assert response["result"]["tools"] == [{"name": "search_docs"}]
