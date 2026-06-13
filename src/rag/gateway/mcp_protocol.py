from __future__ import annotations

import json
from typing import Any


def handle_mcp_request(request: dict, service: Any, server_name: str) -> dict | None:
    request_id = request.get("id")
    method = request.get("method")

    if "id" not in request and method != "initialize":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": server_name, "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": service.list_tools()},
        }

    if method == "tools/call":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, -32602, "tools/call params must be an object")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "tools/call arguments must be an object")

        tool_name = params.get("name", "")
        try:
            result = service.call_tool(tool_name, arguments)
        except KeyError:
            return _error(request_id, -32601, f"Tool not found: {tool_name}")
        except Exception as exc:
            return _error(request_id, -32000, str(exc))

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
                ]
            },
        }

    return _error(request_id, -32601, f"Method not found: {method}")


def _error(request_id: Any, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
