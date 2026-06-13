from __future__ import annotations

import json
import sys
from typing import Any

from rag.gateway.codegraph_client import CodeGraphClient
from rag.gateway.config import load_gateway_config
from rag.gateway.doc_backend import DocRagBackend
from rag.gateway.tools import CODEGRAPH_LIFECYCLE_TOOL_NAMES, DOC_TOOL_NAMES, GatewayTools
from rag.gateway.codegraph_lifecycle import CodeGraphLifecycle
from rag.server import TOOLS


_SMART_SEARCH_TOOL = {
    "name": "smart_search",
    "description": "Search code usage first, then map to API docs with doc-rag fallback.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}

_CODEGRAPH_LIFECYCLE_TOOLS = [
    {
        "name": "codegraph_init",
        "description": "Initialize the configured CodeGraph project and build its first index.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "codegraph_reindex",
        "description": "Rebuild the configured CodeGraph index.",
        "inputSchema": {
            "type": "object",
            "properties": {"force": {"type": "boolean", "default": False}},
        },
    },
    {
        "name": "codegraph_sync",
        "description": "Incrementally sync the configured CodeGraph index.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "codegraph_index_status",
        "description": "Report configured CodeGraph index and gateway subprocess health.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "codegraph_restart",
        "description": "Restart the configured CodeGraph MCP subprocess and reload its tools.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

def build_tools_list(codegraph_tools: list[dict] | None, include_codegraph_lifecycle: bool = False) -> list[dict]:
    doc_tools = [tool for tool in TOOLS if tool.get("name") in DOC_TOOL_NAMES]
    lifecycle_tools = _CODEGRAPH_LIFECYCLE_TOOLS if include_codegraph_lifecycle else []
    return [_SMART_SEARCH_TOOL, *doc_tools, *lifecycle_tools, *(codegraph_tools or [])]


# Adapter that wraps GatewayTools for the shared MCP protocol helper.
# Intentionally mirrors GatewayToolService.list_tools() logic rather than
# coupling the direct gateway path to GatewayToolService — avoids circular
# imports and behavioral churn.  Both call the shared build_tools_list().
class _LegacyGatewayService:
    def __init__(self, tools_handler: GatewayTools) -> None:
        self.tools_handler = tools_handler

    def list_tools(self) -> list[dict]:
        codegraph_client = getattr(self.tools_handler, "codegraph_client", None)
        codegraph_tools = getattr(codegraph_client, "tools", []) if codegraph_client is not None else []
        codegraph_lifecycle = getattr(self.tools_handler, "codegraph_lifecycle", None)
        return build_tools_list(codegraph_tools, codegraph_lifecycle is not None)

    def call_tool(self, name: str, arguments: dict) -> Any:
        result = self.tools_handler.call_tool(name, arguments)
        if _is_codegraph_error_result(self.tools_handler, name, result):
            raise RuntimeError(str(result["error"]))
        return result


def handle_request(request: dict, tools_handler: GatewayTools) -> dict | None:
    from rag.gateway.mcp_protocol import handle_mcp_request

    service = _LegacyGatewayService(tools_handler)
    return handle_mcp_request(request, service, "mcp-doc-rag-gateway")


def _is_codegraph_error_result(tools_handler: GatewayTools, tool_name: str, result: Any) -> bool:
    codegraph_client = getattr(tools_handler, "codegraph_client", None)
    if codegraph_client is None:
        return False
    if tool_name not in getattr(codegraph_client, "tool_names", []):
        return False
    return isinstance(result, dict) and "error" in result


def create_tools(config_path: str | None = None) -> GatewayTools:
    config = load_gateway_config(config_path)
    doc_backend = DocRagBackend(config.doc_rag_config_path)
    codegraph_client = CodeGraphClient(config.codegraph)
    try:
        codegraph_client.start()
    except Exception:
        pass
    codegraph_lifecycle = CodeGraphLifecycle(config.codegraph, codegraph_client) if config.codegraph is not None else None
    return GatewayTools(doc_backend, codegraph_client, codegraph_lifecycle)


def main() -> None:
    tools_handler = create_tools()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                request: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = handle_request(request, tools_handler)
            if response is None:
                continue

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        codegraph_client = getattr(tools_handler, "codegraph_client", None)
        shutdown = getattr(codegraph_client, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    main()
