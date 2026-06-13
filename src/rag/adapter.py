from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TextIO

from rag.daemon.client import DaemonClient
from rag.daemon.config import default_runtime_path, normalize_gateway_config_path
from rag.daemon.process import ensure_daemon
from rag.gateway.config import load_gateway_config
from rag.gateway.mcp_protocol import handle_mcp_request


class AdapterService:
    def __init__(self, client: DaemonClient) -> None:
        self.client = client

    def list_tools(self) -> list[dict]:
        return self.client.list_tools()

    def call_tool(self, name: str, arguments: dict):
        return self.client.call_tool(name, arguments)


def resolve_adapter_metadata(config_path: str | None = None):
    gateway_config_path = normalize_gateway_config_path(config_path or os.environ.get("GATEWAY_CONFIG_PATH"))
    config = load_gateway_config(gateway_config_path)
    runtime_path = default_runtime_path(
        gateway_config_path,
        config.daemon.runtime_dir,
        project_root=Path.cwd(),
    )
    # Derive log path from the same runtime_dir so user overrides are respected.
    log_path = runtime_path.with_suffix(".log")
    return ensure_daemon(
        gateway_config_path=gateway_config_path,
        runtime_path=runtime_path,
        autostart=config.daemon.autostart,
        log_path=log_path,
    )


def run_stdio(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    metadata = resolve_adapter_metadata()
    if metadata is None:
        service = None
    else:
        service = AdapterService(DaemonClient(metadata))

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        if service is None:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": "Gateway daemon is not running and autostart failed"},
            }
        else:
            response = handle_mcp_request(request, service, "mcp-doc-rag-gateway-adapter")

        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()


def main() -> None:
    run_stdio()
