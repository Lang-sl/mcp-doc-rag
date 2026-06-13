from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

_DEFAULT_CODEGRAPH_ARGS = ["-y", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"]
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class CodeGraphConfig:
    command: str = "npx"
    args: list[str] = field(default_factory=lambda: list(_DEFAULT_CODEGRAPH_ARGS))
    cwd: str = "."


@dataclass(frozen=True)
class DaemonConfig:
    autostart: bool = True
    host: str = "127.0.0.1"
    port: int = 0
    runtime_dir: str | None = None


@dataclass(frozen=True)
class GatewayConfig:
    codegraph: CodeGraphConfig | None = None
    doc_rag_config_path: str | None = None
    daemon: DaemonConfig = field(default_factory=DaemonConfig)


def load_gateway_config(path: str | None = None) -> GatewayConfig:
    if path is None:
        path = os.environ.get("GATEWAY_CONFIG_PATH", "./gateway.yaml")

    if not os.path.isfile(path):
        return GatewayConfig()

    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if not isinstance(loaded, dict):
        return GatewayConfig()

    codegraph_data = loaded.get("codegraph")
    codegraph = None
    if isinstance(codegraph_data, dict):
        args = codegraph_data.get("args", _DEFAULT_CODEGRAPH_ARGS)
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            args = _DEFAULT_CODEGRAPH_ARGS
        command = codegraph_data.get("command", "npx")
        if not isinstance(command, str):
            command = "npx"
        cwd = codegraph_data.get("cwd", ".")
        if not isinstance(cwd, str):
            cwd = "."
        codegraph = CodeGraphConfig(
            command=command,
            args=list(args),
            cwd=cwd,
        )

    doc_rag = loaded.get("doc_rag")
    doc_rag_config_path = None
    if isinstance(doc_rag, dict):
        doc_rag_config_path = doc_rag.get("config_path")
        if not isinstance(doc_rag_config_path, str):
            doc_rag_config_path = None

    return GatewayConfig(codegraph=codegraph, doc_rag_config_path=doc_rag_config_path, daemon=_load_daemon_config(loaded.get("daemon")))



def _load_daemon_config(data: object) -> DaemonConfig:
    if not isinstance(data, dict):
        return DaemonConfig()

    autostart = data.get("autostart", True)
    if not isinstance(autostart, bool):
        autostart = True

    host = data.get("host", "127.0.0.1")
    if not isinstance(host, str) or host not in _LOOPBACK_HOSTS:
        host = "127.0.0.1"

    port = data.get("port", 0)
    if not isinstance(port, int) or not (0 <= port <= 65535):
        port = 0

    runtime_dir = data.get("runtime_dir")
    if not isinstance(runtime_dir, str) or not runtime_dir.strip():
        runtime_dir = None

    return DaemonConfig(
        autostart=autostart,
        host=host,
        port=port,
        runtime_dir=runtime_dir,
    )
