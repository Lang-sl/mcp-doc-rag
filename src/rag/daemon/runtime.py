from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeMetadata:
    pid: int
    host: str
    port: int
    token: str
    gateway_config_path: str
    identity: str
    started_at: str
    log_path: str = ""


def read_metadata(path: Path) -> RuntimeMetadata | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        return RuntimeMetadata(
            pid=int(data["pid"]),
            host=str(data["host"]),
            port=int(data["port"]),
            token=str(data["token"]),
            gateway_config_path=str(data["gateway_config_path"]),
            identity=str(data["identity"]),
            started_at=str(data["started_at"]),
            log_path=str(data.get("log_path", "")),
        )
    except Exception:
        return None


def write_metadata(path: Path, metadata: RuntimeMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")


def delete_metadata(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def format_status(metadata: RuntimeMetadata | None) -> str:
    if metadata is None:
        return "Daemon runtime metadata: not found"
    return "\n".join(
        [
            f"Daemon PID: {metadata.pid}",
            f"Daemon address: {metadata.host}:{metadata.port}",
            f"Gateway config: {metadata.gateway_config_path}",
            f"Identity: {metadata.identity}",
            f"Started at: {metadata.started_at}",
            f"Log: {metadata.log_path}",
        ]
    )
