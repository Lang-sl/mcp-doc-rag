from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_gateway_config_path(path: str | None) -> str:
    raw = path or "./gateway.yaml"
    return str(Path(raw).expanduser().resolve())


def gateway_config_identity(path: str | None) -> str:
    """Derive a stable daemon identity from the normalized gateway config path.

    Identity is path-based, not content-based. Editing gateway.yaml does NOT
    change the identity — the same daemon and runtime metadata are reused, and
    ``rag daemon reload`` picks up config changes without spawning a new daemon.
    """
    normalized = normalize_gateway_config_path(path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def default_runtime_path(
    gateway_config_path: str | None,
    runtime_dir: str | None,
    project_root: Path | None = None,
) -> Path:
    if runtime_dir:
        base = Path(runtime_dir).expanduser()
    else:
        base = (project_root or Path.cwd()) / "output" / "runtime"

    identity = gateway_config_identity(gateway_config_path)
    return base / f"daemon-{identity}.json"


def default_log_path(
    gateway_config_path: str | None,
    runtime_dir: str | None,
    project_root: Path | None = None,
) -> Path:
    if runtime_dir:
        base = Path(runtime_dir).expanduser()
    else:
        base = (project_root or Path.cwd()) / "output" / "runtime"

    identity = gateway_config_identity(gateway_config_path)
    return base / f"daemon-{identity}.log"
