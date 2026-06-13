from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rag.daemon.client import DaemonClient
from rag.daemon.runtime import RuntimeMetadata, read_metadata


StartProcess = Callable[[str, Path, Path | None], bool]
HealthCheck = Callable[[RuntimeMetadata], bool]


def ensure_daemon(
    gateway_config_path: str,
    runtime_path: Path,
    autostart: bool,
    start_process: StartProcess | None = None,
    health_check: HealthCheck | None = None,
    log_path: Path | None = None,
    wait_seconds: float = 5.0,
) -> RuntimeMetadata | None:
    metadata = read_metadata(runtime_path)
    checker = health_check or _metadata_is_healthy
    if metadata is not None and checker(metadata):
        return metadata

    if not autostart:
        return None

    starter = start_process or start_daemon_process
    if log_path is None:
        log_path = runtime_path.with_suffix(".log")
    starter(gateway_config_path, runtime_path, log_path)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        metadata = read_metadata(runtime_path)
        if metadata is not None and checker(metadata):
            return metadata
        time.sleep(0.1)
    return metadata


def start_daemon_process(gateway_config_path: str, runtime_path: Path, log_path: Path | None = None) -> bool:
    if log_path is None:
        log_path = runtime_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "rag",
        "daemon",
        "start",
        "--gateway-config",
        gateway_config_path,
        "--runtime-path",
        str(runtime_path),
        "--log-path",
        str(log_path),
    ]

    # Parent opens the log file in "w" mode to capture early child errors
    # (import failures, CLI parse errors, etc.) that happen before
    # commands.start() takes over.  The handle is closed immediately after
    # Popen returns — the child inherits the fd and the parent does not
    # keep a long-lived handle.
    log_file = open(str(log_path), "w", encoding="utf-8")
    log_file.write(f"daemon autostart {datetime.now(timezone.utc).isoformat()}\n")
    log_file.flush()
    subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    log_file.close()
    return True


def _metadata_is_healthy(metadata: RuntimeMetadata) -> bool:
    try:
        health = DaemonClient(metadata).health()
        return bool(health.get("ok", False))
    except Exception:
        return False
