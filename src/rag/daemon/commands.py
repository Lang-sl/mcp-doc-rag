from __future__ import annotations

import argparse
import atexit
import os
import secrets
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from rag.daemon.client import DaemonClient
from rag.daemon.config import default_runtime_path, default_log_path, gateway_config_identity, normalize_gateway_config_path
from rag.daemon.http_server import GatewayDaemonHttpServer
from rag.daemon.runtime import RuntimeMetadata, delete_metadata, format_status, read_metadata, write_metadata
from rag.gateway.config import load_gateway_config
from rag.gateway.service import GatewayToolService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag daemon")
    sub = parser.add_subparsers(dest="action", required=True)
    p_start = sub.add_parser("start")
    p_start.add_argument("--gateway-config", default=None)
    p_start.add_argument("--runtime-path", default=None)
    p_start.add_argument("--log-path", default=None)
    for name in ("stop", "status", "reload"):
        p = sub.add_parser(name)
        p.add_argument("--gateway-config", default=None)
        p.add_argument("--runtime-path", default=None)

    args = parser.parse_args(argv)
    gateway_config_path = normalize_gateway_config_path(args.gateway_config or os.environ.get("GATEWAY_CONFIG_PATH"))
    config = load_gateway_config(gateway_config_path)
    runtime_path = Path(args.runtime_path) if args.runtime_path else default_runtime_path(
        gateway_config_path,
        config.daemon.runtime_dir,
        project_root=Path.cwd(),
    )

    if args.action == "start":
        log_path = Path(args.log_path) if args.log_path else default_log_path(
            gateway_config_path, config.daemon.runtime_dir
        )
        return start(gateway_config_path, runtime_path, log_path)
    if args.action == "stop":
        return stop(runtime_path)
    if args.action == "status":
        return status(runtime_path)
    if args.action == "reload":
        return reload(runtime_path)
    return 1


def start(gateway_config_path: str, runtime_path: Path, log_path: Path | None = None) -> int:
    existing = read_metadata(runtime_path)
    if existing is not None:
        try:
            if DaemonClient(existing).health().get("ok", False):
                print(format_status(existing))
                return 0
        except Exception:
            delete_metadata(runtime_path)

    config = load_gateway_config(gateway_config_path)
    if log_path is None:
        log_path = default_log_path(gateway_config_path, config.daemon.runtime_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode. When started via autostart the parent has already
    # opened the file in "w" mode, captured early child errors, and closed its
    # handle. Appending preserves the parent's content. When started directly
    # by the user this is the first open.
    log_file = open(str(log_path), "a", encoding="utf-8")
    log_file.write(f"\n--- daemon start {datetime.now(timezone.utc).isoformat()} ---\n")
    log_file.flush()

    token = secrets.token_urlsafe(32)
    service = GatewayToolService(config, gateway_config_path=gateway_config_path)

    # Save original streams, then redirect to log file.
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    server = GatewayDaemonHttpServer(config.daemon.host, config.daemon.port, token, service)
    metadata = RuntimeMetadata(
        pid=os.getpid(),
        host=config.daemon.host,
        port=server.port,
        token=token,
        gateway_config_path=gateway_config_path,
        identity=gateway_config_identity(gateway_config_path),
        started_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        log_path=str(log_path),
    )
    write_metadata(runtime_path, metadata)

    # Register cleanup for crash/unexpected termination paths that the
    # finally block below cannot cover (SIGTERM, SIGINT, SystemExit).
    atexit.register(_cleanup, runtime_path, _stdout, _stderr, log_file)

    def _handle_signal(signum: int, frame: object) -> None:
        # Delete metadata in the handler itself rather than relying on the
        # finally block below.  On Windows os.kill(pid, SIGTERM) maps to
        # TerminateProcess — a hard kill where no Python code runs after the
        # handler returns — so the finally block is never reached.  For that
        # case process.py:ensure_daemon() provides a second cleanup path via
        # pid_is_running().  On POSIX and for SIGINT the handler + finally
        # provide two independent cleanup attempts.
        try:
            delete_metadata(runtime_path)
        except Exception:
            pass
        try:
            server.stop()
        except Exception:
            pass

    original_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    original_sigint = signal.signal(signal.SIGINT, _handle_signal)

    # Print status to the original stdout so the CLI user sees it.
    _stdout.write(format_status(metadata) + "\n")
    _stdout.flush()

    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
        atexit.unregister(_cleanup)
        # Restore original streams before closing the log file.
        sys.stdout, sys.stderr = _stdout, _stderr
        log_file.close()
        delete_metadata(runtime_path)
    return 0


def _cleanup(runtime_path: Path, _stdout, _stderr, log_file) -> None:
    """atexit handler: restore streams, close log, delete metadata.

    Runs when the process terminates via SystemExit or normal exit, covering
    paths the finally block in start() cannot reach (SIGTERM after the signal
    handler triggers server.stop(), os._exit in a third-party lib, etc.).
    """
    try:
        sys.stdout, sys.stderr = _stdout, _stderr
    except Exception:
        pass
    try:
        log_file.close()
    except Exception:
        pass
    try:
        delete_metadata(runtime_path)
    except Exception:
        pass


def stop(runtime_path: Path) -> int:
    metadata = read_metadata(runtime_path)
    if metadata is None:
        print("Gateway daemon is not running")
        return 0
    try:
        DaemonClient(metadata).shutdown()
    except Exception:
        delete_metadata(runtime_path)
    return 0


def status(runtime_path: Path) -> int:
    metadata = read_metadata(runtime_path)
    print(format_status(metadata))
    if metadata is None:
        return 0
    try:
        print(DaemonClient(metadata).health())
    except Exception as exc:
        print(f"Daemon health check failed: {exc}")
    return 0


def reload(runtime_path: Path) -> int:
    metadata = read_metadata(runtime_path)
    if metadata is None:
        print("Gateway daemon is not running")
        return 1
    print(DaemonClient(metadata).reload())
    return 0
