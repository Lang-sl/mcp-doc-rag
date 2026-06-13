from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from rag.gateway.config import CodeGraphConfig


CommandRunner = Callable[[list[str], str], subprocess.CompletedProcess]


class CodeGraphLifecycle:
    def __init__(
        self,
        config: CodeGraphConfig | None,
        codegraph_client: Any | None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.codegraph_client = codegraph_client
        self.command_runner = command_runner or _default_command_runner

    def index_status(self) -> dict:
        validation = self._validate_project(require_index=False)
        if not validation["configured"] or not validation["cwd_exists"] or not validation["index_exists"]:
            return {**validation, "ok": False, "gateway": self._gateway_health()}

        cli = self._run_cli("status")
        return {
            **validation,
            "ok": cli["returncode"] == 0,
            "cli": cli,
            "gateway": self._gateway_health(),
        }

    def init(self) -> dict:
        validation = self._validate_project(require_index=False)
        if not validation["configured"] or not validation["cwd_exists"]:
            return {**validation, "ok": False, "gateway": self._gateway_health()}
        if validation["index_exists"]:
            return {
                **self.index_status(),
                "ok": True,
                "action": "already_initialized",
                "message": "CodeGraph index already exists.",
            }

        cli = self._run_cli("init", "-i", validation["cwd"])
        result = {
            **self._validate_project(require_index=False),
            "ok": cli["returncode"] == 0,
            "action": "initialized" if cli["returncode"] == 0 else "init_failed",
            "cli": cli,
            "gateway": self._gateway_health(),
        }
        if result["ok"]:
            result["restart"] = self._restart_client()
            result["gateway"] = self._gateway_health()
        return result

    def reindex(self, force: bool = False) -> dict:
        validation = self._validate_project(require_index=True)
        if not validation["ok"]:
            return {**validation, "gateway": self._gateway_health()}

        extra_args = ["--force"] if force else []
        cli = self._run_cli("index", *extra_args)
        result = {
            **self._validate_project(require_index=False),
            "ok": cli["returncode"] == 0,
            "action": "reindexed" if cli["returncode"] == 0 else "reindex_failed",
            "cli": cli,
            "gateway": self._gateway_health(),
        }
        if result["ok"]:
            result["restart"] = self._restart_client()
            result["gateway"] = self._gateway_health()
        return result

    def sync(self) -> dict:
        validation = self._validate_project(require_index=True)
        if not validation["ok"]:
            return {**validation, "gateway": self._gateway_health()}

        cli = self._run_cli("sync")
        changed = _stdout_suggests_changes(cli["stdout"])
        result = {
            **self._validate_project(require_index=False),
            "ok": cli["returncode"] == 0,
            "action": "synced" if cli["returncode"] == 0 else "sync_failed",
            "changed": changed,
            "cli": cli,
            "gateway": self._gateway_health(),
        }
        if result["ok"] and (changed or not result["gateway"]["process_running"]):
            result["restart"] = self._restart_client()
            result["gateway"] = self._gateway_health()
        return result

    def restart(self) -> dict:
        validation = self._validate_project(require_index=False)
        restart = self._restart_client()
        return {
            **validation,
            "ok": restart["ok"],
            "action": "restarted" if restart["ok"] else "restart_failed",
            "restart": restart,
            "gateway": self._gateway_health(),
        }

    def _validate_project(self, require_index: bool) -> dict:
        if self.config is None:
            return {
                "ok": False,
                "configured": False,
                "cwd": None,
                "cwd_exists": False,
                "index_exists": False,
                "error": "CodeGraph is not configured.",
            }

        cwd_path = Path(self.config.cwd)
        cwd_exists = cwd_path.is_dir()
        index_exists = (cwd_path / ".codegraph").is_dir() if cwd_exists else False
        result = {
            "ok": True,
            "configured": True,
            "cwd": str(cwd_path),
            "cwd_exists": cwd_exists,
            "index_exists": index_exists,
        }
        if not cwd_exists:
            result.update({"ok": False, "error": "CodeGraph project directory does not exist."})
        elif require_index and not index_exists:
            result.update({"ok": False, "error": "CodeGraph index is not initialized. Run codegraph_init first."})
        return result

    def _run_cli(self, command_name: str, *extra_args: str) -> dict:
        if self.config is None:
            return {"returncode": 1, "stdout": "", "stderr": "CodeGraph is not configured.", "command": []}

        command = [_resolve_command(self.config.command), *derive_lifecycle_args(self.config, command_name, *extra_args)]
        completed = self.command_runner(command, self.config.cwd)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": _trim_output(completed.stdout),
            "stderr": _trim_output(completed.stderr),
        }

    def _restart_client(self) -> dict:
        if self.codegraph_client is None:
            return {"ok": False, "error": "CodeGraph MCP client is not available."}
        ok = bool(self.codegraph_client.restart())
        return {"ok": ok, "tool_names": list(getattr(self.codegraph_client, "tool_names", []))}

    def _gateway_health(self) -> dict:
        if self.codegraph_client is None:
            return {
                "available": False,
                "process_running": False,
                "tool_names": [],
                "last_startup_error": "CodeGraph MCP client is not available.",
            }
        if hasattr(self.codegraph_client, "health"):
            return self.codegraph_client.health()
        return {
            "available": bool(getattr(self.codegraph_client, "available", False)),
            "process_running": bool(getattr(self.codegraph_client, "is_running", lambda: False)()),
            "tool_names": list(getattr(self.codegraph_client, "tool_names", [])),
            "last_startup_error": None,
        }


def derive_lifecycle_args(config: CodeGraphConfig, command_name: str, *extra_args: str) -> list[str]:
    serve_index = _find_serve_index(config.args)
    prefix = config.args[:serve_index]
    return [*prefix, command_name, *extra_args]


def _default_command_runner(command: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def _find_serve_index(args: list[str]) -> int:
    for index, value in enumerate(args):
        if value == "serve":
            return index
    return len(args)


def _resolve_command(command: str) -> str:
    if os.name == "nt" and command.lower() == "npx":
        return "npx.cmd"
    return command


def _trim_output(value: str | None, limit: int = 4000) -> str:
    if not value:
        return ""
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def _stdout_suggests_changes(stdout: str) -> bool:
    lowered = stdout.lower()
    unchanged_markers = ["up to date", "no changes", "unchanged", "0 changed"]
    if any(marker in lowered for marker in unchanged_markers):
        return False
    return bool(stdout.strip())
