from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from rag.gateway.config import CodeGraphConfig
from rag.gateway.codegraph_lifecycle import CodeGraphLifecycle, derive_lifecycle_args


def _expected_npx() -> str:
    """Return the expected npx command for the current platform."""
    return "npx.cmd" if os.name == "nt" else "npx"


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeCodeGraphClient:
    def __init__(self, available: bool = False, running: bool = False) -> None:
        self.available = available
        self.tool_names = ["codegraph_search"] if available else []
        self.running = running
        self.restart_count = 0

    def is_running(self) -> bool:
        return self.running

    def health(self) -> dict:
        return {
            "available": self.available,
            "process_running": self.running,
            "tool_names": self.tool_names,
            "last_startup_error": None,
        }

    def restart(self) -> bool:
        self.restart_count += 1
        self.available = True
        self.running = True
        self.tool_names = ["codegraph_search"]
        return True


def test_derive_lifecycle_args_reuses_pinned_codegraph_package() -> None:
    config = CodeGraphConfig(args=["-y", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"])
    assert derive_lifecycle_args(config, "status") == ["-y", "@colbymchenry/codegraph@0.9.9", "status"]
    assert derive_lifecycle_args(config, "init", "-i", "project") == [
        "-y",
        "@colbymchenry/codegraph@0.9.9",
        "init",
        "-i",
        "project",
    ]


def test_derive_lifecycle_args_keeps_custom_prefix_before_serve() -> None:
    config = CodeGraphConfig(args=["--yes", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"])
    assert derive_lifecycle_args(config, "sync") == ["--yes", "@colbymchenry/codegraph@0.9.9", "sync"]


def test_index_status_reports_unconfigured_codegraph() -> None:
    lifecycle = CodeGraphLifecycle(None, None)

    result = lifecycle.index_status()

    assert result["configured"] is False
    assert result["ok"] is False
    assert result["error"] == "CodeGraph is not configured."


def test_index_status_reports_missing_project_directory(tmp_path: Path) -> None:
    config = CodeGraphConfig(cwd=str(tmp_path / "missing"))
    lifecycle = CodeGraphLifecycle(config, FakeCodeGraphClient())

    result = lifecycle.index_status()

    assert result["configured"] is True
    assert result["cwd_exists"] is False
    assert result["index_exists"] is False
    assert result["ok"] is False


def test_index_status_runs_status_when_index_exists(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".codegraph").mkdir(parents=True)
    calls: list[tuple[list[str], str]] = []
    npx = _expected_npx()

    def runner(command: list[str], cwd: str) -> FakeCompletedProcess:
        calls.append((command, cwd))
        return FakeCompletedProcess(stdout="Index up to date")

    lifecycle = CodeGraphLifecycle(CodeGraphConfig(cwd=str(project)), FakeCodeGraphClient(True, True), runner)

    result = lifecycle.index_status()

    assert result["ok"] is True
    assert result["index_exists"] is True
    assert result["cli"]["stdout"] == "Index up to date"
    assert calls == [([npx, "-y", "@colbymchenry/codegraph@0.9.9", "status"], str(project))]


def test_init_runs_only_when_index_missing_and_restarts_client(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    client = FakeCodeGraphClient()
    calls: list[list[str]] = []
    npx = _expected_npx()

    def runner(command: list[str], cwd: str) -> FakeCompletedProcess:
        calls.append(command)
        (project / ".codegraph").mkdir()
        return FakeCompletedProcess(stdout="initialized")

    lifecycle = CodeGraphLifecycle(CodeGraphConfig(cwd=str(project)), client, runner)

    result = lifecycle.init()

    assert result["ok"] is True
    assert result["action"] == "initialized"
    assert client.restart_count == 1
    assert calls == [[npx, "-y", "@colbymchenry/codegraph@0.9.9", "init", "-i", str(project)]]


def test_reindex_requires_existing_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    lifecycle = CodeGraphLifecycle(CodeGraphConfig(cwd=str(project)), FakeCodeGraphClient())

    result = lifecycle.reindex()

    assert result["ok"] is False
    assert result["error"] == "CodeGraph index is not initialized. Run codegraph_init first."


def test_reindex_force_restarts_client_after_success(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".codegraph").mkdir(parents=True)
    client = FakeCodeGraphClient(True, True)
    calls: list[list[str]] = []
    npx = _expected_npx()

    def runner(command: list[str], cwd: str) -> FakeCompletedProcess:
        calls.append(command)
        return FakeCompletedProcess(stdout="indexed")

    lifecycle = CodeGraphLifecycle(CodeGraphConfig(cwd=str(project)), client, runner)

    result = lifecycle.reindex(force=True)

    assert result["ok"] is True
    assert result["action"] == "reindexed"
    assert client.restart_count == 1
    assert calls == [[npx, "-y", "@colbymchenry/codegraph@0.9.9", "index", "--force"]]
