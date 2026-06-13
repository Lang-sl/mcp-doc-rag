from __future__ import annotations

import json
from io import StringIO
from pathlib import Path


def test_adapter_handles_initialize_and_tools_list(monkeypatch, tmp_path: Path):
    from rag import adapter
    from rag.daemon.runtime import RuntimeMetadata

    metadata = RuntimeMetadata(
        pid=1,
        host="127.0.0.1",
        port=1234,
        token="secret",
        gateway_config_path=str(tmp_path / "gateway.yaml"),
        identity="abc",
        started_at="2026-06-13T00:00:00Z",
        log_path=str(tmp_path / "output" / "runtime" / "daemon-abc.log"),
    )

    class FakeClient:
        def __init__(self, incoming_metadata):
            assert incoming_metadata == metadata

        def list_tools(self):
            return [{"name": "search_docs"}]

        def call_tool(self, name, arguments):
            return {"name": name, "arguments": arguments}

    monkeypatch.setattr(adapter, "resolve_adapter_metadata", lambda config_path=None: metadata)
    monkeypatch.setattr(adapter, "DaemonClient", FakeClient)

    stdin = StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n"
    )
    stdout = StringIO()

    adapter.run_stdio(stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["result"]["serverInfo"]["name"] == "mcp-doc-rag-gateway-adapter"
    assert responses[1]["result"]["tools"] == [{"name": "search_docs"}]


def test_adapter_tools_call_returns_mcp_text(monkeypatch, tmp_path: Path):
    from rag import adapter
    from rag.daemon.runtime import RuntimeMetadata

    metadata = RuntimeMetadata(
        pid=1,
        host="127.0.0.1",
        port=1234,
        token="secret",
        gateway_config_path=str(tmp_path / "gateway.yaml"),
        identity="abc",
        started_at="2026-06-13T00:00:00Z",
        log_path=str(tmp_path / "output" / "runtime" / "daemon-abc.log"),
    )

    class FakeClient:
        def __init__(self, incoming_metadata):
            pass

        def list_tools(self):
            return []

        def call_tool(self, name, arguments):
            return {"ok": True, "name": name, "arguments": arguments}

    monkeypatch.setattr(adapter, "resolve_adapter_metadata", lambda config_path=None: metadata)
    monkeypatch.setattr(adapter, "DaemonClient", FakeClient)

    stdin = StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search_docs", "arguments": {"query": "render"}},
            }
        ) + "\n"
    )
    stdout = StringIO()

    adapter.run_stdio(stdin=stdin, stdout=stdout)

    response = json.loads(stdout.getvalue())
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload == {"ok": True, "name": "search_docs", "arguments": {"query": "render"}}
