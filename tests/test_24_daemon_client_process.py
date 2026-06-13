from __future__ import annotations

from pathlib import Path


def test_daemon_client_builds_authorized_requests(monkeypatch):
    from rag.daemon.client import DaemonClient
    from rag.daemon.runtime import RuntimeMetadata

    captured = {}

    def fake_request_json(method, url, token, payload=None):
        captured["method"] = method
        captured["url"] = url
        captured["token"] = token
        captured["payload"] = payload
        return {"ok": True, "result": {"value": 1}}

    metadata = RuntimeMetadata(
        pid=1,
        host="127.0.0.1",
        port=4321,
        token="secret",
        gateway_config_path="gateway.yaml",
        identity="abc",
        started_at="2026-06-13T00:00:00Z",
        log_path="output/runtime/daemon-abc.log",
    )
    client = DaemonClient(metadata, request_json=fake_request_json)

    assert client.call_tool("search_docs", {"query": "render"}) == {"value": 1}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:4321/tools/call"
    assert captured["token"] == "secret"


def test_ensure_daemon_autostarts_when_metadata_missing(tmp_path: Path):
    from rag.daemon.process import ensure_daemon
    from rag.daemon.runtime import RuntimeMetadata, write_metadata

    calls = []
    runtime_path = tmp_path / "runtime.json"

    def fake_start(gateway_config_path, runtime_path, log_path):
        # Simulate the daemon writing metadata after start.
        write_metadata(runtime_path, RuntimeMetadata(
            pid=9999, host="127.0.0.1", port=5678, token="t",
            gateway_config_path=gateway_config_path, identity="id",
            started_at="2026-06-13T00:00:00Z", log_path=str(log_path),
        ))
        calls.append(True)
        return True

    metadata = ensure_daemon(
        gateway_config_path=str(tmp_path / "gateway.yaml"),
        runtime_path=runtime_path,
        autostart=True,
        start_process=fake_start,
        wait_seconds=0.01,
    )

    assert metadata is not None
    assert metadata.port == 5678
    assert len(calls) == 1


def test_ensure_daemon_does_not_autostart_when_disabled(tmp_path: Path):
    from rag.daemon.process import ensure_daemon

    calls = []

    metadata = ensure_daemon(
        gateway_config_path=str(tmp_path / "gateway.yaml"),
        runtime_path=tmp_path / "runtime.json",
        autostart=False,
        start_process=lambda gateway_config_path, runtime_path, log_path: calls.append(True) or True,
        health_check=lambda metadata: False,
        wait_seconds=0.01,
    )

    assert metadata is None
    assert calls == []


def test_daemon_client_reload_sends_post_reload():
    from rag.daemon.client import DaemonClient
    from rag.daemon.runtime import RuntimeMetadata

    captured = {}

    def fake_request_json(method, url, token, payload=None):
        captured.update({"method": method, "url": url, "token": token, "payload": payload})
        return {"ok": True, "reloaded": True}

    metadata = RuntimeMetadata(
        pid=1, host="127.0.0.1", port=4321, token="secret",
        gateway_config_path="gateway.yaml", identity="abc",
        started_at="2026-06-13T00:00:00Z", log_path="output/runtime/daemon-abc.log",
    )
    client = DaemonClient(metadata, request_json=fake_request_json)
    result = client.reload()

    assert result["reloaded"] is True
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:4321/reload"
