from __future__ import annotations

import json
import urllib.error
import urllib.request


class FakeService:
    def __init__(self):
        self.shutdown_called = False

    def health(self):
        return {"ok": True, "tool_count": 1}

    def list_tools(self):
        return [{"name": "search_docs"}]

    def call_tool(self, name, arguments):
        if name == "missing":
            raise KeyError(name)
        return {"name": name, "arguments": arguments}

    def reload(self):
        return {"ok": True, "reloaded": True}

    def shutdown(self):
        self.shutdown_called = True


def request_json(url, token, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def test_daemon_http_requires_token():
    from rag.daemon.http_server import GatewayDaemonHttpServer

    server = GatewayDaemonHttpServer("127.0.0.1", 0, "secret", FakeService())
    server.start_in_thread()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=2):
            assert False, "request should fail without token"
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    finally:
        server.stop()


def test_daemon_http_serves_health_tools_and_calls():
    from rag.daemon.http_server import GatewayDaemonHttpServer

    server = GatewayDaemonHttpServer("127.0.0.1", 0, "secret", FakeService())
    server.start_in_thread()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert request_json(base + "/health", "secret")["ok"] is True
        assert request_json(base + "/tools", "secret")["tools"] == [{"name": "search_docs"}]
        result = request_json(
            base + "/tools/call",
            "secret",
            method="POST",
            payload={"name": "search_docs", "arguments": {"query": "render"}},
        )
        assert result == {"ok": True, "result": {"name": "search_docs", "arguments": {"query": "render"}}}
    finally:
        server.stop()


def test_daemon_http_reload_calls_service_reload():
    from rag.daemon.http_server import GatewayDaemonHttpServer

    class ReloadService:
        def health(self):
            return {"ok": True}
        def list_tools(self):
            return []
        def call_tool(self, name, arguments):
            return {}
        def reload(self):
            return {"ok": True, "reloaded": True}
        def shutdown(self):
            pass

    server = GatewayDaemonHttpServer("127.0.0.1", 0, "secret", ReloadService())
    server.start_in_thread()
    try:
        base = f"http://127.0.0.1:{server.port}"
        result = request_json(base + "/reload", "secret", method="POST", payload={})
        assert result == {"ok": True, "reloaded": True}
    finally:
        server.stop()


def test_daemon_http_maps_missing_tool_to_structured_error():
    from rag.daemon.http_server import GatewayDaemonHttpServer

    server = GatewayDaemonHttpServer("127.0.0.1", 0, "secret", FakeService())
    server.start_in_thread()
    try:
        base = f"http://127.0.0.1:{server.port}"
        result = request_json(
            base + "/tools/call",
            "secret",
            method="POST",
            payload={"name": "missing", "arguments": {}},
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "tool_not_found"
    finally:
        server.stop()
