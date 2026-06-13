from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class GatewayDaemonHttpServer:
    def __init__(self, host: str, port: int, token: str, service: Any) -> None:
        self.host = host
        self.token = token
        self.service = service
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None

    def start_in_thread(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        shutdown = getattr(self.service, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _handler_class(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if not self._authorized():
                    self._send(401, {"ok": False, "error": {"code": "unauthorized", "message": "Unauthorized"}})
                    return
                if self.path == "/health":
                    self._send(200, outer.service.health())
                    return
                if self.path == "/tools":
                    self._send(200, {"tools": outer.service.list_tools()})
                    return
                self._send(404, {"ok": False, "error": {"code": "not_found", "message": self.path}})

            def do_POST(self) -> None:
                if not self._authorized():
                    self._send(401, {"ok": False, "error": {"code": "unauthorized", "message": "Unauthorized"}})
                    return
                payload = self._read_json()
                if self.path == "/tools/call":
                    self._handle_tool_call(payload)
                    return
                if self.path == "/reload":
                    self._send(200, outer.service.reload())
                    return
                if self.path == "/shutdown":
                    self._send(200, {"ok": True})
                    threading.Thread(target=outer.stop, daemon=True).start()
                    return
                self._send(404, {"ok": False, "error": {"code": "not_found", "message": self.path}})

            def log_message(self, format: str, *args: object) -> None:
                return

            def _authorized(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {outer.token}"

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    return {}

            def _handle_tool_call(self, payload: dict) -> None:
                name = payload.get("name", "")
                arguments = payload.get("arguments", {})
                if not isinstance(arguments, dict):
                    self._send(200, {"ok": False, "error": {"code": "invalid_params", "message": "arguments must be an object"}})
                    return
                try:
                    result = outer.service.call_tool(name, arguments)
                except KeyError:
                    self._send(200, {"ok": False, "error": {"code": "tool_not_found", "message": f"Tool not found: {name}"}})
                    return
                except Exception as exc:
                    self._send(200, {"ok": False, "error": {"code": "tool_error", "message": str(exc)}})
                    return
                self._send(200, {"ok": True, "result": result})

            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
