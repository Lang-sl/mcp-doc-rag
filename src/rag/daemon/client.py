from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from rag.daemon.runtime import RuntimeMetadata


class DaemonConnectionError(Exception):
    pass


RequestJson = Callable[[str, str, str, dict | None], dict]


class DaemonClient:
    def __init__(self, metadata: RuntimeMetadata, request_json: RequestJson | None = None) -> None:
        self.metadata = metadata
        self._request_json = request_json or request_json_stdlib

    def health(self) -> dict:
        return self._request_json("GET", self._url("/health"), self.metadata.token, None)

    def list_tools(self) -> list[dict]:
        return self._request_json("GET", self._url("/tools"), self.metadata.token, None).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> Any:
        response = self._request_json(
            "POST",
            self._url("/tools/call"),
            self.metadata.token,
            {"name": name, "arguments": arguments},
        )
        if not response.get("ok", False):
            error = response.get("error", {})
            raise RuntimeError(error.get("message", "Daemon tool call failed"))
        return response.get("result")

    def reload(self) -> dict:
        return self._request_json("POST", self._url("/reload"), self.metadata.token, {})

    def shutdown(self) -> dict:
        return self._request_json("POST", self._url("/shutdown"), self.metadata.token, {})

    def _url(self, path: str) -> str:
        return f"http://{self.metadata.host}:{self.metadata.port}{path}"


def request_json_stdlib(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise DaemonConnectionError(str(exc)) from exc
