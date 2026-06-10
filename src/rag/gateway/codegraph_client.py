from __future__ import annotations

import json
import queue
import subprocess
import threading
from typing import Callable, TextIO

from rag.gateway.config import CodeGraphConfig


ProcessFactory = Callable[[list[str], str], subprocess.Popen]


class CodeGraphRequestError(Exception):
    pass


def _default_process_factory(command: list[str], cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
    )


class CodeGraphClient:
    def __init__(
        self,
        config: CodeGraphConfig | None,
        process_factory: ProcessFactory | None = None,
        response_timeout_seconds: float = 5.0,
    ):
        self.config = config
        self.process_factory = process_factory or _default_process_factory
        self.response_timeout_seconds = response_timeout_seconds
        self.process = None
        self.available = False
        self.tools: list[dict] = []
        self.tool_names: list[str] = []
        self._next_id = 1
        self._responses: queue.Queue[dict | Exception] = queue.Queue()
        self._pending_responses: dict[int, dict] = {}

    def start(self) -> bool:
        if self.config is None:
            return False

        try:
            command = [self.config.command, *self.config.args]
            self.process = self.process_factory(command, self.config.cwd)
            self._start_reader_thread()
            self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "doc-rag-gateway", "version": "0.1.0"},
                },
            )
            self._notify("initialized", {})
            tools_result = self._request("tools/list", {})
        except Exception:
            self._mark_unavailable()
            self._shutdown_process()
            return False

        self.tools = [item for item in tools_result.get("tools", []) if isinstance(item, dict) and "name" in item]
        self.tool_names = [item["name"] for item in self.tools]
        self.available = True
        return True

    def call_tool(self, name: str, arguments: dict) -> dict:
        if not self.available or self.process is None or self.process.poll() is not None:
            self._mark_unavailable()
            return {"error": "CodeGraph unavailable"}

        try:
            return self._request("tools/call", {"name": name, "arguments": arguments})
        except CodeGraphRequestError as exc:
            return {"error": str(exc)}
        except Exception:
            self._mark_unavailable()
            self._shutdown_process()
            return {"error": "CodeGraph unavailable"}

    def _request(self, method: str, params: dict) -> dict:
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        response = self._read_response(request_id)
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise CodeGraphRequestError(error.get("message", "CodeGraph request failed"))
            raise CodeGraphRequestError("CodeGraph request failed")
        return response.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, payload: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("CodeGraph process is not available")

        stdin: TextIO = self.process.stdin
        stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdin.flush()

    def _read_response(self, request_id: int) -> dict:
        if request_id in self._pending_responses:
            return self._pending_responses.pop(request_id)

        while True:
            try:
                item = self._responses.get(timeout=self.response_timeout_seconds)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for CodeGraph response") from exc

            if isinstance(item, Exception):
                raise item
            if not isinstance(item, dict):
                continue

            response_id = item.get("id")
            if response_id == request_id:
                return item
            if isinstance(response_id, int):
                self._pending_responses[response_id] = item

    def _start_reader_thread(self) -> None:
        thread = threading.Thread(target=self._reader_loop, daemon=True)
        thread.start()

    def _reader_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            self._responses.put(RuntimeError("CodeGraph stdout is not available"))
            return

        stdout: TextIO = self.process.stdout
        while True:
            line = stdout.readline()
            if not line:
                self._responses.put(RuntimeError("CodeGraph closed stdout"))
                return
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError as exc:
                self._responses.put(exc)

    def _mark_unavailable(self) -> None:
        self.available = False
        self.tools = []
        self.tool_names = []
        self._pending_responses.clear()

    def _shutdown_process(self) -> None:
        process = self.process
        if process is None:
            return

        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=0.5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
