from __future__ import annotations

from typing import Any, Callable

from rag.daemon.config import normalize_gateway_config_path
from rag.gateway.codegraph_client import CodeGraphClient
from rag.gateway.codegraph_lifecycle import CodeGraphLifecycle
from rag.gateway.config import CodeGraphConfig, GatewayConfig, load_gateway_config
from rag.gateway.doc_backend import DocRagBackend
from rag.gateway.server import build_tools_list
from rag.gateway.tools import GatewayTools


DocBackendFactory = Callable[[str | None], Any]
CodeGraphClientFactory = Callable[[CodeGraphConfig | None], Any | None]
LifecycleFactory = Callable[[CodeGraphConfig, Any], Any]


class GatewayToolService:
    def __init__(
        self,
        config: GatewayConfig,
        doc_backend_factory: DocBackendFactory = DocRagBackend,
        codegraph_client_factory: CodeGraphClientFactory = CodeGraphClient,
        lifecycle_factory: LifecycleFactory = CodeGraphLifecycle,
        config_loader: Callable[[], GatewayConfig] | None = None,
        gateway_config_path: str | None = None,
    ) -> None:
        self._gateway_config_path = gateway_config_path
        self._doc_backend_factory = doc_backend_factory
        self._codegraph_client_factory = codegraph_client_factory
        self._lifecycle_factory = lifecycle_factory
        self._config_loader = config_loader or (lambda: load_gateway_config(gateway_config_path))
        self.config = config
        self.doc_backend = doc_backend_factory(config.doc_rag_config_path)
        self.codegraph_client = None
        self.codegraph_lifecycle = None
        self.last_error: str | None = None

        if config.codegraph is not None:
            self.codegraph_client = codegraph_client_factory(config.codegraph)
            if self.codegraph_client is not None:
                try:
                    self.codegraph_client.start()
                except Exception as exc:
                    self.last_error = str(exc)
                self.codegraph_lifecycle = lifecycle_factory(config.codegraph, self.codegraph_client)

        self.tools = GatewayTools(self.doc_backend, self.codegraph_client, self.codegraph_lifecycle)

    @classmethod
    def from_config_path(cls, config_path: str | None = None) -> "GatewayToolService":
        resolved = normalize_gateway_config_path(config_path)
        config = load_gateway_config(resolved)
        return cls(config, gateway_config_path=resolved)

    def list_tools(self) -> list[dict]:
        codegraph_tools = []
        if self.codegraph_client is not None:
            codegraph_tools = getattr(self.codegraph_client, "tools", [])
        return build_tools_list(codegraph_tools, self.codegraph_lifecycle is not None)

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        return self.tools.call_tool(name, arguments or {})

    def health(self) -> dict:
        return {
            "ok": self.last_error is None,
            "doc_rag": self._doc_health(),
            "codegraph": self._codegraph_health(),
            "tool_count": len(self.list_tools()),
            "last_error": self.last_error,
        }

    def reload(self) -> dict:
        """Reload gateway config and rebuild all service internals.

        Re-reads gateway.yaml, rebuilds DocRagBackend (picking up config.yaml
        changes), and, when CodeGraph is configured, shuts down the old
        CodeGraph subprocess and creates a fresh one.  The daemon identity
        (port, token, runtime metadata) stays the same.

        Build-then-swap with two-tier failure handling:

        * Doc-rag reload failure -- abort immediately, preserve old service
          state intact, return ``{"ok": false, "error": "..."}``.
        * CodeGraph start failure -- accept graceful degradation.  Swap to the
          new doc backend + failed CodeGraph client, record the error in
          ``last_error`` and return it as a warning.  Doc-rag tools remain
          available; CodeGraph tools return structured errors until the next
          successful reload or restart.
        """
        warnings: list[str] = []
        new_last_error: str | None = None

        # Phase 1: build everything new
        new_config = self._config_loader()

        try:
            new_doc_backend = self._doc_backend_factory(new_config.doc_rag_config_path)
        except Exception as exc:
            return {"ok": False, "error": f"doc-rag reload failed: {exc}"}

        new_codegraph_client = None
        new_codegraph_lifecycle = None
        if new_config.codegraph is not None:
            new_codegraph_client = self._codegraph_client_factory(new_config.codegraph)
            if new_codegraph_client is not None:
                try:
                    new_codegraph_client.start()
                except Exception as exc:
                    new_last_error = str(exc)
                    warnings.append(f"CodeGraph start during reload: {exc}")
                new_codegraph_lifecycle = self._lifecycle_factory(new_config.codegraph, new_codegraph_client)

        new_tools = GatewayTools(new_doc_backend, new_codegraph_client, new_codegraph_lifecycle)

        # Phase 2: tear down old CodeGraph
        old_client = self.codegraph_client
        if old_client is not None:
            shutdown = getattr(old_client, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    warnings.append(f"CodeGraph shutdown during reload: {exc}")

        # Phase 3: atomic swap
        self.config = new_config
        self.doc_backend = new_doc_backend
        self.codegraph_client = new_codegraph_client
        self.codegraph_lifecycle = new_codegraph_lifecycle
        self.tools = new_tools
        self.last_error = new_last_error

        result: dict = {"ok": self.last_error is None, "tool_count": len(self.list_tools())}
        if warnings:
            result["warnings"] = warnings
        return result

    def shutdown(self) -> None:
        shutdown = getattr(self.codegraph_client, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _doc_health(self) -> dict:
        health = getattr(self.doc_backend, "health", None)
        if callable(health):
            return health()
        return {"ok": True}

    def _codegraph_health(self) -> dict:
        if self.config.codegraph is None:
            return {"state": "not_configured", "available": False}
        if self.codegraph_client is None:
            return {"state": "configured_unavailable", "available": False}

        health = getattr(self.codegraph_client, "health", lambda: {})()
        available = bool(health.get("available", getattr(self.codegraph_client, "available", False)))
        state = "configured_running" if available else "configured_unavailable"
        return {"state": state, **health}
