from __future__ import annotations

from rag.gateway.config import CodeGraphConfig


def _find_serve_index(args: list[str]) -> int:
    for index, value in enumerate(args):
        if value == "serve":
            return index
    return len(args)


def derive_lifecycle_args(config: CodeGraphConfig, command_name: str, *extra_args: str) -> list[str]:
    serve_index = _find_serve_index(config.args)
    prefix = config.args[:serve_index]
    return [*prefix, command_name, *extra_args]
