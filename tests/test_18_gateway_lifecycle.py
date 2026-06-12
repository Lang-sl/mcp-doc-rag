from __future__ import annotations

from rag.gateway.config import CodeGraphConfig
from rag.gateway.codegraph_lifecycle import derive_lifecycle_args


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
