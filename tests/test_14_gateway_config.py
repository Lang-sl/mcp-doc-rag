from __future__ import annotations

from pathlib import Path

from rag.gateway.config import CodeGraphConfig, GatewayConfig, load_gateway_config


def test_missing_gateway_config_returns_doc_only_defaults(tmp_path: Path):
    path = tmp_path / "missing.yaml"

    config = load_gateway_config(str(path))

    assert isinstance(config, GatewayConfig)
    assert config.codegraph is None
    assert config.doc_rag_config_path is None


def test_non_dict_gateway_config_returns_doc_only_defaults(tmp_path: Path):
    path = tmp_path / "gateway.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    config = load_gateway_config(str(path))

    assert config == GatewayConfig(codegraph=None, doc_rag_config_path=None)


def test_default_codegraph_config_uses_latest_optional_npm_package():
    assert CodeGraphConfig().args == ["-y", "@colbymchenry/codegraph@latest", "serve", "--mcp"]


def test_load_gateway_config_reads_codegraph_and_doc_config(tmp_path: Path):
    path = tmp_path / "gateway.yaml"
    path.write_text(
        "\n".join(
            [
                "codegraph:",
                '  command: "npx"',
                '  args: ["@colbymchenry/codegraph", "serve", "--mcp"]',
                '  cwd: "example-project"',
                "doc_rag:",
                '  config_path: "config.example.yaml"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_gateway_config(str(path))

    assert config.codegraph == CodeGraphConfig(
        command="npx",
        args=["@colbymchenry/codegraph", "serve", "--mcp"],
        cwd="example-project",
    )
    assert config.doc_rag_config_path == "config.example.yaml"


def test_load_gateway_config_uses_environment_path(monkeypatch, tmp_path: Path):
    path = tmp_path / "gateway.yaml"
    path.write_text("codegraph:\n  cwd: example-project\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_CONFIG_PATH", str(path))

    config = load_gateway_config()

    assert config.codegraph == CodeGraphConfig(cwd="example-project")


def test_malformed_gateway_config_values_fall_back_to_safe_defaults(tmp_path: Path):
    path = tmp_path / "gateway.yaml"
    path.write_text(
        "\n".join(
            [
                "codegraph:",
                "  command: false",
                "  args: [serve, true, 1]",
                "  cwd:",
                "    - bad-path",
                "doc_rag:",
                "  config_path: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_gateway_config(str(path))

    assert config.codegraph == CodeGraphConfig()
    assert config.doc_rag_config_path is None
