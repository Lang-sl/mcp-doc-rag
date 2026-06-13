from __future__ import annotations

from pathlib import Path

import yaml


def test_configure_gateway_can_disable_codegraph_and_enable_daemon(monkeypatch, tmp_path: Path):
    import setup_config

    template = tmp_path / "gateway.example.yaml"
    template.write_text("doc_rag:\n  config_path: config.yaml\n", encoding="utf-8")
    target = tmp_path / "gateway.yaml"
    config_target = tmp_path / "config.yaml"
    config_target.write_text("doc_sources: {}\n", encoding="utf-8")

    # First "y" = create gateway, second "n" = skip CodeGraph
    answers = iter(["y", "n"])
    monkeypatch.setattr(setup_config, "GATEWAY_TEMPLATE", template)
    monkeypatch.setattr(setup_config, "GATEWAY_TARGET", target)
    monkeypatch.setattr(setup_config, "TARGET", config_target)
    monkeypatch.setattr(setup_config, "ask", lambda prompt, default="": next(answers, default))

    assert setup_config.configure_gateway() is True

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "codegraph" not in data
    assert data["daemon"]["autostart"] is True
    assert data["daemon"]["host"] == "127.0.0.1"
    assert data["daemon"]["port"] == 0


def test_configure_gateway_can_enable_codegraph(monkeypatch, tmp_path: Path):
    import setup_config

    code_root = tmp_path / "code"
    code_root.mkdir()
    template = tmp_path / "gateway.example.yaml"
    template.write_text("doc_rag:\n  config_path: config.yaml\n", encoding="utf-8")
    target = tmp_path / "gateway.yaml"
    config_target = tmp_path / "config.yaml"
    config_target.write_text("doc_sources: {}\n", encoding="utf-8")

    # First "y" = create gateway, second "y" = enable CodeGraph, third = code project path
    answers = iter(["y", "y", str(code_root)])
    monkeypatch.setattr(setup_config, "GATEWAY_TEMPLATE", template)
    monkeypatch.setattr(setup_config, "GATEWAY_TARGET", target)
    monkeypatch.setattr(setup_config, "TARGET", config_target)
    monkeypatch.setattr(setup_config, "ask", lambda prompt, default="": next(answers, default))

    assert setup_config.configure_gateway() is True

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["codegraph"]["cwd"] == str(code_root)
    assert data["codegraph"]["args"] == ["-y", "@colbymchenry/codegraph@0.9.9", "serve", "--mcp"]
    assert data["daemon"]["autostart"] is True


def test_configure_gateway_can_skip_gateway_entirely(monkeypatch, tmp_path: Path):
    import setup_config

    template = tmp_path / "gateway.example.yaml"
    template.write_text("doc_rag:\n  config_path: config.yaml\n", encoding="utf-8")
    target = tmp_path / "gateway.yaml"
    config_target = tmp_path / "config.yaml"
    config_target.write_text("doc_sources: {}\n", encoding="utf-8")

    # "n" = skip gateway creation
    answers = iter(["n"])
    monkeypatch.setattr(setup_config, "GATEWAY_TEMPLATE", template)
    monkeypatch.setattr(setup_config, "GATEWAY_TARGET", target)
    monkeypatch.setattr(setup_config, "TARGET", config_target)
    monkeypatch.setattr(setup_config, "ask", lambda prompt, default="": next(answers, default))

    assert setup_config.configure_gateway() is False

    # gateway.yaml should not be created
    assert not target.exists()
