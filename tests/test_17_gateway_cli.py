from __future__ import annotations

import pytest


def test_gateway_command_dispatches_to_gateway_server_without_loading_doc_config(monkeypatch):
    from rag import cli

    called: dict[str, bool] = {"gateway_main": False}

    def fake_load_config():
        raise AssertionError("load_config should not run for gateway mode")

    def fake_gateway_main():
        called["gateway_main"] = True

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr("rag.gateway.server.main", fake_gateway_main)
    monkeypatch.setattr(cli.sys, "argv", ["rag", "gateway"])

    cli.main()

    assert called["gateway_main"] is True


def test_no_command_prints_help_and_exits_1(monkeypatch, capsys):
    from rag import cli

    monkeypatch.setattr(cli.sys, "argv", ["rag"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "usage: rag" in captured.out


def test_existing_source_list_command_still_uses_normal_cli_path(monkeypatch, capsys):
    from rag import cli
    import rag.source_manager as source_manager

    config = object()

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(
        source_manager,
        "list_sources",
        lambda incoming_config: [
            {"label": "sdk", "path": "docs/sdk"},
        ] if incoming_config is config else [],
    )
    monkeypatch.setattr(source_manager, "add_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(source_manager, "remove_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.sys, "argv", ["rag", "source", "list"])

    cli.main()

    captured = capsys.readouterr()
    assert "sdk" in captured.out
    assert "docs/sdk" in captured.out


def test_adapter_command_dispatches_to_adapter_without_loading_doc_config(monkeypatch):
    from rag import cli

    called = {"adapter": False}

    monkeypatch.setattr(cli, "load_config", lambda: (_ for _ in ()).throw(AssertionError("load_config should not run")))
    monkeypatch.setattr("rag.adapter.main", lambda: called.__setitem__("adapter", True))
    monkeypatch.setattr(cli.sys, "argv", ["rag", "adapter"])

    cli.main()

    assert called["adapter"] is True
