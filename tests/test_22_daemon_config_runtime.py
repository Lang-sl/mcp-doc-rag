from __future__ import annotations

import json
from pathlib import Path


def test_gateway_config_identity_is_stable(tmp_path: Path):
    from rag.daemon.config import gateway_config_identity

    gateway = tmp_path / "gateway.yaml"
    gateway.write_text("doc_rag:\n  config_path: config.yaml\n", encoding="utf-8")

    assert gateway_config_identity(str(gateway)) == gateway_config_identity(str(gateway))


def test_default_runtime_path_uses_output_runtime(tmp_path: Path):
    from rag.daemon.config import default_runtime_path

    gateway = tmp_path / "gateway.yaml"
    gateway.write_text("", encoding="utf-8")

    runtime_path = default_runtime_path(str(gateway), None, project_root=tmp_path)

    assert runtime_path.parent == tmp_path / "output" / "runtime"
    assert runtime_path.name.startswith("daemon-")
    assert runtime_path.suffix == ".json"


def test_default_log_path_uses_output_runtime(tmp_path: Path):
    from rag.daemon.config import default_log_path

    gateway = tmp_path / "gateway.yaml"
    gateway.write_text("", encoding="utf-8")

    log_path = default_log_path(str(gateway), None, project_root=tmp_path)

    assert log_path.parent == tmp_path / "output" / "runtime"
    assert log_path.name.startswith("daemon-")
    assert log_path.suffix == ".log"


def test_runtime_metadata_round_trip_hides_token(tmp_path: Path):
    from rag.daemon.runtime import RuntimeMetadata, format_status, read_metadata, write_metadata

    path = tmp_path / "runtime.json"
    metadata = RuntimeMetadata(
        pid=123,
        host="127.0.0.1",
        port=4567,
        token="secret-token",
        gateway_config_path="gateway.yaml",
        identity="abc",
        started_at="2026-06-13T00:00:00Z",
    )

    write_metadata(path, metadata)
    loaded = read_metadata(path)

    assert loaded == metadata
    assert "secret-token" not in format_status(loaded)


def test_malformed_runtime_metadata_returns_none(tmp_path: Path):
    from rag.daemon.runtime import read_metadata

    path = tmp_path / "runtime.json"
    path.write_text("{bad json", encoding="utf-8")

    assert read_metadata(path) is None
