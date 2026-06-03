"""Shared fixtures and utilities for mcp-doc-rag tests.

Each test file represents one verification stage. Run stages in order:
  1 (no deps) → 7 (needs real files) → 8 (needs Ollama) → 9 (needs index) → 10 (full E2E)
"""

from __future__ import annotations

import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Ollama availability detection
# ---------------------------------------------------------------------------

_ollama_available: bool | None = None


def is_ollama_available() -> bool:
    """Return True if Ollama is running and nomic-embed-text is available."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available

    try:
        import ollama

        client = ollama.Client(host="http://localhost:11434")
        resp = client.list()
        models = resp.get("models", [])
        model_names = [m.get("model", "") for m in models]
        _ollama_available = any("nomic-embed-text" in name for name in model_names)
    except Exception:
        _ollama_available = False

    return _ollama_available


def requires_ollama():
    """Pytest marker helper — skip if Ollama is not running."""
    return pytest.mark.skipif(
        not is_ollama_available(),
        reason="Ollama not running or nomic-embed-text not pulled",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config_path(monkeypatch):
    """Create a temporary YAML config path, clean up after test."""
    tmp = tempfile.mktemp(suffix=".yaml")
    monkeypatch.setenv("RAG_CONFIG_PATH", tmp)
    yield tmp
    if os.path.isfile(tmp):
        os.remove(tmp)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, clean up after test."""
    tmp = tempfile.mkdtemp()
    yield tmp
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_html_doxygen_function():
    """Return standard Doxygen HTML for a function memitem."""
    return """<html><body>
    <div class="memitem">
    <div class="memproto">
    <table><tr>
    <td class="memname">MwResult&nbsp;MwMultiAxis::CalculateToolpath&nbsp;(const MwGeometry&amp;&nbsp;geo, MwToolpath&amp;&nbsp;tp)</td>
    </tr></table>
    </div>
    <div class="memdoc">
    <p>Performs 5-axis toolpath calculation.</p>
    <dl><dt>Parameters:</dt>
    <dd>geo - Input geometry</dd>
    <dd>tp - Output toolpath</dd>
    </dl>
    </div>
    </div>
    </body></html>"""


@pytest.fixture
def sample_parsed_function():
    """Return a dict that matches what the parser produces for a function."""
    return {
        "type": "function",
        "symbol_id": "MwMultiAxis::CalculateToolpath",
        "class_name": "MwMultiAxis",
        "function_name": "CalculateToolpath",
        "signature": "MwResult CalculateToolpath(const MwGeometry& geo, MwToolpath& tp)",
        "params": [
            {"name": "geo", "type": "const MwGeometry&", "desc": "Input geometry"},
            {"name": "tp", "type": "MwToolpath&", "desc": "Output toolpath"},
        ],
        "return_desc": "MwResult indicating success",
        "remarks": "Performs 5-axis toolpath calculation.",
        "example": "MwMultiAxis ma;\nma.CalculateToolpath(geo, tp);",
        "see_also": ["MwToolpath", "MwGeometry"],
        "references": ["MwGeometry", "MwToolpath"],
        "contains_code": True,
        "source_label": "test",
        "source_module": "core",
        "file_path": "5axis/5axcore/public/mwmulti.h.html",
    }
