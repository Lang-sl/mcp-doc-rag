from __future__ import annotations

import tomllib
from pathlib import Path


def test_pytest_writes_temporary_and_cache_files_under_output() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["addopts"] == "--basetemp=output/pytest-tmp"
    assert pytest_options["cache_dir"] == "output/pytest-cache"
