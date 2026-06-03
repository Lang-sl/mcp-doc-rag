from __future__ import annotations

import os
from typing import Union

from rag.config import Config, save_config


def add_source(config: Config, label: str, path: str) -> dict:
    """Register a new document source. Returns {ok: True/False, label, path} or {ok: False, error: ...}."""
    # Check label uniqueness
    if label in config.doc_sources:
        return {"ok": False, "error": f"Source with label '{label}' already exists."}

    # Normalize and check path
    normalized = os.path.normpath(path)
    if not os.path.isdir(normalized):
        return {"ok": False, "error": f"Path does not exist or is not a directory: {normalized}"}

    # Add to config.doc_sources
    config.doc_sources[label] = normalized

    # Save config
    save_config(config)

    return {"ok": True, "label": label, "path": normalized}


def remove_source(config: Config, label: str) -> dict:
    """Remove a registered source. Returns {ok: True/False, ...}."""
    # Check label exists
    if label not in config.doc_sources:
        return {"ok": False, "error": f"Source with label '{label}' not found."}

    # Pop from config.doc_sources
    removed_path = config.doc_sources.pop(label)

    # Save config
    save_config(config)

    return {"ok": True, "label": label, "path": removed_path}


def list_sources(config: Config) -> list[dict]:
    """List all registered sources. Returns [{label, path}, ...]."""
    return [{"label": label, "path": path} for label, path in config.doc_sources.items()]
