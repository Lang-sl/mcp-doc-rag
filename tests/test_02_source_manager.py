"""Stage 2: Document source CRUD operations.

No external dependencies. Verifies source registration, listing, and removal.

    pytest tests/test_02_source_manager.py -v
"""

from __future__ import annotations

import os

from rag.config import Config
from rag.source_manager import add_source, remove_source, list_sources


class TestAddSource:
    """Registering new document sources."""

    def test_add_valid_source(self, tmp_config_path):
        config = Config()
        config.doc_sources = {}

        result = add_source(config, "test", os.path.dirname(__file__))
        assert result["ok"] is True
        assert config.doc_sources["test"] == os.path.dirname(__file__)

    def test_add_duplicate_label_fails(self, tmp_config_path):
        config = Config()
        config.doc_sources = {"test": "/tmp"}

        result = add_source(config, "test", "/other")
        assert result["ok"] is False

    def test_add_nonexistent_path_fails(self, tmp_config_path):
        config = Config()
        config.doc_sources = {}

        result = add_source(config, "test", "/does/not/exist")
        assert result["ok"] is False


class TestRemoveSource:
    """Removing registered sources."""

    def test_remove_existing_source(self, tmp_config_path):
        config = Config()
        config.doc_sources = {"test": os.path.dirname(__file__)}

        result = remove_source(config, "test")
        assert result["ok"] is True
        assert "test" not in config.doc_sources

    def test_remove_nonexistent_source(self, tmp_config_path):
        config = Config()
        config.doc_sources = {}

        result = remove_source(config, "no_such_source")
        assert result["ok"] is False


class TestListSources:
    """Listing registered sources."""

    def test_list_sources(self, tmp_config_path):
        config = Config()
        config.doc_sources = {"a": "/tmp/a", "b": "/tmp/b"}

        result = list_sources(config)
        # list_sources returns a list of dicts directly
        assert isinstance(result, list)
        assert len(result) == 2
        labels = [s["label"] for s in result]
        assert "a" in labels
        assert "b" in labels
