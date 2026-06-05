"""Stage 1: Configuration loading and validation.

No external dependencies. Run first to verify the config system.

    pytest tests/test_01_config.py -v
"""

from __future__ import annotations

import os
import tempfile

from rag.config import Config, load_config, BM25Weights


class TestConfigDefaults:
    """Verify factory defaults are sensible."""

    def test_default_chroma_dir(self):
        config = Config()
        assert config.chroma_dir == "./output/chroma_db"

    def test_default_embed_dim(self):
        config = Config()
        assert config.embed_dim == 768

    def test_default_top_k(self):
        config = Config()
        assert config.top_k_default == 10

    def test_default_reranker(self):
        config = Config()
        assert "jina-reranker-v2" in config.reranker_model

    def test_bm25_weights_default(self):
        config = Config()
        w = config.bm25_weights
        assert w.symbol_name == 10.0
        assert w.signature == 5.0
        assert w.remarks == 1.0
        assert w.example == 0.5

    def test_code_boost_triggers_not_empty(self):
        config = Config()
        assert len(config.code_boost_triggers) > 0
        assert "how to" in config.code_boost_triggers

    def test_reranker_score_gap_threshold_default(self):
        config = Config()
        assert config.reranker_score_gap_threshold == 0.15

    def test_reranker_max_candidates_default(self):
        config = Config()
        assert config.reranker_max_candidates == 30


class TestLoadConfig:
    """Verify YAML loading with overrides and env var fallback."""

    def test_load_from_yaml_file(self):
        tmp = tempfile.mktemp(suffix=".yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("top_k_default: 42\nembed_dim: 384\n")

        config = load_config(tmp)
        assert config.top_k_default == 42
        assert config.embed_dim == 384
        # Unspecified fields keep defaults, resolved relative to config file dir
        import pathlib
        assert config.chroma_dir == str(pathlib.Path(tmp).parent / "output" / "chroma_db")

        os.remove(tmp)

    def test_load_from_env_var(self, monkeypatch):
        tmp = tempfile.mktemp(suffix=".yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("rrf_k: 15\n")

        monkeypatch.setenv("RAG_CONFIG_PATH", tmp)
        config = load_config()
        assert config.rrf_k == 15

        os.remove(tmp)

    def test_load_missing_file_uses_defaults(self, monkeypatch):
        monkeypatch.setenv("RAG_CONFIG_PATH", "/no/such/path.yaml")
        config = load_config()
        # Should return defaults without crashing
        assert config.top_k_default == 10

    def test_bm25_weights_from_yaml(self):
        tmp = tempfile.mktemp(suffix=".yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("bm25_weights:\n  symbol_name: 20.0\n  remarks: 2.0\n")

        config = load_config(tmp)
        w = config.bm25_weights
        assert w.symbol_name == 20.0
        assert w.remarks == 2.0
        # Unspecified BM25 fields keep defaults
        assert w.signature == 5.0

        os.remove(tmp)

    def test_reranker_fields_from_yaml(self):
        tmp = tempfile.mktemp(suffix=".yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("reranker_score_gap_threshold: 0.0\nreranker_max_candidates: 20\n")

        config = load_config(tmp)
        assert config.reranker_score_gap_threshold == 0.0
        assert config.reranker_max_candidates == 20

        os.remove(tmp)
