from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class BM25Weights:
    symbol_name: float = 10.0
    signature: float = 5.0
    remarks: float = 1.0
    example: float = 0.5


@dataclass
class Config:
    chroma_dir: str = "./chroma_db"
    symbol_index_path: str = "./symbol_index.json"
    index_state_path: str = "./.index_state.json"
    doc_sources: dict[str, str] = field(default_factory=dict)
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    embed_batch_size: int = 256
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    reranker_max_length: int = 512
    chunk_max_chars: int = 2000
    chunk_overlap_ratio: float = 0.10
    top_k_default: int = 10
    candidate_expand_factor: int = 4
    rrf_k: int = 30
    bm25_weights: BM25Weights = field(default_factory=BM25Weights)
    code_boost_ratio: float = 1.20
    code_boost_triggers: list[str] = field(default_factory=lambda: [
        "how to", "create", "example", "sample", "write", "implement", "setup"
    ])
    ref_expansion_max: int = 5
    context_max_tokens: int = 6000
    cache_max_entries: int = 128
    index_batch_size: int = 500
    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = 3
    rrf_bm25_weight: float = 2.0
    embedding_cache_dir: str = "./chroma_db/embedding_cache"
    bm25_cache_dir: str = "./chroma_db/bm25_cache"
    reranker_score_gap_threshold: float = 0.15
    reranker_max_candidates: int = 30


_DEFAULT_CONFIG_PATH = "./config.yaml"

# Fields present in the dataclass but not serialized to YAML
_YAML_SKIP_FIELDS = {"index_state_path"}


def _dict_to_bm25_weights(data: dict) -> BM25Weights:
    """Convert a dict to BM25Weights, using defaults for missing keys."""
    defaults = BM25Weights()
    return BM25Weights(
        symbol_name=data.get("symbol_name", defaults.symbol_name),
        signature=data.get("signature", defaults.signature),
        remarks=data.get("remarks", defaults.remarks),
        example=data.get("example", defaults.example),
    )


def _resolve_path(raw: str, config_dir: str) -> str:
    """If *raw* is a relative path, resolve it against *config_dir*.

    Absolute paths are returned unchanged. This ensures paths like
    ``./chroma_db`` always resolve relative to the config file location,
    not the current working directory (which may differ when launched
    via MCP from another project).
    """
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(config_dir, raw))


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file, filling defaults for missing keys.

    If *path* is None, the ``RAG_CONFIG_PATH`` environment variable is checked,
    falling back to ``./config.yaml`` (current working directory).

    Relative paths in the config (``chroma_dir``, ``symbol_index_path``,
    ``index_state_path``) are resolved against the directory containing the
    config file — not CWD — so the index location is deterministic regardless
    of where the MCP server process is launched from.
    """
    if path is None:
        path = os.environ.get("RAG_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    # Resolve the config path itself first, then derive its directory
    config_path = os.path.abspath(path)
    config_dir = os.path.dirname(config_path)

    defaults = Config()
    yaml_data: dict = {}

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            yaml_data = loaded

    # Resolve each field: use YAML value if the key exists, otherwise the default
    def _get(key: str, default):
        if key in yaml_data:
            return yaml_data[key]
        return default

    bm25 = defaults.bm25_weights
    if "bm25_weights" in yaml_data and isinstance(yaml_data["bm25_weights"], dict):
        bm25 = _dict_to_bm25_weights(yaml_data["bm25_weights"])

    return Config(
        chroma_dir=_resolve_path(_get("chroma_dir", defaults.chroma_dir), config_dir),
        symbol_index_path=_resolve_path(_get("symbol_index_path", defaults.symbol_index_path), config_dir),
        index_state_path=_resolve_path(_get("index_state_path", defaults.index_state_path), config_dir),
        doc_sources=_get("doc_sources", defaults.doc_sources),
        ollama_host=_get("ollama_host", defaults.ollama_host),
        embed_model=_get("embed_model", defaults.embed_model),
        embed_dim=_get("embed_dim", defaults.embed_dim),
        embed_batch_size=_get("embed_batch_size", defaults.embed_batch_size),
        reranker_model=_get("reranker_model", defaults.reranker_model),
        reranker_max_length=_get("reranker_max_length", defaults.reranker_max_length),
        chunk_max_chars=_get("chunk_max_chars", defaults.chunk_max_chars),
        chunk_overlap_ratio=_get("chunk_overlap_ratio", defaults.chunk_overlap_ratio),
        top_k_default=_get("top_k_default", defaults.top_k_default),
        candidate_expand_factor=_get("candidate_expand_factor", defaults.candidate_expand_factor),
        rrf_k=_get("rrf_k", defaults.rrf_k),
        bm25_weights=bm25,
        code_boost_ratio=_get("code_boost_ratio", defaults.code_boost_ratio),
        code_boost_triggers=_get("code_boost_triggers", defaults.code_boost_triggers),
        ref_expansion_max=_get("ref_expansion_max", defaults.ref_expansion_max),
        context_max_tokens=_get("context_max_tokens", defaults.context_max_tokens),
        cache_max_entries=_get("cache_max_entries", defaults.cache_max_entries),
        index_batch_size=_get("index_batch_size", defaults.index_batch_size),
        query_rewrite_enabled=_get("query_rewrite_enabled", defaults.query_rewrite_enabled),
        query_rewrite_max_variants=_get("query_rewrite_max_variants", defaults.query_rewrite_max_variants),
        rrf_bm25_weight=_get("rrf_bm25_weight", defaults.rrf_bm25_weight),
        embedding_cache_dir=_resolve_path(_get("embedding_cache_dir", defaults.embedding_cache_dir), config_dir),
        bm25_cache_dir=_resolve_path(_get("bm25_cache_dir", defaults.bm25_cache_dir), config_dir),
        reranker_score_gap_threshold=_get("reranker_score_gap_threshold", defaults.reranker_score_gap_threshold),
        reranker_max_candidates=_get("reranker_max_candidates", defaults.reranker_max_candidates),
    )


def save_config(config: Config, path: Optional[str] = None) -> None:
    """Write *config* to a YAML file, creating parent directories as needed.

    If *path* is None, the ``RAG_CONFIG_PATH`` environment variable is checked,
    falling back to ``./config.yaml`` (current working directory).
    """
    if path is None:
        path = os.environ.get("RAG_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    from dataclasses import asdict

    data = asdict(config)

    # Remove fields that should not appear in the YAML file
    for skip_field in _YAML_SKIP_FIELDS:
        data.pop(skip_field, None)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
