from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    type: str  # function, class, enum, macro, typedef, narrative, pdf_section
    symbol_id: str | None = None
    class_name: str | None = None
    function_name: str | None = None
    signature: str | None = None
    params: list[dict[str, str]] = field(default_factory=list)
    return_desc: str | None = None
    remarks: str | None = None
    example: str | None = None
    see_also: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    contains_code: bool = False
    source_label: str = ""
    source_module: str = ""
    source_file: str = ""
    embed_text: str = ""
    bm25_fields: dict[str, str] = field(default_factory=dict)

    @property
    def collection_name(self) -> str:
        return f"{self.source_label}.{self.source_module}"

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        # Fields excluded from ChromaDB metadata storage
        metadata.pop("embed_text", None)
        metadata.pop("params", None)
        metadata.pop("remarks", None)
        metadata.pop("example", None)
        metadata.pop("see_also", None)
        metadata.pop("bm25_fields", None)
        # Join references list into a comma-separated string
        metadata["references"] = ",".join(self.references)
        # ChromaDB Rust backend only accepts str|int|float|bool — convert None to ""
        for key in list(metadata.keys()):
            if metadata[key] is None and key != "contains_code":
                metadata[key] = ""
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], embed_text: str) -> Chunk:
        references_str: str = metadata.get("references", "")
        references: list[str] = [
            r.strip() for r in references_str.split(",") if r.strip()
        ]

        return cls(
            chunk_id=metadata["chunk_id"],
            type=metadata["type"],
            symbol_id=metadata.get("symbol_id"),
            class_name=metadata.get("class_name"),
            function_name=metadata.get("function_name"),
            signature=metadata.get("signature"),
            return_desc=metadata.get("return_desc"),
            contains_code=bool(metadata.get("contains_code", False)),
            source_label=metadata.get("source_label", ""),
            source_module=metadata.get("source_module", ""),
            source_file=metadata.get("source_file", ""),
            references=references,
            embed_text=embed_text,
        )


@dataclass
class Source:
    label: str
    path: str
    file_count: int = 0
    last_indexed: str | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class IndexStats:
    total_chunks: int
    total_sources: int
    per_source: dict[str, int]  # source_label -> chunk count


@dataclass
class StageTrace:
    """One stage of the retrieval pipeline."""
    stage: str           # "bm25" | "vector" | "rrf" | "reranker" | "final"
    results: list[str]   # ordered chunk_ids at this stage


@dataclass
class PipelineTrace:
    """Per-query trace of all retrieval stages, for eval analysis."""
    query: str
    traces: list[StageTrace] = field(default_factory=list)
    rewrite_variants: list[str] = field(default_factory=list)

    def recall_at(self, stage: str, relevant: set[str], k: int) -> float:
        """Compute Recall@K against *relevant* for a given stage."""
        for t in self.traces:
            if t.stage == stage:
                if not relevant:
                    return 1.0
                top_k = set(t.results[:k])
                return len(top_k & relevant) / len(relevant)
        return 0.0


@dataclass
class BadCase:
    """One bad case for analysis and feedback."""
    query: str
    category: str  # "knowledge_gap" | "ranking_failure" | "rewrite_regression" | "reranker_regression"
    detail: str    # one-line explanation of what went wrong


@dataclass
class RewriteResult:
    """Output of LLM query rewriting."""
    completed: str          # polished complete question
    sub_queries: list[str]  # 0-3 decomposed single-step queries
    variants: list[str]     # 0-3 semantic variants
