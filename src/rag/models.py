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
