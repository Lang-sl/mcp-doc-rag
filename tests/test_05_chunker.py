"""Stage 5: Structured chunk assembly.

No external dependencies. Verifies that parsed elements become valid Chunk objects
with correct collection names, BM25 fields, and embed text.

    pytest tests/test_05_chunker.py -v
"""

from __future__ import annotations

from rag.config import Config
from rag.indexer.chunker import build_chunks


class TestBuildChunks:
    """Chunk assembly from parsed elements."""

    def test_build_function_chunk(self, sample_parsed_function):
        config = Config()
        chunks = build_chunks([sample_parsed_function], config)
        assert len(chunks) == 1

        chunk = chunks[0]
        assert chunk.symbol_id == "MwMultiAxis::CalculateToolpath"
        assert chunk.class_name == "MwMultiAxis"
        assert chunk.function_name == "CalculateToolpath"
        assert chunk.contains_code is True
        assert chunk.collection_name == "test.core"

    def test_references_in_chunk(self, sample_parsed_function):
        config = Config()
        chunks = build_chunks([sample_parsed_function], config)
        chunk = chunks[0]

        assert "MwGeometry" in chunk.references
        assert "MwToolpath" in chunk.references

    def test_embed_text_contains_key_info(self, sample_parsed_function):
        config = Config()
        chunks = build_chunks([sample_parsed_function], config)
        chunk = chunks[0]

        assert "MwMultiAxis" in chunk.embed_text
        assert "CalculateToolpath" in chunk.embed_text
        assert "5-axis" in chunk.embed_text

    def test_bm25_fields(self, sample_parsed_function):
        config = Config()
        chunks = build_chunks([sample_parsed_function], config)
        chunk = chunks[0]

        assert "CalculateToolpath" in chunk.bm25_fields["symbol_name"]
        assert "MwResult" in chunk.bm25_fields.get("signature", "")

    def test_chunk_with_no_remarks_or_signature_is_discarded(self):
        """Elements with neither remarks nor signature should be skipped."""
        config = Config()
        parsed = {
            "type": "narrative",
            "remarks": "",
            "signature": "",
            "source_label": "test",
            "source_module": "core",
            "file_path": "empty.html",
        }
        chunks = build_chunks([parsed], config)
        assert len(chunks) == 0

    def test_class_with_signature_but_no_remarks_is_kept(self):
        """Typedefs, enums, class declarations have signatures but no remarks."""
        config = Config()
        parsed = {
            "type": "typedef",
            "symbol_id": "MwVector",
            "signature": "typedef std::array<double, 3> MwVector;",
            "remarks": "",
            "source_label": "test",
            "source_module": "core",
            "file_path": "types.h",
        }
        chunks = build_chunks([parsed], config)
        assert len(chunks) == 1
        assert chunks[0].symbol_id == "MwVector"
