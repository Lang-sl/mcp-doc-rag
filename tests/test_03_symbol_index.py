"""Stage 3: Symbol index operations.

No external dependencies. Verifies O(1) hash-map lookup and source-scoped removal.

    pytest tests/test_03_symbol_index.py -v
"""

from __future__ import annotations

import os
import tempfile

from rag.models import Chunk
from rag.symbol_index import SymbolIndex


class TestSymbolLookup:
    """Add and lookup symbols by ID."""

    def test_add_and_lookup(self):
        tmp = tempfile.mktemp(suffix=".json")
        idx = SymbolIndex(tmp)

        chunk = Chunk(
            chunk_id="1",
            type="function",
            symbol_id="TestClass::TestMethod",
            class_name="TestClass",
            function_name="TestMethod",
            source_label="test",
            source_module="core",
            source_file="test.h",
        )
        idx.add_chunk(chunk)

        result = idx.lookup("TestClass::TestMethod")
        assert result is not None
        assert result["class_name"] == "TestClass"
        assert result["function_name"] == "TestMethod"

        if os.path.isfile(tmp):
            os.remove(tmp)

    def test_lookup_missing_returns_none(self):
        tmp = tempfile.mktemp(suffix=".json")
        idx = SymbolIndex(tmp)

        assert idx.lookup("Does::NotExist") is None

        if os.path.isfile(tmp):
            os.remove(tmp)

    def test_multiple_symbols(self):
        tmp = tempfile.mktemp(suffix=".json")
        idx = SymbolIndex(tmp)

        chunks = [
            Chunk(
                chunk_id="1", type="function", symbol_id="A::foo",
                source_label="s1", source_module="x", source_file="x",
            ),
            Chunk(
                chunk_id="2", type="function", symbol_id="B::bar",
                source_label="s1", source_module="x", source_file="x",
            ),
            Chunk(
                chunk_id="3", type="class", symbol_id="C",
                source_label="s1", source_module="x", source_file="x",
            ),
        ]
        for c in chunks:
            idx.add_chunk(c)

        assert len(idx) == 3
        assert idx.lookup("A::foo") is not None
        assert idx.lookup("B::bar") is not None
        assert idx.lookup("C") is not None

        if os.path.isfile(tmp):
            os.remove(tmp)


class TestSourceRemoval:
    """Removing all symbols belonging to a source."""

    def test_remove_source(self):
        tmp = tempfile.mktemp(suffix=".json")
        idx = SymbolIndex(tmp)

        c1 = Chunk(
            chunk_id="1", type="function", symbol_id="A::b",
            source_label="s1", source_module="x", source_file="x",
        )
        c2 = Chunk(
            chunk_id="2", type="function", symbol_id="C::d",
            source_label="s2", source_module="x", source_file="x",
        )
        idx.add_chunk(c1)
        idx.add_chunk(c2)

        assert len(idx) == 2
        removed = idx.remove_source("s1")
        assert removed == 1
        assert len(idx) == 1
        assert idx.lookup("A::b") is None
        assert idx.lookup("C::d") is not None

        if os.path.isfile(tmp):
            os.remove(tmp)
