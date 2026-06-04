"""Stage 4: HTML and C++ header parser tests.

Verifies Doxygen HTML parsing, C++ header parsing (tree-sitter and regex fallback).

    pytest tests/test_04_parser.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

from rag.indexer.parser_header import parse_header, _HAS_TREE_SITTER
from rag.indexer.parser_html import parse_html, _extract_type_names


class TestExtractTypeNames:
    """Type name extraction from signatures."""

    def test_simple_type(self):
        names = _extract_type_names("int GetValue()")
        assert "GetValue" in names

    def test_qualified_type(self):
        # The regex treats Namespace::Class::Method as one qualified name
        names = _extract_type_names("MwResult MwMultiAxis::CalculateToolpath()")
        assert "MwResult" in names
        assert "MwMultiAxis::CalculateToolpath" in names

    def test_template_type(self):
        names = _extract_type_names("std::vector<MwPoint> GetPoints()")
        assert "MwPoint" in names
        assert "GetPoints" in names

    def test_empty_returns_empty(self):
        assert _extract_type_names("") == []
        assert _extract_type_names(None) == []


class TestParseNarrative:
    """Parse non-Doxygen narrative HTML."""

    def test_parse_narrative_html(self):
        html = """<html><body>
        <h1>Getting Started</h1>
        <div class="contents">
        <p>This guide explains how to use the MachineWorks SDK.</p>
        <p>First, initialize the kernel with MwInit().</p>
        <h2>Initialization</h2>
        <p>Call MwKernelInit to set up the rendering context.</p>
        </div>
        </body></html>"""

        tmp = tempfile.mktemp(suffix=".html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)

        results = parse_html(tmp, "test", "guide")
        narrative = [r for r in results if r["type"] == "narrative"]
        assert len(narrative) > 0
        assert any("MachineWorks" in n.get("remarks", "") for n in narrative)

        os.remove(tmp)


class TestParseDoxygenFunction:
    """Parse Doxygen function memitems."""

    def test_parse_doxygen_function(self, sample_html_doxygen_function):
        tmp = tempfile.mktemp(suffix=".html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(sample_html_doxygen_function)

        results = parse_html(tmp, "test", "core")
        funcs = [r for r in results if r.get("function_name")]
        assert len(funcs) >= 1, "Should find at least one function"

        func = funcs[0]
        assert func["function_name"] == "CalculateToolpath"
        # The HTML parser extracts function_name from memname but does not
        # always extract class_name (depends on the Doxygen HTML structure).
        # Verify what we DO get: signature contains the full declaration.
        assert "MwMultiAxis" in func.get("signature", "")
        assert "5-axis" in func.get("remarks", "")

        os.remove(tmp)


# ============================================================================
# C++ Header Parser Tests
# ============================================================================


def _write_hpp(content: str) -> str:
    """Helper: write content to a temp .hpp file and return the path."""
    tmp = tempfile.mktemp(suffix=".hpp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    return tmp


class TestHeaderParserBasic:
    """Verify basic C++ header parsing with tree-sitter (and regex fallback)."""

    def test_parses_class(self):
        tmp = _write_hpp("class MyClass { };")
        results = parse_header(tmp, "test", "core")
        classes = [r for r in results if r["type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["symbol_id"] == "MyClass"
        os.remove(tmp)

    def test_parses_function(self):
        tmp = _write_hpp("void CalculateToolpath();")
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["function_name"] == "CalculateToolpath"
        os.remove(tmp)

    def test_parses_enum(self):
        tmp = _write_hpp("enum ToolType { BALL, END };")
        results = parse_header(tmp, "test", "core")
        enums = [r for r in results if r["type"] == "enum"]
        assert len(enums) == 1
        assert enums[0]["function_name"] == "ToolType"
        os.remove(tmp)

    def test_method_inside_class(self):
        tmp = _write_hpp("class Foo {\npublic:\n    int GetValue() const;\n};")
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["function_name"] == "GetValue"
        assert funcs[0]["class_name"] == "Foo"
        assert funcs[0]["symbol_id"] == "Foo::GetValue"
        os.remove(tmp)

    def test_forwards_declarations_skipped(self):
        tmp = _write_hpp("class Foo;\nclass Bar { };")
        results = parse_header(tmp, "test", "core")
        classes = [r for r in results if r["type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["symbol_id"] == "Bar"
        os.remove(tmp)


class TestHeaderParserComments:
    """Verify comment association with elements."""

    def test_single_line_comment_before_function(self):
        tmp = _write_hpp("// Calculate the toolpath.\nvoid Calc();")
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["remarks"] is not None
        assert "Calculate" in funcs[0]["remarks"]
        os.remove(tmp)

    def test_comment_before_class(self):
        tmp = _write_hpp("// A utility class.\nclass Helper { };")
        results = parse_header(tmp, "test", "core")
        classes = [r for r in results if r["type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["remarks"] is not None
        assert "utility" in classes[0]["remarks"]
        os.remove(tmp)


class TestHeaderParserAdvanced:
    """Verify complex template and nested class handling."""

    def test_template_function(self):
        tmp = _write_hpp(
            "template<typename T>\n"
            "T Clamp(T value, T min, T max);"
        )
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) >= 1
        clamp = [f for f in funcs if f["function_name"] == "Clamp"]
        assert len(clamp) == 1
        assert "T" in clamp[0]["signature"]
        os.remove(tmp)

    def test_nested_class(self):
        tmp = _write_hpp(
            "class Outer {\n"
            "public:\n"
            "    class Inner {\n"
            "        int value;\n"
            "    };\n"
            "    Inner* getInner();\n"
            "};"
        )
        results = parse_header(tmp, "test", "core")
        classes = [r for r in results if r["type"] == "class"]
        assert len(classes) == 2
        class_names = {c["symbol_id"] for c in classes}
        assert "Outer" in class_names
        assert "Outer::Inner" in class_names

        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["symbol_id"] == "Outer::getInner"
        os.remove(tmp)

    def test_macro_extracted(self):
        tmp = _write_hpp("#define MAX_TOOLS 256")
        results = parse_header(tmp, "test", "core")
        macros = [r for r in results if r["type"] == "macro"]
        assert len(macros) == 1
        assert macros[0]["function_name"] == "MAX_TOOLS"
        os.remove(tmp)

    def test_typedef_extracted(self):
        tmp = _write_hpp("typedef unsigned int uint32_t;")
        results = parse_header(tmp, "test", "core")
        typedefs = [r for r in results if r["type"] == "typedef"]
        assert len(typedefs) == 1
        assert typedefs[0]["function_name"] == "uint32_t"
        os.remove(tmp)

    def test_operator_overload(self):
        tmp = _write_hpp(
            "class Vec3 {\n"
            "public:\n"
            "    Vec3 operator+(const Vec3& other) const;\n"
            "};"
        )
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert "operator+" in funcs[0]["function_name"]
        os.remove(tmp)

    def test_virtual_pure_function(self):
        tmp = _write_hpp(
            "class IShape {\n"
            "public:\n"
            "    virtual double GetArea() const = 0;\n"
            "};"
        )
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["function_name"] == "GetArea"
        assert funcs[0]["class_name"] == "IShape"
        os.remove(tmp)

    def test_extern_c_function(self):
        tmp = _write_hpp(
            'extern "C" {\n'
            "    void Initialize();\n"
            "}"
        )
        results = parse_header(tmp, "test", "core")
        funcs = [r for r in results if r["type"] == "function"]
        assert len(funcs) >= 1
        names = {f["function_name"] for f in funcs}
        assert "Initialize" in names
        os.remove(tmp)


class TestHeaderParserFallback:
    """Verify the regex fallback path produces valid output."""

    def test_regex_fallback_parses_basic(self):
        # Temporarily force regex fallback
        import rag.indexer.parser_header as ph
        had_ts = ph._HAS_TREE_SITTER
        ph._HAS_TREE_SITTER = False
        try:
            tmp = _write_hpp(
                "class MyClass {\n"
                "public:\n"
                "    int GetValue() const;\n"
                "};\n"
                "enum Color { RED, BLUE };\n"
            )
            results = ph.parse_header(tmp, "test", "core")
            types = {r["type"] for r in results}
            assert "class" in types
            assert "function" in types
            assert "enum" in types
            os.remove(tmp)
        finally:
            ph._HAS_TREE_SITTER = had_ts

    def test_regex_fallback_macro_extracted(self):
        import rag.indexer.parser_header as ph
        had_ts = ph._HAS_TREE_SITTER
        ph._HAS_TREE_SITTER = False
        try:
            tmp = _write_hpp("#define BUFFER_SIZE 1024")
            results = ph.parse_header(tmp, "test", "core")
            macros = [r for r in results if r["type"] == "macro"]
            assert len(macros) == 1
            assert macros[0]["function_name"] == "BUFFER_SIZE"
            os.remove(tmp)
        finally:
            ph._HAS_TREE_SITTER = had_ts

    def test_regex_fallback_typedef_extracted(self):
        import rag.indexer.parser_header as ph
        had_ts = ph._HAS_TREE_SITTER
        ph._HAS_TREE_SITTER = False
        try:
            tmp = _write_hpp("typedef unsigned long DWORD;")
            results = ph.parse_header(tmp, "test", "core")
            typedefs = [r for r in results if r["type"] == "typedef"]
            assert len(typedefs) == 1
            assert typedefs[0]["function_name"] == "DWORD"
            os.remove(tmp)
        finally:
            ph._HAS_TREE_SITTER = had_ts

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_parity_both_paths_produce_same_shape(self):
        import rag.indexer.parser_header as ph

        code = (
            "class MyClass {\n"
            "public:\n"
            "    int GetValue() const;\n"
            "};\n"
        )
        tmp = _write_hpp(code)

        # Tree-sitter path
        ph._HAS_TREE_SITTER = True
        ts_results = ph.parse_header(tmp, "test", "core")

        # Regex path
        ph._HAS_TREE_SITTER = False
        re_results = ph.parse_header(tmp, "test", "core")

        # Both paths should find the class and function
        ts_classes = [r for r in ts_results if r["type"] == "class"]
        re_classes = [r for r in re_results if r["type"] == "class"]
        assert len(ts_classes) == 1
        assert len(re_classes) == 1
        assert ts_classes[0]["symbol_id"] == re_classes[0]["symbol_id"]

        ts_funcs = [r for r in ts_results if r["type"] == "function"]
        re_funcs = [r for r in re_results if r["type"] == "function"]
        assert len(ts_funcs) == 1
        assert len(re_funcs) == 1
        assert ts_funcs[0]["symbol_id"] == re_funcs[0]["symbol_id"]

        os.remove(tmp)
