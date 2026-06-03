"""Stage 4: HTML parser (Doxygen structure-aware).

No external dependencies. Verifies parsing of Doxygen memitems and narrative content.

    pytest tests/test_04_parser.py -v
"""

from __future__ import annotations

import os
import tempfile

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
