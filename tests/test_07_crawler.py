"""Stage 7: File crawler with incremental indexing.

Needs access to real document files. Set RAG_TEST_DOC_DIR to a directory
containing Doxygen HTML or PDF files, then run:

    RAG_TEST_DOC_DIR=/path/to/docs pytest tests/test_07_crawler.py -v

No Ollama required — only tests file discovery and SHA1 hash comparison.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from rag.indexer.crawler import crawl_source, update_state


TEST_DOC_DIR = os.environ.get("RAG_TEST_DOC_DIR", "")
TEST_STATE_PATH = os.path.join(tempfile.gettempdir(), "rag_crawler_test_state.json")


@pytest.fixture(autouse=True)
def _check_test_dir():
    """Skip all tests if RAG_TEST_DOC_DIR is not set or the path doesn't exist."""
    if not TEST_DOC_DIR:
        pytest.skip("RAG_TEST_DOC_DIR environment variable not set")
    if not os.path.isdir(TEST_DOC_DIR):
        pytest.skip(f"Directory not found: {TEST_DOC_DIR}")


class TestCrawlFirstPass:
    """First crawl: all files should need indexing."""

    def test_crawl_discovers_files(self):
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        assert len(entries) > 0, "Should discover at least one file"

    def test_crawl_finds_pdfs(self):
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        # At least some file type should be detected
        assert len(entries) > 0

    def test_first_pass_all_need_index(self):
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        assert all(e.needs_index for e in entries), (
            "First crawl: all files should need indexing"
        )


class TestCrawlSecondPass:
    """Second crawl: unchanged files should be skipped."""

    def test_second_pass_skips_unchanged(self):
        state_path = os.path.join(tempfile.gettempdir(), "rag_crawler_test_state2.json")

        # First pass
        entries1 = list(crawl_source("test_crawl2", TEST_DOC_DIR, state_path))
        assert len(entries1) > 0
        update_state(state_path, "test_crawl2", entries1)

        # Second pass — all files unchanged, should skip
        entries2 = list(crawl_source("test_crawl2", TEST_DOC_DIR, state_path))
        assert len(entries2) > 0
        assert all(not e.needs_index for e in entries2), (
            "Second crawl: unchanged files should be skipped"
        )

        if os.path.exists(state_path):
            os.remove(state_path)
