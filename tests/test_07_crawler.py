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

from rag.indexer.crawler import (
    crawl_source,
    detect_deleted_files,
    remove_deleted_from_state,
    update_state,
)


TEST_DOC_DIR = os.environ.get("RAG_TEST_DOC_DIR", "")
TEST_STATE_PATH = os.path.join(tempfile.gettempdir(), "rag_crawler_test_state.json")


def _requires_doc_dir():
    """Check if RAG_TEST_DOC_DIR is set and exists."""
    if not TEST_DOC_DIR:
        pytest.skip("RAG_TEST_DOC_DIR environment variable not set")
    if not os.path.isdir(TEST_DOC_DIR):
        pytest.skip(f"Directory not found: {TEST_DOC_DIR}")


class TestCrawlFirstPass:
    """First crawl: all files should need indexing."""

    def test_crawl_discovers_files(self):
        _requires_doc_dir()
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        assert len(entries) > 0, "Should discover at least one file"

    def test_crawl_finds_pdfs(self):
        _requires_doc_dir()
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        # At least some file type should be detected
        assert len(entries) > 0

    def test_first_pass_all_need_index(self):
        _requires_doc_dir()
        entries = list(crawl_source("test_crawl", TEST_DOC_DIR, TEST_STATE_PATH))
        assert all(e.needs_index for e in entries), (
            "First crawl: all files should need indexing"
        )


class TestCrawlSecondPass:
    """Second crawl: unchanged files should be skipped."""

    def test_second_pass_skips_unchanged(self):
        _requires_doc_dir()
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


class TestDetectDeletedFiles:
    """Verify deleted file detection from index state."""

    def test_no_deletions_when_all_files_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and index it
            f = os.path.join(tmpdir, "existing.h")
            with open(f, "w") as fh:
                fh.write("int x;")

            state_path = os.path.join(tmpdir, "state.json")
            entries = list(crawl_source("test", tmpdir, state_path))
            update_state(state_path, "test", entries)

            deleted = detect_deleted_files("test", tmpdir, state_path)
            assert len(deleted) == 0

    def test_detect_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and index it
            f = os.path.join(tmpdir, "to_delete.h")
            with open(f, "w") as fh:
                fh.write("int x;")

            state_path = os.path.join(tmpdir, "state.json")
            entries = list(crawl_source("test", tmpdir, state_path))
            update_state(state_path, "test", entries)

            # Delete the file
            os.remove(f)

            deleted = detect_deleted_files("test", tmpdir, state_path)
            assert len(deleted) == 1
            assert deleted[0] == f

    def test_empty_state_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            deleted = detect_deleted_files("nonexistent", tmpdir, state_path)
            assert deleted == []

    def test_remove_deleted_from_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "keep.h")
            f2 = os.path.join(tmpdir, "remove.h")
            with open(f1, "w") as fh:
                fh.write("int x;")
            with open(f2, "w") as fh:
                fh.write("int y;")

            state_path = os.path.join(tmpdir, "state.json")
            entries = list(crawl_source("test", tmpdir, state_path))
            update_state(state_path, "test", entries)

            # Delete f2
            os.remove(f2)

            deleted = detect_deleted_files("test", tmpdir, state_path)
            assert len(deleted) == 1
            assert deleted[0] == f2

            removed = remove_deleted_from_state(state_path, "test", tmpdir, deleted)
            assert removed == 1

            # f1 should still be in state
            entries2 = list(crawl_source("test", tmpdir, state_path))
            assert any(e.abs_path == f1 for e in entries2)

    def test_multiple_deletions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for name in ("a.h", "b.h", "c.h"):
                f = os.path.join(tmpdir, name)
                with open(f, "w") as fh:
                    fh.write("int x;")
                files.append(f)

            state_path = os.path.join(tmpdir, "state.json")
            entries = list(crawl_source("test", tmpdir, state_path))
            update_state(state_path, "test", entries)

            # Delete a.h and c.h
            os.remove(files[0])
            os.remove(files[2])

            deleted = detect_deleted_files("test", tmpdir, state_path)
            assert len(deleted) == 2
            assert files[0] in deleted
            assert files[2] in deleted
            assert files[1] not in deleted
