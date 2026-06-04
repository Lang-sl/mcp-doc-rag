"""Stage 11: Full end-to-end pipeline.

Needs: Ollama running + document files available.
Set RAG_TEST_DOC_DIR to a directory with Doxygen HTML or PDF files.

    RAG_TEST_DOC_DIR=/path/to/docs pytest tests/test_11_e2e.py -v -m slow

Or skip with:

    pytest tests/ -v -k "not slow"
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tests.conftest import requires_ollama, is_ollama_available

TEST_DOC_DIR = os.environ.get("RAG_TEST_DOC_DIR", "")

needs_e2e_prereqs = pytest.mark.skipif(
    not (is_ollama_available() and os.path.isdir(TEST_DOC_DIR)),
    reason="E2E test needs Ollama running and RAG_TEST_DOC_DIR set to a valid directory",
)


@pytest.mark.slow
@needs_e2e_prereqs
class TestEndToEnd:
    """Full pipeline: index a small doc set -> search -> verify results."""

    def test_index_and_search(self):
        from rag.config import Config
        from rag.indexer.orchestrator import index_all
        from rag.retriever.hybrid import HybridRetriever

        # Use temp dirs to avoid polluting the real index
        base = tempfile.mkdtemp(prefix="rag_e2e_")

        config = Config()
        config.doc_sources = {"e2e_test": TEST_DOC_DIR}
        config.chroma_dir = os.path.join(base, "chroma_db")
        config.symbol_index_path = os.path.join(base, "symbol_index.json")
        config.index_state_path = os.path.join(base, ".index_state.json")

        try:
            # -- Index --
            result = index_all(config)
            assert result["total_chunks"] > 0, (
                f"E2E index should produce chunks, got {result}"
            )
            print(f"  Indexed {result['total_chunks']} chunks "
                  f"in {result.get('total_collections', '?')} collections")

            # -- Search --
            retriever = HybridRetriever(config)
            results = retriever.search("initialize function", top_k=5)
            assert len(results) > 0, "E2E search should return results"

            for r in results:
                print(f"  [{r.chunk.type}] {r.chunk.symbol_id or '(narrative)'} "
                      f"score={r.score:.4f}")

        finally:
            # -- Cleanup --
            import chromadb
            import shutil

            try:
                client = chromadb.PersistentClient(path=config.chroma_dir)
                for coll in client.list_collections():
                    client.delete_collection(name=coll.name)
            except Exception:
                pass

            shutil.rmtree(base, ignore_errors=True)
