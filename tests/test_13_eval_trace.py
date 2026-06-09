"""Unit tests for PipelineTrace, BadCase classification, and stage aggregation."""
from rag.models import StageTrace, PipelineTrace


class TestPipelineTrace:
    """Tests for PipelineTrace.recall_at()."""

    def test_recall_at_full_match(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="bm25", results=["a", "b", "c"]))
        assert trace.recall_at("bm25", {"a", "b"}, k=3) == 1.0

    def test_recall_at_partial_match(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="bm25", results=["a", "b", "c"]))
        assert trace.recall_at("bm25", {"a", "b"}, k=1) == 0.5

    def test_recall_at_no_match(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="vector", results=["d", "e"]))
        assert trace.recall_at("vector", {"a", "b"}, k=2) == 0.0

    def test_recall_at_empty_relevant(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="bm25", results=["a"]))
        assert trace.recall_at("bm25", set(), k=1) == 1.0

    def test_recall_at_missing_stage(self):
        trace = PipelineTrace(query="test")
        assert trace.recall_at("nonexistent", {"a"}, k=5) == 0.0

    def test_recall_at_k_larger_than_results(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="final", results=["a"]))
        assert trace.recall_at("final", {"a", "b"}, k=10) == 0.5

    def test_multiple_stages_independent(self):
        trace = PipelineTrace(query="test")
        trace.traces.append(StageTrace(stage="bm25", results=["c", "a"]))
        trace.traces.append(StageTrace(stage="reranker", results=["a", "c"]))
        # BM25: "a" at rank 2 not in top-1
        assert trace.recall_at("bm25", {"a"}, k=1) == 0.0
        # Reranker: "a" at rank 1 found
        assert trace.recall_at("reranker", {"a"}, k=1) == 1.0


class TestBadCaseClassification:
    """Tests for _classify_bad_case logic."""

    def test_knowledge_gap_detected(self):
        from rag.eval import _classify_bad_case

        trace = PipelineTrace(query="how to export dxf")
        trace.traces.append(StageTrace(stage="bm25", results=["x", "y"]))
        trace.traces.append(StageTrace(stage="vector", results=["p", "q"]))
        bad = _classify_bad_case("how to export dxf", trace, {"a", "b"}, 0.0)
        categories = [b["category"] for b in bad]
        assert "knowledge_gap" in categories

    def test_ranking_failure_detected(self):
        from rag.eval import _classify_bad_case

        trace = PipelineTrace(query="init renderer")
        trace.traces.append(StageTrace(stage="bm25", results=["a", "x", "y"]))
        trace.traces.append(StageTrace(stage="vector", results=["p", "q"]))
        trace.traces.append(StageTrace(stage="final", results=["x", "y", "z", "w"]))
        bad = _classify_bad_case("init renderer", trace, {"a"}, 0.0)
        categories = [b["category"] for b in bad]
        assert "ranking_failure" in categories

    def test_reranker_regression_detected(self):
        from rag.eval import _classify_bad_case

        trace = PipelineTrace(query="setup axis")
        trace.traces.append(StageTrace(stage="bm25", results=["x"]))
        trace.traces.append(StageTrace(stage="vector", results=["x"]))
        trace.traces.append(StageTrace(stage="rrf", results=["a", "b", "c"]))
        trace.traces.append(StageTrace(stage="reranker", results=["c", "b"]))
        bad = _classify_bad_case("setup axis", trace, {"a"}, 0.5)
        categories = [b["category"] for b in bad]
        assert "reranker_regression" in categories

    def test_no_bad_case_when_found(self):
        from rag.eval import _classify_bad_case

        trace = PipelineTrace(query="good query")
        trace.traces.append(StageTrace(stage="bm25", results=["a", "b"]))
        trace.traces.append(StageTrace(stage="final", results=["b", "a"]))
        bad = _classify_bad_case("good query", trace, {"a"}, 1.0)
        assert len(bad) == 0
