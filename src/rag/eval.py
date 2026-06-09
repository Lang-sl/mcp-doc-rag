"""Retrieval evaluation metrics and runner.

Provides Recall@K, MRR, NDCG@K metrics plus a structured evaluation
runner that operates on a JSONL file of annotated (query, relevant_chunk_ids) pairs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field


@dataclass
class StageEvalResult:
    """Aggregated per-stage metrics."""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0


@dataclass
class EvalResult:
    """Aggregated evaluation results across all queries."""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    num_queries: int = 0
    num_zero_recall: int = 0
    zero_recall_queries: list[str] = field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    # Per-stage metrics (populated when eval_mode=True is passed to evaluate())
    stages: dict[str, StageEvalResult] = field(default_factory=dict)
    # Bad case classification
    bad_cases: list[dict] = field(default_factory=list)
    num_knowledge_gap: int = 0
    num_ranking_failure: int = 0
    num_rewrite_regression: int = 0
    num_reranker_regression: int = 0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant chunks that appear in the top-K results.

    When *relevant_ids* is empty returns 1.0 (vacuous truth — the query
    has nothing to find, so any result set is trivially complete).
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result.

    Returns 0.0 when no relevant result is found in *retrieved_ids*.
    """
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg(scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain at K.

    Uses binary relevance: 1.0 for relevant, 0.0 for non-relevant.
    Discount is log2(rank + 1) so the first position (rank=1) gets
    log2(2) = 1 as the denominator.
    """
    return sum(
        score / math.log2(i + 2)
        for i, score in enumerate(scores[:k])
    )


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalized DCG@K — actual DCG divided by ideal DCG.

    Ideal ordering places all relevant results first.  Returns 0.0
    when no result is relevant (avoiding division by zero).
    """
    scores = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids[:k]]
    ideal = sorted(scores, reverse=True)

    actual_dcg = dcg(scores, k)
    ideal_dcg = dcg(ideal, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def _classify_bad_case(
    query: str,
    trace,  # PipelineTrace | None
    relevant_ids: set[str],
    final_recall_at_10: float,
) -> list[dict]:
    """Classify a zero-recall query into one or more bad-case categories."""
    bad: list[dict] = []

    bm25_recall = trace.recall_at("bm25", relevant_ids, 10) if trace else 0.0
    vec_recall = trace.recall_at("vector", relevant_ids, 10) if trace else 0.0
    rrf_recall = trace.recall_at("rrf", relevant_ids, 10) if trace else 0.0
    reranker_recall = trace.recall_at("reranker", relevant_ids, 10) if trace else 0.0

    # knowledge_gap: neither BM25 nor vector found anything
    if bm25_recall == 0.0 and vec_recall == 0.0:
        bad.append({
            "query": query,
            "category": "knowledge_gap",
            "detail": (
                "No relevant chunks in any index "
                "(BM25 Recall@10=0, Vector Recall@10=0)"
            ),
        })

    # ranking_failure: BM25 or vector found it but final results lost it
    if (bm25_recall > 0.0 or vec_recall > 0.0) and final_recall_at_10 == 0.0:
        bad.append({
            "query": query,
            "category": "ranking_failure",
            "detail": (
                f"BM25 found relevant chunks (R@10={bm25_recall:.2f}) but "
                f"none survived to final top-10"
            ),
        })

    # reranker_regression: reranker made things worse
    if rrf_recall > 0.0 and reranker_recall < rrf_recall:
        bad.append({
            "query": query,
            "category": "reranker_regression",
            "detail": (
                f"RRF Recall@10={rrf_recall:.2f} dropped to "
                f"{reranker_recall:.2f} after reranker"
            ),
        })

    return bad


def _aggregate_stage_metrics(
    traces: list,  # list[PipelineTrace]
    relevant_ids_map: dict[str, set[str]],
) -> dict[str, StageEvalResult]:
    """Aggregate per-stage Recall@5, Recall@10, MRR across all queries."""
    stage_names = ["bm25", "vector", "rrf", "reranker", "final"]
    stage_data: dict[str, dict[str, list[float]]] = {
        s: {"recall_5": [], "recall_10": [], "mrr": []}
        for s in stage_names
    }

    for trace in traces:
        relevant = relevant_ids_map.get(trace.query, set())
        for stage in stage_names:
            r5 = trace.recall_at(stage, relevant, 5)
            r10 = trace.recall_at(stage, relevant, 10)
            # Approximate MRR from trace results
            mrr_val = 0.0
            stage_results = None
            for t in trace.traces:
                if t.stage == stage:
                    stage_results = t.results
                    break
            if stage_results and relevant:
                for i, cid in enumerate(stage_results, start=1):
                    if cid in relevant:
                        mrr_val = 1.0 / i
                        break

            stage_data[stage]["recall_5"].append(r5)
            stage_data[stage]["recall_10"].append(r10)
            stage_data[stage]["mrr"].append(mrr_val)

    result: dict[str, StageEvalResult] = {}
    for stage in stage_names:
        result[stage] = StageEvalResult(
            recall_at_5=_safe_mean(stage_data[stage]["recall_5"]),
            recall_at_10=_safe_mean(stage_data[stage]["recall_10"]),
            mrr=_safe_mean(stage_data[stage]["mrr"]),
        )
    return result


def evaluate(
    retriever,
    queries_path: str,
    k_values: list[int] | None = None,
    source_label: str | None = None,
    enable_rewrite: bool = False,
) -> EvalResult:
    """Run a full evaluation pass over annotated queries.

    Args:
        retriever: A ``HybridRetriever`` instance (or any object with a
            ``search(query, source_label=..., enable_rewrite=...)`` method).
        queries_path: Path to a JSONL file where each line is
            ``{"query": "...", "relevant_chunk_ids": ["id1", ...]}``.
        k_values: List of K values for Recall@K / NDCG@K.  Defaults to
            ``[1, 3, 5, 10]``.
        source_label: Optional source label passed through to ``search()``
            to restrict the search scope.
        enable_rewrite: If True, passes ``enable_rewrite=True`` to the
            retriever's ``search()`` call.  Use this to compare retrieval
            quality with and without query rewrite.

    Returns:
        ``EvalResult`` with all aggregated metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    queries: list[dict] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    if not queries:
        return EvalResult()

    all_recall: dict[int, list[float]] = {k: [] for k in k_values}
    all_mrr: list[float] = []
    all_ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    all_latency: list[float] = []
    zero_recall_queries: list[str] = []
    traces: list = []  # list of PipelineTrace for per-stage analysis
    all_bad_cases: list[dict] = []

    for q in queries:
        query_text = q["query"]
        relevant = set(q["relevant_chunk_ids"])

        t0 = time.perf_counter()
        raw_result = retriever.search(
            query_text,
            source_label=source_label,
            enable_rewrite=enable_rewrite,
            eval_mode=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_latency.append(elapsed_ms)

        # Unpack trace from result
        query_trace = None
        if isinstance(raw_result, tuple):
            results, query_trace = raw_result
        else:
            results = raw_result

        retrieved_ids = [r.chunk.chunk_id for r in results]

        for k in k_values:
            r = recall_at_k(retrieved_ids, relevant, k)
            all_recall[k].append(r)

        all_mrr.append(mrr(retrieved_ids, relevant))

        for k in k_values:
            all_ndcg[k].append(ndcg_at_k(retrieved_ids, relevant, k))

        # Collect trace for per-stage analysis
        if query_trace is not None:
            traces.append(query_trace)

        # A query has zero recall when none of its relevant chunks appear
        # at the largest K value.
        final_recall = all_recall[max(k_values)][-1]
        if final_recall == 0.0:
            zero_recall_queries.append(query_text)
            if query_trace is not None:
                all_bad_cases.extend(
                    _classify_bad_case(query_text, query_trace, relevant, final_recall)
                )

    result = EvalResult(
        num_queries=len(queries),
        num_zero_recall=len(zero_recall_queries),
        zero_recall_queries=zero_recall_queries,
        mrr=_safe_mean(all_mrr),
        latency_p50_ms=_percentile(all_latency, 50),
        latency_p95_ms=_percentile(all_latency, 95),
    )

    # Per-stage metrics aggregation
    if traces:
        relevant_ids_map = {q["query"]: set(q["relevant_chunk_ids"]) for q in queries}
        result.stages = _aggregate_stage_metrics(traces, relevant_ids_map)

        # Attach bad cases and count categories
        result.bad_cases = all_bad_cases
        for bc in result.bad_cases:
            cat = bc["category"]
            if cat == "knowledge_gap":
                result.num_knowledge_gap += 1
            elif cat == "ranking_failure":
                result.num_ranking_failure += 1
            elif cat == "rewrite_regression":
                result.num_rewrite_regression += 1
            elif cat == "reranker_regression":
                result.num_reranker_regression += 1

    for k in k_values:
        setattr(result, f"recall_at_{k}", _safe_mean(all_recall[k]))
    for k in k_values:
        setattr(result, f"ndcg_at_{k}", _safe_mean(all_ndcg[k]))

    return result


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(math.ceil(p / 100.0 * len(sorted_vals))) - 1
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]
