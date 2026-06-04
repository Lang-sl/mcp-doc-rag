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


def evaluate(
    retriever,
    queries_path: str,
    k_values: list[int] | None = None,
    source_label: str | None = None,
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

    for q in queries:
        query_text = q["query"]
        relevant = set(q["relevant_chunk_ids"])

        t0 = time.perf_counter()
        results = retriever.search(query_text, source_label=source_label)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        all_latency.append(elapsed_ms)

        retrieved_ids = [r.chunk.chunk_id for r in results]

        for k in k_values:
            r = recall_at_k(retrieved_ids, relevant, k)
            all_recall[k].append(r)

        all_mrr.append(mrr(retrieved_ids, relevant))

        for k in k_values:
            all_ndcg[k].append(ndcg_at_k(retrieved_ids, relevant, k))

        # A query has zero recall when none of its relevant chunks appear
        # at the largest K value.
        if all_recall[max(k_values)][-1] == 0.0:
            zero_recall_queries.append(query_text)

    result = EvalResult(
        num_queries=len(queries),
        num_zero_recall=len(zero_recall_queries),
        zero_recall_queries=zero_recall_queries,
        mrr=_safe_mean(all_mrr),
        latency_p50_ms=_percentile(all_latency, 50),
        latency_p95_ms=_percentile(all_latency, 95),
    )

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
