"""Unit tests for evaluation metric functions."""

from rag.eval import recall_at_k, mrr, ndcg_at_k


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_all_relevant():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_half_relevant():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == 0.5


def test_recall_at_k_none_relevant():
    assert recall_at_k(["d", "e"], {"a", "b"}, k=2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=2) == 1.0


def test_recall_at_k_k_larger_than_retrieved():
    assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5


def test_recall_at_k_exact_one_match():
    assert recall_at_k(["a"], {"a"}, k=1) == 1.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

def test_mrr_first_position():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_position():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5


def test_mrr_not_found():
    assert mrr(["a", "b", "c"], {"d"}) == 0.0


def test_mrr_empty_retrieved():
    assert mrr([], {"a"}) == 0.0


def test_mrr_empty_relevant():
    # When nothing is relevant, MRR is 0 by definition
    assert mrr(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_at_k_perfect():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_ndcg_at_k_worse():
    # With non-relevant results interspersed, ordering matters
    score = ndcg_at_k(["x", "b", "x", "a", "x"], {"a", "b", "c"}, k=5)
    assert score < 1.0
    assert score > 0.0


def test_ndcg_at_k_all_irrelevant():
    assert ndcg_at_k(["d", "e", "f"], {"a", "b"}, k=3) == 0.0


def test_ndcg_at_k_partial_relevant():
    # Relevant results at positions 4 and 5 is worse than at 1 and 2
    score_bad = ndcg_at_k(["x", "x", "x", "a", "b"], {"a", "b"}, k=5)
    score_good = ndcg_at_k(["a", "b", "x", "x", "x"], {"a", "b"}, k=5)
    assert score_good > score_bad
