"""
tests/test_evaluate.py — Unit tests for src/evaluate.py metric implementations.

Run with: pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from evaluate import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    mrr_at_k,
    evaluate_query,
    aggregate_metrics,
)


# ===========================================================================
# Precision@K
# ===========================================================================

def test_precision_perfect():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d2", "d3"}
    assert precision_at_k(retrieved, relevant, 3) == 1.0


def test_precision_zero():
    assert precision_at_k(["d1", "d2"], {"d3"}, 2) == 0.0


def test_precision_partial():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d1", "d3"}
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)


def test_precision_k_larger_than_retrieved():
    retrieved = ["d1", "d2"]
    relevant = {"d1"}
    # k=10 but only 2 retrieved — P@10 = 1/10 = 0.1
    assert precision_at_k(retrieved, relevant, 10) == pytest.approx(0.1)


# ===========================================================================
# Recall@K
# ===========================================================================

def test_recall_perfect():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d2", "d3"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_recall_zero():
    assert recall_at_k(["d1"], {"d2"}, 1) == 0.0


def test_recall_partial():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d4"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)


def test_recall_empty_relevant():
    assert recall_at_k(["d1", "d2"], set(), 2) == 0.0


# ===========================================================================
# Average Precision (AP)
# ===========================================================================

def test_ap_perfect():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d2", "d3"}
    assert average_precision(retrieved, relevant) == pytest.approx(1.0)


def test_ap_worst():
    """Relevant docs at the end → low AP."""
    retrieved = ["d4", "d5", "d1"]
    relevant = {"d1"}
    # AP = P@3 when d1 is found at rank 3 = (1/3) / 1 = 0.333
    assert average_precision(retrieved, relevant) == pytest.approx(1 / 3)


def test_ap_empty_relevant():
    assert average_precision(["d1", "d2"], set()) == 0.0


def test_ap_known_value():
    """Manual AP: retrieved=[d1, d2, d3, d4], relevant={d1, d3}
    P@1=1/1=1.0 (d1 relevant), P@3=2/3 (d3 relevant) → AP=(1.0+2/3)/2 = 5/6"""
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d1", "d3"}
    expected = (1.0 + 2 / 3) / 2
    assert average_precision(retrieved, relevant) == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# nDCG@K
# ===========================================================================

def test_ndcg_perfect():
    relevant = {"d1": 1, "d2": 1, "d3": 1}
    retrieved = ["d1", "d2", "d3"]
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_ndcg_zero():
    relevant = {"d1": 1}
    retrieved = ["d2", "d3", "d4"]
    assert ndcg_at_k(retrieved, relevant, 3) == 0.0


def test_ndcg_partial_binary():
    """d1 at rank 1, d2 not retrieved, d3 at rank 2 → compare to ideal [d1, d3]."""
    relevant = {"d1": 1, "d3": 1}
    retrieved = ["d1", "d4", "d3"]
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63... = 1.63...
    import math
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(dcg / idcg, rel=1e-5)


# ===========================================================================
# MRR@K
# ===========================================================================

def test_mrr_first():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1"}
    assert mrr_at_k(retrieved, relevant, 10) == pytest.approx(1.0)


def test_mrr_third():
    retrieved = ["d2", "d3", "d1"]
    relevant = {"d1"}
    assert mrr_at_k(retrieved, relevant, 10) == pytest.approx(1 / 3)


def test_mrr_not_found():
    retrieved = ["d2", "d3"]
    relevant = {"d1"}
    assert mrr_at_k(retrieved, relevant, 10) == 0.0


# ===========================================================================
# Integration: evaluate_query + aggregate_metrics
# ===========================================================================

def test_evaluate_query_keys():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    qrels = {"d1": 1, "d3": 1}
    result = evaluate_query(retrieved, qrels, cutoffs=[10, 50, 100])
    for key in ["P@10", "nDCG@10", "MRR@10", "AP", "R@10", "R@50", "R@100"]:
        assert key in result, f"Missing metric: {key}"


def test_aggregate_map_key():
    per_query = {
        "q1": {"P@10": 0.5, "nDCG@10": 0.6, "MRR@10": 1.0, "AP": 0.7, "R@10": 0.3},
        "q2": {"P@10": 0.3, "nDCG@10": 0.4, "MRR@10": 0.5, "AP": 0.5, "R@10": 0.2},
    }
    agg = aggregate_metrics(per_query)
    assert "MAP" in agg
    assert "AP" not in agg
    assert agg["MAP"] == pytest.approx((0.7 + 0.5) / 2)
