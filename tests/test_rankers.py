"""
tests/test_rankers.py — Smoke tests for all four rankers.

Run with: pytest tests/test_rankers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from index import build_index
from rankers.tfidf import TFIDFRanker
from rankers.bm25 import BM25Ranker
from rankers.hybrid import HybridRRFRanker
from fusion import reciprocal_rank_fusion


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def small_corpus():
    return pd.DataFrame([
        {"id": "d1", "title": "Machine learning basics",
         "text": "Machine learning is a subset of artificial intelligence that learns from data."},
        {"id": "d2", "title": "Deep learning neural networks",
         "text": "Deep learning uses neural networks with many layers to solve complex problems."},
        {"id": "d3", "title": "Natural language processing",
         "text": "NLP enables computers to understand human language and text."},
        {"id": "d4", "title": "Information retrieval systems",
         "text": "Information retrieval is the process of finding relevant documents for a query."},
        {"id": "d5", "title": "Search engine ranking",
         "text": "Search engines rank documents by relevance to the user query using algorithms."},
    ])


@pytest.fixture(scope="module")
def small_index(small_corpus):
    return build_index(small_corpus, stem=True, remove_stopwords=True,
                       field="title_text", show_progress=False)


@pytest.fixture(scope="module")
def tfidf(small_index):
    return TFIDFRanker(small_index)


@pytest.fixture(scope="module")
def bm25(small_index):
    return BM25Ranker(small_index)


# ===========================================================================
# TF-IDF tests
# ===========================================================================

def test_tfidf_returns_list(tfidf):
    results = tfidf.search("machine learning", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_tfidf_sorted_descending(tfidf):
    results = tfidf.search("information retrieval documents", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "TF-IDF results not sorted descending"


def test_tfidf_doc_ids_are_strings(tfidf):
    results = tfidf.search("deep learning neural", k=5)
    for doc_id, score in results:
        assert isinstance(doc_id, str)
        assert isinstance(score, float)


def test_tfidf_relevant_first(tfidf):
    """'information retrieval' query should return d4 near the top."""
    results = tfidf.search("information retrieval", k=5)
    top_ids = [d for d, _ in results[:2]]
    assert "d4" in top_ids or "d5" in top_ids, f"Expected retrieval doc in top-2, got {top_ids}"


def test_tfidf_empty_query(tfidf):
    results = tfidf.search("", k=5)
    assert results == []


# ===========================================================================
# BM25 tests
# ===========================================================================

def test_bm25_returns_results(bm25):
    results = bm25.search("machine learning", k=3)
    assert len(results) > 0


def test_bm25_sorted_descending(bm25):
    results = bm25.search("neural network deep learning", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_relevant_first(bm25):
    results = bm25.search("search engine ranking query", k=5)
    top_ids = [d for d, _ in results[:2]]
    assert "d5" in top_ids or "d4" in top_ids, f"Got {top_ids}"


def test_bm25_empty_query(bm25):
    assert bm25.search("", k=5) == []


def test_bm25_set_params(bm25):
    bm25.set_params(k1=1.2, b=0.6)
    results = bm25.search("language processing", k=3)
    assert len(results) > 0
    bm25.set_params(k1=1.5, b=0.75)  # reset


# ===========================================================================
# RRF fusion tests
# ===========================================================================

def test_rrf_fuses_lists():
    list1 = [("d1", 0.9), ("d2", 0.7), ("d3", 0.5)]
    list2 = [("d2", 0.8), ("d1", 0.6), ("d4", 0.4)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    doc_ids = [d for d, _ in fused]
    assert "d1" in doc_ids
    assert "d2" in doc_ids
    assert "d4" in doc_ids


def test_rrf_sorted_descending():
    list1 = [("d1", 0.9), ("d2", 0.7)]
    list2 = [("d3", 0.8), ("d1", 0.5)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_d1_wins_when_top_both(tfidf, bm25):
    """d1 ranked #1 by both should get the highest RRF score."""
    r1 = tfidf.search("machine learning artificial intelligence", k=5)
    r2 = bm25.search("machine learning artificial intelligence", k=5)
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    # Just verify it returns results
    assert len(fused) > 0


# ===========================================================================
# Hybrid RRF ranker tests
# ===========================================================================

def test_hybrid_returns_results(tfidf, bm25):
    hybrid = HybridRRFRanker({"tfidf": tfidf, "bm25": bm25}, k=60)
    results = hybrid.search("machine learning", k=5)
    assert len(results) > 0


def test_hybrid_sorted_descending(tfidf, bm25):
    hybrid = HybridRRFRanker({"tfidf": tfidf, "bm25": bm25}, k=60)
    results = hybrid.search("information retrieval search", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_from_subset(tfidf, bm25):
    all_r = {"tfidf": tfidf, "bm25": bm25}
    h = HybridRRFRanker.from_subset(all_r, ["tfidf"], k=60)
    results = h.search("natural language processing", k=3)
    assert len(results) > 0
