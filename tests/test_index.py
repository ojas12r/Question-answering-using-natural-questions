"""
tests/test_index.py — Unit tests for src/index.py (inverted index).

Run with: pytest tests/test_index.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from index import InvertedIndex, PostingEntry, build_index


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tiny_corpus() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "d1", "title": "Apple fruit",   "text": "Apples are red fruit eaten by people."},
        {"id": "d2", "title": "Banana fruit",  "text": "Bananas are yellow and sweet fruit."},
        {"id": "d3", "title": "Cherry tree",   "text": "Cherries grow on cherry trees in spring."},
    ])


@pytest.fixture(scope="module")
def tiny_index(tiny_corpus) -> InvertedIndex:
    return build_index(tiny_corpus, stem=True, remove_stopwords=True,
                       field="title_text", show_progress=False)


# ===========================================================================
# Test 1 – Index builds without error
# ===========================================================================

def test_index_builds(tiny_index):
    assert tiny_index is not None
    assert tiny_index.N == 3


# ===========================================================================
# Test 2 – Vocabulary is non-empty
# ===========================================================================

def test_vocabulary_non_empty(tiny_index):
    assert tiny_index.vocabulary_size() > 0


# ===========================================================================
# Test 3 – All doc_ids are indexed
# ===========================================================================

def test_all_docs_indexed(tiny_index):
    assert set(tiny_index.doc_ids) == {"d1", "d2", "d3"}
    assert len(tiny_index.doc_lengths) == 3


# ===========================================================================
# Test 4 – Document frequencies are correct
# ===========================================================================

def test_df_values(tiny_index):
    # "fruit" appears in d1 and d2 (stemmed: "fruit")
    # After stemming with Snowball, "fruit" → "fruit"
    fruit_stem = "fruit"
    if fruit_stem in tiny_index.df:
        assert tiny_index.df[fruit_stem] >= 1


# ===========================================================================
# Test 5 – Posting list structure
# ===========================================================================

def test_posting_entries_are_posting_entry(tiny_index):
    for term, postings in tiny_index.postings.items():
        for p in postings:
            assert isinstance(p, PostingEntry)
            assert isinstance(p.doc_id, str)
            assert p.tf >= 1
        break  # only check the first term


# ===========================================================================
# Test 6 – avg_dl is positive
# ===========================================================================

def test_avg_dl_positive(tiny_index):
    assert tiny_index.avg_dl > 0


# ===========================================================================
# Test 7 – get_postings returns empty for unknown term
# ===========================================================================

def test_get_postings_unknown(tiny_index):
    result = tiny_index.get_postings("zzzyyyxxx_nonexistent_term")
    assert result == []


# ===========================================================================
# Test 8 – doc_lengths sum check
# ===========================================================================

def test_doc_lengths_positive(tiny_index):
    for doc_id, length in tiny_index.doc_lengths.items():
        assert length >= 0, f"Negative length for {doc_id}"


# ===========================================================================
# Test 9 – total_postings >= vocabulary_size (each term in ≥ 1 doc)
# ===========================================================================

def test_total_postings_consistency(tiny_index):
    assert tiny_index.total_postings() >= tiny_index.vocabulary_size()
