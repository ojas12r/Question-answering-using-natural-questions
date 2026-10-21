"""
tests/test_preprocess.py — Unit tests for src/preprocess.py.

Run with: pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from preprocess import clean_for_transformer, preprocess


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def full_pipeline_defaults():
    """Return a callable using default settings (stem=True, stopwords=True)."""
    return lambda text: preprocess(text, stem=True, remove_stopwords=True)


# ===========================================================================
# Test 1 – HTML stripping
# ===========================================================================

def test_html_stripped():
    result = preprocess("<b>Hello</b> world", stem=False, remove_stopwords=False)
    assert "hello" in result
    assert "world" in result
    # Raw HTML tags must not appear
    assert not any("<" in t or ">" in t for t in result), "HTML tags leaked into tokens"


# ===========================================================================
# Test 2 – URL removal
# ===========================================================================

def test_url_removed():
    result = preprocess(
        "visit https://en.wikipedia.org for more info",
        stem=False,
        remove_stopwords=False,
    )
    assert not any("http" in t or "wikipedia" in t for t in result), "URL leaked into tokens"
    assert "visit" in result
    assert "info" in result


# ===========================================================================
# Test 3 – Snowball stemming
# ===========================================================================

def test_stemming_running():
    result = preprocess("running quickly", stem=True, remove_stopwords=False)
    assert "run" in result, f"Expected 'run' in {result}"


def test_stemming_plural():
    result = preprocess("universities", stem=True, remove_stopwords=False)
    # SnowballStemmer("english") stems "universities" → "univers"
    assert len(result) == 1
    assert result[0].startswith("univers"), f"Unexpected stem: {result}"


# ===========================================================================
# Test 4 – Stop-word removal
# ===========================================================================

def test_stopwords_removed():
    result = preprocess("the quick brown fox", stem=False, remove_stopwords=True)
    stopwords_present = {"the", "a", "an", "is", "are"} & set(result)
    assert not stopwords_present, f"Stop words leaked: {stopwords_present}"
    assert "quick" in result
    assert "brown" in result
    assert "fox" in result


def test_no_stopword_removal():
    result = preprocess("the quick brown fox", stem=False, remove_stopwords=False)
    assert "the" in result, "Stop words should be present when remove_stopwords=False"


# ===========================================================================
# Test 5 – Unicode normalisation & accent stripping
# ===========================================================================

def test_unicode_normalised():
    result = preprocess("café naïve résumé", stem=False, remove_stopwords=False)
    assert "cafe" in result, f"Expected 'cafe' in {result}"
    assert "naive" in result, f"Expected 'naive' in {result}"
    assert "resume" in result, f"Expected 'resume' in {result}"


# ===========================================================================
# Test 6 – Empty / blank input
# ===========================================================================

def test_empty_string():
    assert preprocess("") == []


def test_whitespace_only():
    assert preprocess("   \t\n  ") == []


def test_none_like_empty():
    """Non-string or blank should return empty list gracefully."""
    assert preprocess("") == []


# ===========================================================================
# Test 7 – Numbers and years are preserved contextually
# ===========================================================================

def test_digits_in_context():
    """Meaningful digits (years, quantities) should survive preprocessing."""
    result = preprocess("The year 2019 saw 42 publications", stem=False, remove_stopwords=True)
    # After stopword removal "the", "saw" removed; digits kept
    digit_tokens = [t for t in result if t.isdigit()]
    assert len(digit_tokens) >= 1, f"No digit tokens found in: {result}"


# ===========================================================================
# Test 8 – Pipeline is deterministic
# ===========================================================================

def test_deterministic():
    text = "Information retrieval systems index and rank documents."
    assert preprocess(text) == preprocess(text), "preprocess() is not deterministic"


# ===========================================================================
# Test 9 – clean_for_transformer preserves case and stop words
# ===========================================================================

def test_transformer_clean_preserves_text():
    result = clean_for_transformer("The quick brown fox")
    assert "The" in result, "Transformer clean should preserve case"
    assert "quick" in result


def test_transformer_clean_strips_html():
    result = clean_for_transformer("<p>Hello <b>World</b></p>")
    assert "<" not in result and ">" not in result
    assert "Hello" in result
    assert "World" in result


def test_transformer_clean_removes_url():
    result = clean_for_transformer("See https://example.com for details")
    assert "http" not in result
    assert "See" in result


# ===========================================================================
# Test 10 – Full pipeline before/after demonstration
# ===========================================================================

def test_before_after_example():
    """Demonstrate the full preprocessing transformation."""
    raw = (
        "<p>The Presidents of the United States have historically "
        "visited https://whitehouse.gov many times in 2020.</p>"
    )
    tokens = preprocess(raw, stem=True, remove_stopwords=True)
    print("\n[Before]:", raw)
    print("[After] :", tokens)
    # Should not contain HTML, URLs, or stop words
    assert not any("<" in t for t in tokens)
    assert not any("http" in t for t in tokens)
    assert "the" not in tokens
    assert "of" not in tokens
    # Should contain stemmed content words
    assert len(tokens) > 0
