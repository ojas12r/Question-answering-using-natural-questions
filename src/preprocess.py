"""
src/preprocess.py — Text preprocessing pipeline for NQ Evidence Retrieval.

Two modes:
  1. Full pipeline  (stem=True, remove_stopwords=True)  → used for TF-IDF and BM25.
  2. Light cleaning (clean_for_transformer)             → used for Sentence-Transformer.

Design rationale
----------------
Stemming and stop-word removal benefit sparse models by reducing vocabulary
mismatch. However, these steps *hurt* transformer-based models because:
  - Sub-word tokenisers rely on full word-forms for correct tokenisation.
  - Stop words carry syntactic signals that attention heads exploit.
  - Stemming is lossy and irreversible; transformers already learn morphology.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List

import nltk


# ---------------------------------------------------------------------------
# One-time NLTK resource downloads (idempotent)
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download NLTK resources if not already present (runs at import time)."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


_ensure_nltk_resources()

from nltk.corpus import stopwords as nltk_stopwords  # noqa: E402 (after download)
from nltk.stem import SnowballStemmer, PorterStemmer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

# Pre-build stop-word set (frozen for speed)
_STOPWORDS: frozenset[str] = frozenset(nltk_stopwords.words("english"))

# Pre-build stemmers (one instance, reused)
_SNOWBALL = SnowballStemmer("english")
_PORTER = PorterStemmer()

# Regex patterns (compiled once)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WIKI_MARKUP_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")  # [[link|text]] → text
_WHITESPACE_RE = re.compile(r"\s+")
# Remove punctuation but keep: hyphens inside words, apostrophes, digits
# We strip leading/trailing punctuation from tokens later.
_PUNCT_RE = re.compile(r"[^\w\s\'-]")


# ---------------------------------------------------------------------------
# Light cleaning (used for Sentence-Transformer inputs)
# ---------------------------------------------------------------------------

def clean_for_transformer(text: str) -> str:
    """
    Apply minimal cleaning suitable for transformer models.

    Steps:
      1. Unicode NFKD normalisation (e.g. convert ligatures, full-width chars).
      2. Decode into ASCII where possible (strip accents from latin chars).
      3. Strip HTML/XML tags.
      4. Expand wiki markup ``[[link|text]]`` → ``text``.
      5. Remove bare URLs.
      6. Collapse multiple whitespace into a single space.

    Parameters
    ----------
    text : str
        Raw passage or query text.

    Returns
    -------
    str
        Lightly cleaned text, preserving case and stop words.
    """
    if not text:
        return ""
    # 1. Unicode normalise
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # 2. Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # 3. Expand wiki markup
    text = _WIKI_MARKUP_RE.sub(r"\1", text)
    # 4. Remove URLs
    text = _URL_RE.sub(" ", text)
    # 5. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (TF-IDF / BM25)
# ---------------------------------------------------------------------------

def preprocess(
    text: str,
    stem: bool = True,
    remove_stopwords: bool = True,
    stemmer: str = "snowball",
    lowercase: bool = True,
) -> List[str]:
    """
    Full text preprocessing pipeline: clean → tokenise → stop-word removal → stem.

    Parameters
    ----------
    text : str
        Raw input text (passage title, passage body, or query).
    stem : bool
        If True, apply Snowball (or Porter) stemming to each token.
    remove_stopwords : bool
        If True, drop English NLTK stop words.
    stemmer : str
        "snowball" (default) or "porter".
    lowercase : bool
        If True, lowercase the text before tokenisation.

    Returns
    -------
    List[str]
        Ordered list of processed tokens (may be empty for blank input).

    Examples
    --------
    >>> preprocess("The quick brown fox jumps over the lazy dog")
    ['quick', 'brown', 'fox', 'jump', 'over', 'lazi', 'dog']

    >>> preprocess("<b>Hello</b> World https://example.com", stem=False)
    ['hello', 'world']
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # ---- Step 1: Clean -------------------------------------------------------
    # Unicode normalise
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Strip HTML
    text = _HTML_TAG_RE.sub(" ", text)
    # Expand wiki markup
    text = _WIKI_MARKUP_RE.sub(r"\1", text)
    # Remove URLs
    text = _URL_RE.sub(" ", text)
    # Lowercase
    if lowercase:
        text = text.lower()
    # Remove punctuation (keep digits, hyphens inside words, apostrophes)
    text = _PUNCT_RE.sub(" ", text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # ---- Step 2: Tokenise ----------------------------------------------------
    tokens: List[str] = word_tokenize(text)

    # ---- Step 3: Stop-word removal -------------------------------------------
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # ---- Step 4: Stemming ----------------------------------------------------
    if stem:
        _stemmer = _SNOWBALL if stemmer == "snowball" else _PORTER
        tokens = [_stemmer.stem(t) for t in tokens]

    # Drop empty strings that can appear after stemming edge cases
    tokens = [t for t in tokens if t.strip()]
    return tokens


# ---------------------------------------------------------------------------
# Convenience: preprocess a dict of documents
# ---------------------------------------------------------------------------

def preprocess_corpus(
    docs: dict[str, str],
    stem: bool = True,
    remove_stopwords: bool = True,
) -> dict[str, List[str]]:
    """
    Preprocess an entire corpus dict mapping doc_id → text.

    Returns
    -------
    dict[str, List[str]]
        Mapping doc_id → token list.
    """
    return {
        doc_id: preprocess(text, stem=stem, remove_stopwords=remove_stopwords)
        for doc_id, text in docs.items()
    }


if __name__ == "__main__":
    examples = [
        "<b>Natural Questions:</b> What is the capital of Australia?",
        "Visit https://en.wikipedia.org/wiki/Canberra for details on Canberra's 1913 establishment.",
        "The quick brown foxes were jumping over the lazy dogs in naïve cafés!",
        "Who was elected president of the United States in the year 2020?",
    ]

    print("=" * 70)
    print(" NQ Evidence Retrieval — Preprocessing Pipeline Demonstration")
    print("=" * 70)

    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Raw Input:             {ex}")
        print(f"Transformer Clean:     {clean_for_transformer(ex)}")
        sparse_tokens = preprocess(ex, stem=True, remove_stopwords=True)
        print(f"Sparse Pipeline (BM25): {sparse_tokens}")
    print("\n" + "=" * 70)

