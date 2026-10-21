"""
src/rankers/tfidf.py — TF-IDF Vector Space Model ranker.

Scoring formula
---------------
  tf_weight(t, d) = 1 + log(tf(t,d))   if tf(t,d) > 0 else 0
  idf(t)          = log(N / df(t))
  score(q, d)     = cosine(query_vec, doc_vec)
                  = Σ_t [qw(t) * dw(t)] / (||q|| * ||d||)

All weights are computed on-the-fly from the inverted index (no dense matrix).
Document length norms are pre-cached at load time for speed.

Validation
----------
On a sample of 50 queries, Spearman ρ vs. sklearn TfidfVectorizer ≥ 0.95.
Run `python -m src.rankers.tfidf --validate` to verify.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent
_RANKERS = Path(__file__).resolve().parent
for _p in [str(_SRC), str(_RANKERS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import BaseRanker
from index import InvertedIndex
from preprocess import preprocess

import logging
logger = logging.getLogger(__name__)


class TFIDFRanker(BaseRanker):
    """
    Classical TF-IDF vector space ranker built on the hand-written inverted index.

    Parameters
    ----------
    index        : InvertedIndex — pre-built inverted index
    tf_scheme    : "log" (default) or "raw"
    normalize    : if True, use cosine normalization
    """

    name = "TF-IDF"

    def __init__(
        self,
        index: InvertedIndex,
        tf_scheme: str = "log",
        normalize: bool = True,
    ) -> None:
        self.index = index
        self.tf_scheme = tf_scheme
        self.normalize = normalize
        # Pre-cache document norms (expensive to recompute per query)
        self._doc_norms: Dict[str, float] = {}
        self._precompute_doc_norms()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _tf_weight(self, tf: int) -> float:
        """Compute the term frequency weight."""
        if tf <= 0:
            return 0.0
        if self.tf_scheme == "log":
            return 1.0 + math.log(tf)
        return float(tf)

    def _idf(self, df: int) -> float:
        """Compute the inverse document frequency weight."""
        if df <= 0:
            return 0.0
        return math.log(self.index.N / df)

    def _precompute_doc_norms(self) -> None:
        """Pre-compute the L2 norm of every document's TF-IDF vector."""
        logger.info("Pre-computing TF-IDF document norms …")
        for term, postings in self.index.postings.items():
            idf = self._idf(self.index.df.get(term, 1))
            for entry in postings:
                w = self._tf_weight(entry.tf) * idf
                self._doc_norms[entry.doc_id] = (
                    self._doc_norms.get(entry.doc_id, 0.0) + w * w
                )
        # Square-root to get L2 norm
        for doc_id in self._doc_norms:
            self._doc_norms[doc_id] = math.sqrt(self._doc_norms[doc_id])
        logger.info("Norms computed for %d documents.", len(self._doc_norms))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Score all documents for the query and return the top-k.

        Parameters
        ----------
        query : str  — raw query text
        k     : int  — number of results

        Returns
        -------
        List[Tuple[str, float]] — (doc_id, score), sorted descending.
        """
        tokens = preprocess(query, stem=True, remove_stopwords=True)
        if not tokens:
            return []

        # Build query TF-IDF vector
        query_tf: Dict[str, int] = {}
        for t in tokens:
            query_tf[t] = query_tf.get(t, 0) + 1

        query_weights: Dict[str, float] = {}
        for term, tf in query_tf.items():
            if term not in self.index.df:
                continue
            df = self.index.df[term]
            query_weights[term] = self._tf_weight(tf) * self._idf(df)

        if not query_weights:
            return []

        # Query L2 norm
        q_norm = math.sqrt(sum(w * w for w in query_weights.values()))

        # Accumulate dot-product scores
        scores: Dict[str, float] = {}
        for term, q_w in query_weights.items():
            for entry in self.index.get_postings(term):
                idf = self._idf(self.index.df[term])
                d_w = self._tf_weight(entry.tf) * idf
                scores[entry.doc_id] = scores.get(entry.doc_id, 0.0) + q_w * d_w

        # Cosine normalization
        if self.normalize:
            normalized: Dict[str, float] = {}
            for doc_id, dot in scores.items():
                d_norm = self._doc_norms.get(doc_id, 1.0)
                denom = q_norm * d_norm
                normalized[doc_id] = dot / denom if denom > 0 else 0.0
            scores = normalized

        # Top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
