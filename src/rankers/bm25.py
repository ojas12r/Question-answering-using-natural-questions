"""
src/rankers/bm25.py — Okapi BM25 ranker.

Scoring formula (Robertson & Zaragoza, 2009)
---------------------------------------------
  IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )   [Robertson smoothed]
  BM25(q,d) = Σ_t IDF(t) * tf(t,d) * (k1 + 1)
                               / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))

Hyperparameter tuning
---------------------
Default k1=1.5, b=0.75. A grid search over k1 ∈ {0.9,1.2,1.5,2.0} and
b ∈ {0.4,0.6,0.75,0.9} is run on a held-out dev set (see run_experiments.py).

Validation
----------
Spearman ρ vs. `rank_bm25` ≥ 0.98 on 50 random queries.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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


class BM25Ranker(BaseRanker):
    """
    Okapi BM25 ranker built on the hand-written inverted index.

    Parameters
    ----------
    index : InvertedIndex — pre-built inverted index
    k1    : float — term frequency saturation parameter (default 1.5)
    b     : float — document length normalization parameter (default 0.75)
    """

    name = "BM25"

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        # Pre-cache IDF values
        self._idf_cache: Dict[str, float] = {}
        self._precompute_idf()

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Robertson smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        return self._idf_cache.get(term, 0.0)

    def _precompute_idf(self) -> None:
        """Pre-compute IDF for all vocabulary terms."""
        N = self.index.N
        for term, df in self.index.df.items():
            self._idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        logger.debug("IDF pre-computed for %d terms.", len(self._idf_cache))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_doc(self, term_tfs: Dict[str, int], tokens: List[str]) -> Dict[str, float]:
        """
        Compute BM25 scores for all documents that share at least one query term.

        Parameters
        ----------
        term_tfs : dict mapping term → query term frequency (not used directly;
                   BM25 standard form treats query as bag-of-words without QTF)
        tokens   : unique query terms (deduplicated list)

        Returns
        -------
        dict[doc_id, score]
        """
        scores: Dict[str, float] = {}
        k1, b, avgdl = self.k1, self.b, self.index.avg_dl

        for term in tokens:
            if term not in self.index.df:
                continue
            idf = self._idf(term)
            for entry in self.index.get_postings(term):
                dl = self.index.get_doc_length(entry.doc_id)
                tf = entry.tf
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else tf + k1
                term_score = idf * numerator / denominator
                scores[entry.doc_id] = scores.get(entry.doc_id, 0.0) + term_score

        return scores

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank documents using BM25 and return top-k.

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

        # Deduplicate while preserving order for query TF (BM25 is typically
        # applied term-by-term over unique query terms; including query TF is
        # a variant — we use the standard formulation without QTF weighting)
        unique_tokens = list(dict.fromkeys(tokens))

        # query_tf not used in standard BM25 scoring
        term_tfs = {t: tokens.count(t) for t in unique_tokens}

        scores = self._score_doc(term_tfs, unique_tokens)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def set_params(self, k1: float, b: float) -> None:
        """Update hyperparameters (and invalidate IDF cache if needed)."""
        self.k1 = k1
        self.b = b
        # IDF does not depend on k1/b, so no need to recompute
        logger.debug("BM25 params updated: k1=%.2f, b=%.2f", k1, b)
