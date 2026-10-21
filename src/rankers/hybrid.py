"""
src/rankers/hybrid.py — Hybrid Reciprocal Rank Fusion ranker.

Combines any subset of {tfidf, bm25, dense} ranked lists using RRF:
    RRF_score(d) = Σ_{m} 1 / (k + rank_m(d))

Primary combination: all three methods (as specified in the assignment).
Ablation combinations are also supported and reported.

k = 60 (Cormack et al. 2009 default), configurable.
Each method contributes its top-`candidates_per_method` documents before fusion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import BaseRanker
from fusion import reciprocal_rank_fusion, rrf_with_rank_breakdown

import logging
logger = logging.getLogger(__name__)


class HybridRRFRanker(BaseRanker):
    """
    Hybrid Reciprocal Rank Fusion ranker.

    Parameters
    ----------
    rankers               : dict mapping method_name → BaseRanker instance
                            e.g. {"tfidf": tfidf_ranker, "bm25": bm25_ranker, "dense": dense_ranker}
    k                     : RRF constant (default 60)
    candidates_per_method : number of top candidates to retrieve from each method
                            before fusion (default 100)
    """

    name = "Hybrid-RRF"

    def __init__(
        self,
        rankers: Dict[str, BaseRanker],
        k: int = 60,
        candidates_per_method: int = 100,
    ) -> None:
        if not rankers:
            raise ValueError("At least one ranker must be provided.")
        self.rankers = rankers
        self.k = k
        self.candidates_per_method = candidates_per_method
        self._method_names = list(rankers.keys())
        self.name = f"Hybrid-RRF({'+'.join(self._method_names)})"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents via RRF fusion of constituent rankers.

        Parameters
        ----------
        query : str  — raw query text
        k     : int  — final number of results to return

        Returns
        -------
        List[Tuple[str, float]] — (doc_id, rrf_score), sorted descending.
        """
        # Step 1: Get top-C candidates from each method
        ranked_lists: List[List[Tuple[str, float]]] = []
        for method_name, ranker in self.rankers.items():
            results = ranker.search(query, k=self.candidates_per_method)
            ranked_lists.append(results)
            logger.debug("Method '%s' returned %d candidates.", method_name, len(results))

        # Step 2: RRF fusion
        fused = reciprocal_rank_fusion(ranked_lists, k=self.k)

        return fused[:k]

    def search_with_breakdown(
        self,
        query: str,
        k: int = 10,
        top_n_explain: int = 20,
    ) -> Tuple[List[Tuple[str, float]], List[Dict]]:
        """
        Search and also return a per-document rank breakdown for the explainer.

        Returns
        -------
        (ranked_results, breakdown)
          - ranked_results : List[Tuple[str, float]]
          - breakdown      : List[Dict] with keys doc_id, rrf_score, ranks, contributions
        """
        ranked_lists: List[List[Tuple[str, float]]] = []
        for ranker in self.rankers.values():
            ranked_lists.append(ranker.search(query, k=self.candidates_per_method))

        breakdown = rrf_with_rank_breakdown(
            ranked_lists,
            method_names=self._method_names,
            k=self.k,
            top_n=max(top_n_explain, k),
        )

        ranked_results = [
            (entry["doc_id"], entry["rrf_score"])
            for entry in breakdown[:k]
        ]

        return ranked_results, breakdown[:top_n_explain]

    # ------------------------------------------------------------------
    # Ablation helper
    # ------------------------------------------------------------------

    @classmethod
    def from_subset(
        cls,
        all_rankers: Dict[str, BaseRanker],
        subset: List[str],
        k: int = 60,
        candidates_per_method: int = 100,
    ) -> "HybridRRFRanker":
        """
        Create a HybridRRFRanker using only the named subset of rankers.

        Parameters
        ----------
        all_rankers : full dict of rankers
        subset      : list of method names to include
        """
        selected = {name: all_rankers[name] for name in subset if name in all_rankers}
        if not selected:
            raise ValueError(f"None of {subset} found in available rankers: {list(all_rankers)}")
        return cls(selected, k=k, candidates_per_method=candidates_per_method)
