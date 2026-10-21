"""
src/rankers/base.py — Abstract base class for all ranking models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseRanker(ABC):
    """
    Common interface for all retrieval models.

    Every ranker must implement `search(query, k)` which returns a ranked list
    of (doc_id, score) tuples sorted by score descending.
    """

    name: str = "BaseRanker"

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank documents for a query.

        Parameters
        ----------
        query : str
            Natural-language query string (raw text; preprocessing handled internally).
        k     : int
            Number of results to return.

        Returns
        -------
        List[Tuple[str, float]]
            List of (doc_id, score) pairs, sorted by score descending, length ≤ k.
        """
        ...

    def batch_search(
        self, queries: Dict[str, str], k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search for multiple queries.

        Parameters
        ----------
        queries : Dict[str, str]
            Mapping query_id → query_text.
        k       : int
            Number of results per query.

        Returns
        -------
        Dict[str, List[Tuple[str, float]]]
            Mapping query_id → ranked list of (doc_id, score).
        """
        return {qid: self.search(qtext, k) for qid, qtext in queries.items()}
