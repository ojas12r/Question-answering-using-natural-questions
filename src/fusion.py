"""
src/fusion.py — Reciprocal Rank Fusion (RRF) core math.

Formula (Cormack, Clarke & Buettcher, 2009):
    RRF_score(d) = Σ_{m ∈ methods} 1 / (k + rank_m(d))

where rank_m(d) is the 1-based rank of document d in the ranked list produced
by method m. Documents not appearing in a method's top-K candidate list are
simply absent from that method's contribution.

k = 60 is the default value from the original paper; it is configurable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked lists, each a list of (doc_id, score) pairs
                   sorted by score descending. Scores from the input lists are
                   *not* used — only ranks matter.
    k            : RRF constant (default 60, from Cormack et al. 2009).

    Returns
    -------
    List[Tuple[str, float]]
        Merged ranked list of (doc_id, rrf_score) sorted by rrf_score descending.

    Examples
    --------
    >>> lists = [
    ...     [("d1", 0.9), ("d2", 0.7), ("d3", 0.5)],
    ...     [("d2", 0.8), ("d1", 0.6), ("d4", 0.4)],
    ... ]
    >>> reciprocal_rank_fusion(lists, k=60)
    [('d1', 0.032...), ('d2', 0.032...), ...]
    """
    rrf_scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rrf_with_rank_breakdown(
    ranked_lists: List[List[Tuple[str, float]]],
    method_names: List[str],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict]:
    """
    RRF fusion with a per-document rank breakdown (used by the Streamlit explainer).

    Parameters
    ----------
    ranked_lists  : list of ranked lists (one per method), (doc_id, score) pairs
    method_names  : names of the methods (parallel to ranked_lists)
    k             : RRF constant
    top_n         : return detailed breakdown for this many top documents

    Returns
    -------
    List of dicts with keys:
      - doc_id       : str
      - rrf_score    : float
      - ranks        : dict[method_name, int | None]   (None = not in list)
      - contributions: dict[method_name, float]         (1/(k+rank) or 0)
    """
    # Build rank lookup per method
    doc_ranks: Dict[str, Dict[str, Optional[int]]] = {}
    for method_name, ranked_list in zip(method_names, ranked_lists):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
            doc_ranks[doc_id][method_name] = rank

    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}
    for doc_id, method_rank_map in doc_ranks.items():
        total = 0.0
        for method_name in method_names:
            rank = method_rank_map.get(method_name)
            if rank is not None:
                total += 1.0 / (k + rank)
        rrf_scores[doc_id] = total

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    breakdown = []
    for doc_id, rrf_score in sorted_docs:
        ranks = {}
        contributions = {}
        for method_name in method_names:
            rank = doc_ranks.get(doc_id, {}).get(method_name)
            ranks[method_name] = rank
            contributions[method_name] = (1.0 / (k + rank)) if rank is not None else 0.0
        breakdown.append(
            {
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "ranks": ranks,
                "contributions": contributions,
            }
        )

    return breakdown
