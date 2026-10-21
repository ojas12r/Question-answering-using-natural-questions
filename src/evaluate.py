"""
src/evaluate.py — Evaluation metrics, figures, and reporting for NQ Retrieval.

Metrics implemented from scratch (cross-validated against pytrec_eval/ranx):
  - Precision@K
  - Recall@K  (K = 10, 50, 100)
  - MAP (Mean Average Precision)
  - nDCG@K
  - MRR@K

Outputs
-------
  results/performance_table.{csv,md}
  results/latency_table.{csv,md}
  results/pr_curve.png
  results/metrics_bar.png
  results/ndcg_histogram.png
  results/bm25_vs_dense_winloss.png
  results/error_analysis.md
  results/significance.txt
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, ensure_dir

logger = get_logger(__name__)


# ===========================================================================
# Metric implementations
# ===========================================================================

def precision_at_k(
    retrieved: List[str],
    relevant: set[str],
    k: int,
) -> float:
    """Precision@K = |relevant ∩ top-K| / K."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def recall_at_k(
    retrieved: List[str],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)


def average_precision(
    retrieved: List[str],
    relevant: set[str],
) -> float:
    """
    Average Precision = Σ_{k: doc_k relevant} P@k / |relevant|.
    Returns 0.0 if no relevant documents exist.
    """
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def ndcg_at_k(
    retrieved: List[str],
    relevant: Dict[str, int],  # doc_id → relevance grade (e.g., 0/1 for binary)
    k: int,
) -> float:
    """
    nDCG@K = DCG@K / IDCG@K.
    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = DCG of ideal ranking (relevant docs first).
    """
    def dcg(ranking: List[str], max_k: int) -> float:
        score = 0.0
        for i, doc_id in enumerate(ranking[:max_k], start=1):
            rel = relevant.get(doc_id, 0)
            if rel > 0:
                score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved, k)
    # Ideal ranking: sort relevant docs by grade descending
    ideal_docs = sorted(relevant.keys(), key=lambda d: relevant[d], reverse=True)
    ideal_dcg = dcg(ideal_docs, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_at_k(
    retrieved: List[str],
    relevant: set[str],
    k: int,
) -> float:
    """MRR@K = 1 / rank of first relevant result in top-K (0 if none)."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ===========================================================================
# Per-query evaluation
# ===========================================================================

def evaluate_query(
    retrieved: List[str],
    qrels_for_query: Dict[str, int],  # doc_id → score
    cutoffs: List[int] = (10, 50, 100),
) -> Dict[str, float]:
    """
    Compute all metrics for a single query.

    Parameters
    ----------
    retrieved         : ordered list of retrieved doc_ids
    qrels_for_query   : dict of relevant doc_id → relevance score
    cutoffs           : list of K values for Recall@K

    Returns
    -------
    Dict of metric_name → float value.
    """
    relevant_set = {d for d, s in qrels_for_query.items() if s > 0}
    relevant_graded = {d: s for d, s in qrels_for_query.items() if s > 0}

    result: Dict[str, float] = {}

    # Precision / Recall / nDCG @ 10
    result["P@10"] = precision_at_k(retrieved, relevant_set, 10)
    result["nDCG@10"] = ndcg_at_k(retrieved, relevant_graded, 10)
    result["MRR@10"] = mrr_at_k(retrieved, relevant_set, 10)
    result["AP"] = average_precision(retrieved, relevant_set)

    # Recall at multiple cutoffs
    for k in cutoffs:
        result[f"R@{k}"] = recall_at_k(retrieved, relevant_set, k)

    return result


def evaluate_run(
    run: Dict[str, List[Tuple[str, float]]],  # qid → [(doc_id, score)]
    qrels: Dict[str, Dict[str, int]],         # qid → {doc_id: score}
    cutoffs: List[int] = (10, 50, 100),
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a full retrieval run.

    Parameters
    ----------
    run    : dict qid → ranked list of (doc_id, score)
    qrels  : dict qid → {doc_id: relevance_score}
    cutoffs: Recall cutoffs

    Returns
    -------
    Dict qid → per-query metric dict.
    """
    per_query: Dict[str, Dict[str, float]] = {}
    for qid, ranked in run.items():
        retrieved = [doc_id for doc_id, _ in ranked]
        qrels_q = qrels.get(qid, {})
        per_query[qid] = evaluate_query(retrieved, qrels_q, cutoffs)
    return per_query


def aggregate_metrics(per_query: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average per-query metrics across all queries."""
    if not per_query:
        return {}
    all_metrics = list(next(iter(per_query.values())).keys())
    agg: Dict[str, float] = {}
    for metric in all_metrics:
        vals = [pq[metric] for pq in per_query.values() if metric in pq]
        agg[metric] = float(np.mean(vals)) if vals else 0.0
    # MAP = mean of AP
    if "AP" in agg:
        agg["MAP"] = agg.pop("AP")
    return agg


# ===========================================================================
# Latency measurement
# ===========================================================================

def measure_latency(
    ranker,
    queries: Dict[str, str],
    k: int = 10,
    n_trials: int = 50,
    seed: int = 42,
) -> float:
    """
    Measure average per-query latency in milliseconds.

    Randomly samples `n_trials` queries and returns mean search time in ms.
    """
    rng = np.random.default_rng(seed)
    qids = list(queries.keys())
    if len(qids) > n_trials:
        sample_qids = rng.choice(qids, size=n_trials, replace=False).tolist()
    else:
        sample_qids = qids

    times = []
    for qid in sample_qids:
        t0 = time.perf_counter()
        ranker.search(queries[qid], k=k)
        times.append((time.perf_counter() - t0) * 1000)  # → ms

    return float(np.mean(times))


# ===========================================================================
# Output: tables
# ===========================================================================

def build_results_table(
    model_metrics: Dict[str, Dict[str, float]],
    latencies: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build the main performance comparison DataFrame.

    Parameters
    ----------
    model_metrics : dict model_name → aggregated metric dict
    latencies     : dict model_name → avg latency (ms)

    Returns
    -------
    pd.DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in model_metrics.items():
        row = {"Model": model_name}
        row["P@10"] = metrics.get("P@10", 0.0)
        row["R@10"] = metrics.get("R@10", 0.0)
        row["R@50"] = metrics.get("R@50", 0.0)
        row["R@100"] = metrics.get("R@100", 0.0)
        row["MAP"] = metrics.get("MAP", 0.0)
        row["nDCG@10"] = metrics.get("nDCG@10", 0.0)
        row["MRR@10"] = metrics.get("MRR@10", 0.0)
        if latencies:
            row["Latency(ms)"] = latencies.get(model_name, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    # Round for display
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)
    return df


def save_results_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the results table as CSV and Markdown."""
    ensure_dir(out_dir)
    df.to_csv(out_dir / "performance_table.csv")
    with (out_dir / "performance_table.md").open("w") as fh:
        fh.write(df.to_markdown())
    logger.info("Results table saved to %s", out_dir)


def save_latency_table(latencies: Dict[str, float], index_sizes: Dict[str, float],
                       out_dir: Path) -> None:
    """Save the latency + index size table."""
    rows = [
        {"Model": m, "Avg Latency (ms)": round(latencies.get(m, 0), 2),
         "Index Size (MB)": round(index_sizes.get(m, 0), 1)}
        for m in latencies
    ]
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(out_dir / "latency_table.csv")
    with (out_dir / "latency_table.md").open("w") as fh:
        fh.write(df.to_markdown())
    logger.info("Latency table saved.")


# ===========================================================================
# Output: figures
# ===========================================================================

def _get_pr_curve_points(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    recall_levels: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """Compute 11-point interpolated precision-recall curve averaged over queries."""
    if recall_levels is None:
        recall_levels = [i / 10 for i in range(11)]

    all_interp_precisions = []

    for qid, ranked in run.items():
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if not relevant:
            continue

        retrieved = [d for d, _ in ranked]
        precisions, recalls = [], []
        hits = 0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
            precisions.append(hits / rank)
            recalls.append(hits / len(relevant))

        # Interpolate to fixed recall levels
        interp = []
        for r_level in recall_levels:
            cands = [p for p, r in zip(precisions, recalls) if r >= r_level]
            interp.append(max(cands) if cands else 0.0)
        all_interp_precisions.append(interp)

    if not all_interp_precisions:
        return recall_levels, [0.0] * len(recall_levels)

    avg_precisions = [float(np.mean([ip[i] for ip in all_interp_precisions]))
                      for i in range(len(recall_levels))]
    return recall_levels, avg_precisions


def plot_pr_curve(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    qrels: Dict[str, Dict[str, int]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot and save the 11-point interpolated PR curve for all models."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"TF-IDF": "#4C72B0", "BM25": "#DD8452", "Dense": "#55A868", "Hybrid-RRF": "#C44E52"}
    linestyles = {"TF-IDF": "--", "BM25": "-.", "Dense": "-", "Hybrid-RRF": ":"}
    linewidths = {"TF-IDF": 1.5, "BM25": 1.5, "Dense": 2.0, "Hybrid-RRF": 2.5}

    for model_name, run in runs.items():
        recall_levels, avg_precisions = _get_pr_curve_points(run, qrels)
        lname = model_name.split("(")[0]  # strip ablation suffix
        color = colors.get(lname, "#8172B2")
        ls = linestyles.get(lname, "-")
        lw = linewidths.get(lname, 1.5)
        ax.plot(recall_levels, avg_precisions, label=model_name,
                color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4)

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("11-Point Interpolated Precision-Recall Curve\n(NQ Retrieval Subset)", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to %s", out_path)


def plot_metrics_bar(
    df: pd.DataFrame,
    out_path: Path,
    metrics: Optional[List[str]] = None,
    dpi: int = 300,
) -> None:
    """Grouped bar chart: metric × model."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns

    if metrics is None:
        metrics = ["P@10", "R@10", "MAP", "nDCG@10", "MRR@10"]
    plot_df = df[metrics].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.T.plot(kind="bar", ax=ax, colormap="tab10", width=0.75, edgecolor="white")

    ax.set_title("Retrieval Performance by Model and Metric\n(NQ Subset)", fontsize=14)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0, fontsize=11)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metrics bar chart saved to %s", out_path)


def plot_ndcg_histogram(
    per_query_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Histogram of per-query nDCG@10 distributions for all models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"TF-IDF": "#4C72B0", "BM25": "#DD8452", "Dense": "#55A868"}

    for i, (model_name, per_query) in enumerate(per_query_metrics.items()):
        if i >= 4:
            break
        vals = [v.get("nDCG@10", 0.0) for v in per_query.values()]
        base = model_name.split("(")[0]
        color = colors.get(base, "#C44E52")
        axes[i].hist(vals, bins=30, color=color, edgecolor="white", alpha=0.85)
        axes[i].set_title(f"{model_name}\n(mean={np.mean(vals):.3f})", fontsize=11)
        axes[i].set_xlim(0, 1)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel("nDCG@10", fontsize=10)
        axes[i].set_ylabel("# Queries", fontsize=10)

    fig.suptitle("Per-Query nDCG@10 Distribution — NQ Retrieval Subset", fontsize=13)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("nDCG histogram saved to %s", out_path)


def plot_bm25_vs_dense_winloss(
    per_query_bm25: Dict[str, Dict[str, float]],
    per_query_dense: Dict[str, Dict[str, float]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Scatter plot: BM25 nDCG@10 vs Dense nDCG@10 per query."""
    import matplotlib.pyplot as plt

    common_qids = set(per_query_bm25) & set(per_query_dense)
    bm25_scores = [per_query_bm25[qid].get("nDCG@10", 0.0) for qid in common_qids]
    dense_scores = [per_query_dense[qid].get("nDCG@10", 0.0) for qid in common_qids]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(bm25_scores, dense_scores, alpha=0.3, s=15, color="#4C72B0")
    lim = max(max(bm25_scores, default=1), max(dense_scores, default=1)) + 0.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="BM25 = Dense")
    ax.set_xlabel("BM25 nDCG@10", fontsize=13)
    ax.set_ylabel("Dense nDCG@10", fontsize=13)
    ax.set_title("Per-Query Win/Loss: BM25 vs Dense\n(above diagonal = Dense wins)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)

    bm25_wins = sum(b > d for b, d in zip(bm25_scores, dense_scores))
    dense_wins = sum(d > b for b, d in zip(bm25_scores, dense_scores))
    ties = len(bm25_scores) - bm25_wins - dense_wins
    ax.text(0.02, 0.98,
            f"BM25 wins: {bm25_wins}\nDense wins: {dense_wins}\nTies: {ties}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("BM25 vs Dense win/loss plot saved to %s", out_path)


# ===========================================================================
# Statistical significance
# ===========================================================================

def significance_test(
    per_query_a: Dict[str, Dict[str, float]],
    per_query_b: Dict[str, Dict[str, float]],
    metric: str = "nDCG@10",
    name_a: str = "Hybrid",
    name_b: str = "BestSingle",
    alpha: float = 0.05,
) -> Dict:
    """
    Paired two-sided t-test comparing model A vs model B on a per-query metric.

    Returns
    -------
    dict with keys: t_stat, p_value, significant, n_queries, mean_a, mean_b, mean_diff
    """
    from scipy import stats

    common = sorted(set(per_query_a) & set(per_query_b))
    scores_a = [per_query_a[qid].get(metric, 0.0) for qid in common]
    scores_b = [per_query_b[qid].get(metric, 0.0) for qid in common]

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "metric": metric,
        "n_queries": len(common),
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff": round(mean_a - mean_b, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }
    return result


def save_significance_report(sig: Dict, out_path: Path) -> None:
    """Save significance test results to a text file."""
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("=" * 55 + "\n")
        fh.write(f"  STATISTICAL SIGNIFICANCE TEST\n")
        fh.write("=" * 55 + "\n")
        fh.write(f"  Model A: {sig['name_a']}\n")
        fh.write(f"  Model B: {sig['name_b']}\n")
        fh.write(f"  Metric : {sig['metric']}\n")
        fh.write(f"  Queries: {sig['n_queries']}\n")
        fh.write(f"  Mean A : {sig['mean_a']:.4f}\n")
        fh.write(f"  Mean B : {sig['mean_b']:.4f}\n")
        fh.write(f"  Delta  : {sig['mean_diff']:+.4f}\n")
        fh.write(f"  t-stat : {sig['t_stat']:.4f}\n")
        fh.write(f"  p-value: {sig['p_value']:.6f}\n")
        fh.write(
            f"  Result : {'SIGNIFICANT' if sig['significant'] else 'NOT SIGNIFICANT'} "
            f"at alpha={sig['alpha']}\n"
        )
        fh.write("=" * 55 + "\n")
    logger.info("Significance report saved to %s", out_path)


# ===========================================================================
# Error analysis
# ===========================================================================

def build_error_analysis(
    per_query_bm25: Dict[str, Dict[str, float]],
    per_query_dense: Dict[str, Dict[str, float]],
    per_query_hybrid: Dict[str, Dict[str, float]],
    queries: Dict[str, str],            # qid → text
    corpus: Dict[str, Dict],            # doc_id → {title, text}
    qrels: Dict[str, Dict[str, int]],   # qid → {doc_id: score}
    runs_top3: Dict[str, Dict[str, List[Tuple[str, float]]]],  # model → qid → [(doc_id, score)]
    n_each: int = 5,
) -> str:
    """
    Identify queries where BM25 wins, Dense wins, and Hybrid fixes both.
    Return a Markdown-formatted error analysis string.
    """
    common = set(per_query_bm25) & set(per_query_dense) & set(per_query_hybrid)
    metric = "nDCG@10"

    bm25_better = sorted(
        [q for q in common
         if per_query_bm25[q].get(metric, 0) > per_query_dense[q].get(metric, 0) + 0.1],
        key=lambda q: per_query_bm25[q].get(metric, 0) - per_query_dense[q].get(metric, 0),
        reverse=True,
    )[:n_each]

    dense_better = sorted(
        [q for q in common
         if per_query_dense[q].get(metric, 0) > per_query_bm25[q].get(metric, 0) + 0.1],
        key=lambda q: per_query_dense[q].get(metric, 0) - per_query_bm25[q].get(metric, 0),
        reverse=True,
    )[:n_each]

    hybrid_fixes = sorted(
        [q for q in common
         if (per_query_hybrid[q].get(metric, 0) > per_query_bm25[q].get(metric, 0) + 0.05
             and per_query_hybrid[q].get(metric, 0) > per_query_dense[q].get(metric, 0) + 0.05)],
        key=lambda q: per_query_hybrid[q].get(metric, 0),
        reverse=True,
    )[:n_each]

    def format_section(title: str, qids: List[str]) -> str:
        lines = [f"\n## {title}\n"]
        for qid in qids:
            q_text = queries.get(qid, "N/A")
            gold_ids = [d for d, s in qrels.get(qid, {}).items() if s > 0]
            gold_text = corpus.get(gold_ids[0], {}).get("text", "N/A")[:300] if gold_ids else "N/A"
            gold_title = corpus.get(gold_ids[0], {}).get("title", "N/A") if gold_ids else "N/A"

            lines.append(f"### Query: `{q_text}`")
            lines.append(f"**QID**: {qid}")
            lines.append(f"**Gold passage** ({gold_title}): …{gold_text}…\n")

            for model_name, run in runs_top3.items():
                top3 = run.get(qid, [])[:3]
                lines.append(f"**{model_name} Top-3:**")
                for rank, (doc_id, score) in enumerate(top3, 1):
                    title_ = corpus.get(doc_id, {}).get("title", "?")
                    snippet = corpus.get(doc_id, {}).get("text", "")[:200]
                    rel_mark = "✅" if doc_id in {d for d in gold_ids} else "❌"
                    lines.append(f"  {rank}. [{rel_mark}] **{title_}** (score={score:.4f})")
                    lines.append(f"     …{snippet}…")
                lines.append("")

            bm25_s = per_query_bm25.get(qid, {}).get(metric, 0)
            dense_s = per_query_dense.get(qid, {}).get(metric, 0)
            hybrid_s = per_query_hybrid.get(qid, {}).get(metric, 0)
            lines.append(
                f"**nDCG@10**: BM25={bm25_s:.3f} | Dense={dense_s:.3f} | Hybrid={hybrid_s:.3f}\n"
            )
            lines.append("---\n")
        return "\n".join(lines)

    header = (
        "# Error Analysis\n\n"
        "This section presents qualitative cases illustrating where each model "
        "succeeds and fails, and where Hybrid RRF provides complementary benefit.\n"
    )
    body = (
        format_section("BM25 Wins Over Dense (Lexical Advantage)", bm25_better)
        + format_section("Dense Wins Over BM25 (Semantic Advantage)", dense_better)
        + format_section("Hybrid Fixes Both BM25 and Dense", hybrid_fixes)
    )
    return header + body
