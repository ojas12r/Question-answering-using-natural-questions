"""
src/data.py — Dataset download, subsetting, and statistics for NQ Evidence Retrieval.

Pipeline
--------
1.  Try to load the BEIR NQ corpus + queries + qrels via `ir_datasets`.
2.  Fall back to HuggingFace `datasets` if `ir_datasets` is unavailable.
3.  Build a reproducible ~100 k-passage subset:
      - Keep ALL corpus passages that appear in test qrels.
      - Fill remainder with random distractor passages (seed 42).
4.  Save as Parquet to data/processed/{corpus,queries,qrels}.parquet.
5.  Print dataset statistics.

Usage
-----
    python src/data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---- project imports -------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, set_seed, ensure_dir, get_project_root

logger = get_logger(__name__)


# ===========================================================================
# Loaders
# ===========================================================================

def _load_via_ir_datasets(dataset_name: str) -> tuple[
    list[dict], list[dict], list[dict]
]:
    """
    Load corpus, queries, and qrels from ir_datasets (BEIR NQ test split).

    Returns
    -------
    corpus  : list of {id, title, text}
    queries : list of {id, text}
    qrels   : list of {query_id, corpus_id, score}
    """
    import ir_datasets  # type: ignore

    logger.info("Loading dataset via ir_datasets: %s", dataset_name)
    dataset = ir_datasets.load(dataset_name)

    logger.info("Streaming corpus …")
    corpus = [
        {"id": doc.doc_id, "title": getattr(doc, "title", ""), "text": doc.text}
        for doc in dataset.docs_iter()
    ]
    logger.info("  corpus passages: %d", len(corpus))

    logger.info("Streaming queries …")
    queries = [{"id": q.query_id, "text": q.text} for q in dataset.queries_iter()]
    logger.info("  queries: %d", len(queries))

    logger.info("Streaming qrels …")
    qrels = [
        {"query_id": qr.query_id, "corpus_id": qr.doc_id, "score": qr.relevance}
        for qr in dataset.qrels_iter()
    ]
    logger.info("  qrels: %d", len(qrels))

    return corpus, queries, qrels


def _load_via_huggingface(cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load corpus, queries, and qrels from HuggingFace datasets (BEIR NQ).

    Returns
    -------
    corpus  : list of {id, title, text}
    queries : list of {id, text}
    qrels   : list of {query_id, corpus_id, score}
    """
    from datasets import load_dataset  # type: ignore

    hf_corpus = cfg["dataset"]["hf_corpus"]
    hf_qrels = cfg["dataset"]["hf_qrels"]

    logger.info("Loading corpus via HuggingFace: %s (corpus split) …", hf_corpus)
    corpus_ds = load_dataset(hf_corpus, "corpus", split="corpus", trust_remote_code=True)
    corpus = [
        {"id": row["_id"], "title": row.get("title", ""), "text": row["text"]}
        for row in corpus_ds
    ]
    logger.info("  corpus passages: %d", len(corpus))

    logger.info("Loading queries via HuggingFace: %s (queries split) …", hf_corpus)
    queries_ds = load_dataset(hf_corpus, "queries", split="queries", trust_remote_code=True)
    queries = [{"id": row["_id"], "text": row["text"]} for row in queries_ds]
    logger.info("  queries (all splits): %d", len(queries))

    logger.info("Loading qrels via HuggingFace: %s …", hf_qrels)
    qrels_ds = load_dataset(hf_qrels, split="test", trust_remote_code=True)
    qrels = [
        {
            "query_id": row["query-id"],
            "corpus_id": row["corpus-id"],
            "score": row["score"],
        }
        for row in qrels_ds
    ]
    logger.info("  qrels (test): %d", len(qrels))

    # Filter queries to only those in test qrels
    test_qids = {q["query_id"] for q in qrels}
    queries = [q for q in queries if q["id"] in test_qids]
    logger.info("  queries with test qrels: %d", len(queries))

    return corpus, queries, qrels


# ===========================================================================
# Subset construction
# ===========================================================================

def build_subset(
    corpus: list[dict],
    queries: list[dict],
    qrels: list[dict],
    target_size: int,
    max_queries: int | None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a reproducible subset of the BEIR NQ corpus.

    Strategy
    --------
    1.  Optionally cap the number of test queries (random sample, seed=42).
    2.  Identify all relevant corpus_ids for the selected queries.
    3.  Fill the remainder of `target_size` with random distractors.

    Parameters
    ----------
    corpus      : full corpus list of {id, title, text}
    queries     : test queries list of {id, text}
    qrels       : qrels list of {query_id, corpus_id, score}
    target_size : desired approximate total number of passages
    max_queries : if not None, randomly sample this many queries
    seed        : random seed for reproducibility

    Returns
    -------
    (corpus_df, queries_df, qrels_df) as pandas DataFrames.
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Cap queries -------------------------------------------------------
    query_df_full = pd.DataFrame(queries)
    qrels_df_full = pd.DataFrame(qrels)

    if max_queries is not None and max_queries < len(query_df_full):
        # Sample max_queries query ids; ensure each has ≥1 qrel
        qids_with_qrels = set(qrels_df_full["query_id"].unique())
        eligible_qids = [q["id"] for q in queries if q["id"] in qids_with_qrels]
        chosen_qids = set(
            rng.choice(eligible_qids, size=min(max_queries, len(eligible_qids)), replace=False)
        )
        query_df = query_df_full[query_df_full["id"].isin(chosen_qids)].reset_index(drop=True)
        logger.info("Sampled %d queries from %d eligible.", len(query_df), len(eligible_qids))
    else:
        query_df = query_df_full.copy()

    # ---- 2. Filter qrels to chosen queries ------------------------------------
    qrels_df = qrels_df_full[qrels_df_full["query_id"].isin(query_df["id"])].reset_index(drop=True)
    relevant_ids: set[str] = set(qrels_df["corpus_id"].unique())
    logger.info("Relevant passages to retain: %d", len(relevant_ids))

    # ---- 3. Build corpus_df ---------------------------------------------------
    corpus_df_full = pd.DataFrame(corpus)
    relevant_corpus = corpus_df_full[corpus_df_full["id"].isin(relevant_ids)]

    distractor_pool = corpus_df_full[~corpus_df_full["id"].isin(relevant_ids)]
    n_distractors = max(0, target_size - len(relevant_corpus))

    if n_distractors > 0 and len(distractor_pool) > 0:
        n_sample = min(n_distractors, len(distractor_pool))
        distractor_sample = distractor_pool.sample(n=n_sample, random_state=seed)
        logger.info(
            "Adding %d distractor passages (pool size: %d).",
            n_sample,
            len(distractor_pool),
        )
        corpus_df = pd.concat([relevant_corpus, distractor_sample], ignore_index=True)
    else:
        corpus_df = relevant_corpus.reset_index(drop=True)

    corpus_df = corpus_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Verify: no relevant passage is lost
    retained_ids = set(corpus_df["id"].unique())
    missing = relevant_ids - retained_ids
    assert len(missing) == 0, f"BUG: {len(missing)} relevant passages missing from subset!"

    return corpus_df, query_df, qrels_df


# ===========================================================================
# Dataset statistics
# ===========================================================================

def print_dataset_stats(
    corpus_df: pd.DataFrame,
    query_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute and print dataset statistics table. Return dict of stats."""
    import nltk

    def avg_len(texts: pd.Series) -> float:
        return texts.apply(lambda t: len(str(t).split())).mean()

    avg_passage_len = avg_len(corpus_df["text"])
    avg_title_len = avg_len(corpus_df.get("title", pd.Series([""] * len(corpus_df))))
    avg_query_len = avg_len(query_df["text"])

    rel_per_query = qrels_df.groupby("query_id")["corpus_id"].count()

    stats = {
        "n_passages": len(corpus_df),
        "n_queries": len(query_df),
        "n_qrels": len(qrels_df),
        "avg_passage_words": round(avg_passage_len, 1),
        "avg_title_words": round(avg_title_len, 1),
        "avg_query_words": round(avg_query_len, 1),
        "avg_relevant_per_query": round(rel_per_query.mean(), 2),
        "max_relevant_per_query": int(rel_per_query.max()),
        "min_relevant_per_query": int(rel_per_query.min()),
    }

    print("\n" + "=" * 55)
    print("  DATASET STATISTICS")
    print("=" * 55)
    print(f"  {'Corpus passages':<35} {stats['n_passages']:>10,}")
    print(f"  {'Queries (test)':<35} {stats['n_queries']:>10,}")
    print(f"  {'QRels pairs':<35} {stats['n_qrels']:>10,}")
    print(f"  {'Avg passage length (words)':<35} {stats['avg_passage_words']:>10.1f}")
    print(f"  {'Avg title length (words)':<35} {stats['avg_title_words']:>10.1f}")
    print(f"  {'Avg query length (words)':<35} {stats['avg_query_words']:>10.1f}")
    print(f"  {'Avg relevant passages / query':<35} {stats['avg_relevant_per_query']:>10.2f}")
    print(f"  {'Max relevant passages / query':<35} {stats['max_relevant_per_query']:>10,}")
    print(f"  {'Min relevant passages / query':<35} {stats['min_relevant_per_query']:>10,}")
    print("=" * 55 + "\n")

    return stats


# ===========================================================================
# Persistence
# ===========================================================================

def save_processed(
    corpus_df: pd.DataFrame,
    query_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save DataFrames as Parquet files."""
    ensure_dir(out_dir)
    corpus_df.to_parquet(out_dir / "corpus.parquet", index=False)
    query_df.to_parquet(out_dir / "queries.parquet", index=False)
    qrels_df.to_parquet(out_dir / "qrels.parquet", index=False)
    logger.info("Saved parquet files to %s", out_dir)


def load_processed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved processed Parquet files."""
    corpus_df = pd.read_parquet(data_dir / "corpus.parquet")
    query_df = pd.read_parquet(data_dir / "queries.parquet")
    qrels_df = pd.read_parquet(data_dir / "qrels.parquet")
    return corpus_df, query_df, qrels_df


def processed_files_exist(data_dir: Path) -> bool:
    """Return True if all three parquet files already exist."""
    return all(
        (data_dir / name).exists()
        for name in ["corpus.parquet", "queries.parquet", "qrels.parquet"]
    )


# ===========================================================================
# Main entry point
# ===========================================================================

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    root = get_project_root()
    data_dir = root / cfg["dataset"]["data_dir"]
    ensure_dir(data_dir)

    # ---- Skip if already built -----------------------------------------------
    if processed_files_exist(data_dir):
        logger.info(
            "Processed files already exist in %s — loading and printing stats.", data_dir
        )
        corpus_df, query_df, qrels_df = load_processed(data_dir)
        print_dataset_stats(corpus_df, query_df, qrels_df)
        return

    t0 = time.perf_counter()

    # ---- Try ir_datasets first -----------------------------------------------
    corpus, queries, qrels = None, None, None
    try:
        corpus, queries, qrels = _load_via_ir_datasets(cfg["dataset"]["name"])
    except Exception as exc:
        logger.warning("ir_datasets failed (%s). Falling back to HuggingFace.", exc)
        try:
            corpus, queries, qrels = _load_via_huggingface(cfg)
        except Exception as exc2:
            logger.error("HuggingFace fallback also failed: %s", exc2)
            raise RuntimeError(
                "Could not load BEIR NQ dataset via ir_datasets or HuggingFace. "
                "Check your internet connection and installed packages."
            ) from exc2

    load_time = time.perf_counter() - t0
    logger.info("Dataset loaded in %.1f s.", load_time)

    # ---- Build subset --------------------------------------------------------
    t1 = time.perf_counter()
    corpus_df, query_df, qrels_df = build_subset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        target_size=cfg["dataset"]["target_corpus_size"],
        max_queries=cfg["dataset"].get("max_queries"),
        seed=cfg["seed"],
    )
    subset_time = time.perf_counter() - t1
    logger.info("Subset built in %.1f s.", subset_time)

    # ---- Print stats ---------------------------------------------------------
    stats = print_dataset_stats(corpus_df, query_df, qrels_df)

    # ---- Save ----------------------------------------------------------------
    save_processed(corpus_df, query_df, qrels_df, data_dir)

    # ---- Save stats as JSON for report/facts.md ------------------------------
    stats_path = data_dir / "dataset_stats.json"
    with stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Stats saved to %s", stats_path)

    logger.info(
        "Total data phase time: %.1f s  (load=%.1f s, subset=%.1f s)",
        time.perf_counter() - t0,
        load_time,
        subset_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and subset BEIR NQ dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
