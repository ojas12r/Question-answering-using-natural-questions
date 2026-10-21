"""
src/run_experiments.py — Full experiment orchestrator.

Steps
-----
1. Load processed corpus, queries, qrels from parquet.
2. Load inverted index.
3. BM25 hyperparameter grid search on dev split.
4. Encode corpus with Sentence-Transformer (or load from cache).
5. Run all 4 rankers on test queries (retrieve top-100 per query).
6. Evaluate → aggregate metrics → print performance table.
7. Run ablation study (RRF subsets).
8. Compute latencies.
9. Statistical significance test.
10. Generate all figures.
11. Build error analysis.
12. Save report support files (facts.md, tables).

Usage
-----
    python src/run_experiments.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_logger, load_config, set_seed, ensure_dir, get_project_root
from index import load_index, InvertedIndex
from rankers.tfidf import TFIDFRanker
from rankers.bm25 import BM25Ranker
from rankers.dense import DenseRanker
from rankers.hybrid import HybridRRFRanker
import evaluate as ev

logger = get_logger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed parquet files."""
    corpus_df = pd.read_parquet(data_dir / "corpus.parquet")
    query_df = pd.read_parquet(data_dir / "queries.parquet")
    qrels_df = pd.read_parquet(data_dir / "qrels.parquet")
    return corpus_df, query_df, qrels_df


def build_qrels_dict(qrels_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Convert qrels DataFrame to {qid: {doc_id: score}} dict."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_df.itertuples():
        qrels[row.query_id][row.corpus_id] = int(row.score)
    return dict(qrels)


def split_queries(
    query_df: pd.DataFrame,
    qrels: Dict[str, Dict[str, int]],
    dev_size: int,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Split queries into dev (for BM25 tuning) and test sets."""
    rng = np.random.default_rng(seed)
    all_qids = query_df["id"].tolist()
    rng.shuffle(all_qids)
    dev_qids = set(all_qids[:dev_size])
    test_qids = set(all_qids[dev_size:])

    dev_queries = {q["id"]: q["text"] for _, q in query_df.iterrows() if q["id"] in dev_qids}
    test_queries = {q["id"]: q["text"] for _, q in query_df.iterrows() if q["id"] in test_qids}
    logger.info("Dev queries: %d | Test queries: %d", len(dev_queries), len(test_queries))
    return dev_queries, test_queries


# ===========================================================================
# BM25 grid search
# ===========================================================================

def bm25_grid_search(
    index: InvertedIndex,
    dev_queries: Dict[str, str],
    dev_qrels: Dict[str, Dict[str, int]],
    k1_grid: List[float],
    b_grid: List[float],
    k: int = 10,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Grid search over k1 and b on the dev set, optimising nDCG@10.

    Returns
    -------
    (best_k1, best_b, grid_df)
    """
    logger.info("BM25 grid search: %d × %d = %d combos on %d dev queries …",
                len(k1_grid), len(b_grid), len(k1_grid) * len(b_grid), len(dev_queries))
    ranker = BM25Ranker(index)
    rows = []
    best_score = -1.0
    best_k1, best_b = 1.5, 0.75

    for k1 in k1_grid:
        for b in b_grid:
            ranker.set_params(k1, b)
            run = ranker.batch_search(dev_queries, k=k)
            per_query = ev.evaluate_run(run, dev_qrels)
            agg = ev.aggregate_metrics(per_query)
            score = agg.get("nDCG@10", 0.0)
            rows.append({"k1": k1, "b": b, "nDCG@10": round(score, 4)})
            if score > best_score:
                best_score = score
                best_k1, best_b = k1, b

    grid_df = pd.DataFrame(rows)
    logger.info("Best BM25 params: k1=%.2f, b=%.2f -> nDCG@10=%.4f", best_k1, best_b, best_score)
    return best_k1, best_b, grid_df


def plot_bm25_heatmap(grid_df: pd.DataFrame, out_path: Path, dpi: int = 300) -> None:
    """Save BM25 hyperparameter grid search heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = grid_df.pivot(index="k1", columns="b", values="nDCG@10")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("BM25 Grid Search: nDCG@10 on Dev Set", fontsize=13)
    ax.set_xlabel("b", fontsize=12)
    ax.set_ylabel("k1", fontsize=12)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("BM25 heatmap saved to %s", out_path)


# ===========================================================================
# Main experiment pipeline
# ===========================================================================

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    root = get_project_root()
    data_dir = root / cfg["dataset"]["data_dir"]
    results_dir = root / cfg["evaluation"]["results_dir"]
    report_dir = root / "report"
    ensure_dir(results_dir)
    ensure_dir(report_dir / "figures")
    ensure_dir(report_dir / "tables")

    k_eval = cfg["retrieval"]["default_k"]
    cutoffs = cfg["retrieval"]["eval_cutoffs"]

    # ---- 1. Load data --------------------------------------------------------
    logger.info("Loading processed data …")
    corpus_df, query_df, qrels_df = load_data(data_dir)
    qrels = build_qrels_dict(qrels_df)
    corpus_dict = {row["id"]: {"title": row.get("title", ""), "text": row["text"]}
                   for _, row in corpus_df.iterrows()}

    # ---- 2. Load index -------------------------------------------------------
    index_path = root / cfg["index"]["index_path"]
    index = load_index(index_path)

    # ---- 3. Split dev / test -------------------------------------------------
    dev_queries, test_queries = split_queries(
        query_df, qrels, cfg["dataset"]["dev_queries"], cfg["seed"]
    )
    dev_qrels = {qid: qrels[qid] for qid in dev_queries if qid in qrels}
    test_qrels = {qid: qrels[qid] for qid in test_queries if qid in qrels}

    # ---- 4. BM25 grid search -------------------------------------------------
    best_k1, best_b, grid_df = bm25_grid_search(
        index=index,
        dev_queries=dev_queries,
        dev_qrels=dev_qrels,
        k1_grid=cfg["bm25"]["k1_grid"],
        b_grid=cfg["bm25"]["b_grid"],
        k=10,
    )
    plot_bm25_heatmap(grid_df, results_dir / "bm25_heatmap.png")
    grid_df.to_csv(results_dir / "bm25_grid.csv", index=False)

    # ---- 5. Instantiate rankers ----------------------------------------------
    logger.info("Instantiating rankers …")
    tfidf_ranker = TFIDFRanker(
        index,
        tf_scheme=cfg["tfidf"]["tf_scheme"],
        normalize=cfg["tfidf"]["normalize"],
    )
    bm25_ranker = BM25Ranker(index, k1=best_k1, b=best_b)

    dense_cfg = cfg["dense"]
    dense_ranker = DenseRanker(
        model_name=dense_cfg["model_name"],
        embeddings_path=str(root / dense_cfg["embeddings_path"]),
        doc_ids_path=str(root / dense_cfg["doc_ids_path"]),
        batch_size=dense_cfg["batch_size"],
        normalize_embeddings=dense_cfg["normalize_embeddings"],
        use_faiss=dense_cfg["use_faiss"],
        max_seq_length=cfg["preprocessing"]["max_transformer_tokens"],
    )

    # Encode corpus (or load from cache)
    corpus_texts = [
        f"{row.get('title', '')} {row['text']}"
        for _, row in corpus_df.iterrows()
    ]
    corpus_doc_ids = corpus_df["id"].tolist()
    dense_ranker.encode_corpus(corpus_texts, corpus_doc_ids)

    rrf_cfg = cfg["rrf"]
    all_base_rankers = {"tfidf": tfidf_ranker, "bm25": bm25_ranker, "dense": dense_ranker}
    hybrid_ranker = HybridRRFRanker(
        rankers=all_base_rankers,
        k=rrf_cfg["k"],
        candidates_per_method=rrf_cfg["candidates_per_method"],
    )

    # ---- 6. Retrieve on test set ---------------------------------------------
    logger.info("Running retrieval on %d test queries (k=%d) …", len(test_queries), k_eval)
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    ranker_instances = {
        "TF-IDF": tfidf_ranker,
        "BM25": bm25_ranker,
        "Dense": dense_ranker,
        "Hybrid-RRF": hybrid_ranker,
    }

    for model_name, ranker in ranker_instances.items():
        logger.info("  Ranking with %s …", model_name)
        t0 = time.perf_counter()
        runs[model_name] = ranker.batch_search(test_queries, k=k_eval)
        logger.info("  %s done in %.1f s.", model_name, time.perf_counter() - t0)

    # ---- 7. Ablation study ---------------------------------------------------
    logger.info("Running RRF ablation study …")
    for combo in rrf_cfg["combinations"]:
        combo_name = "+".join(combo)
        if combo_name == "tfidf+bm25+dense":
            continue  # already have full hybrid
        ablation_ranker = HybridRRFRanker.from_subset(
            all_base_rankers, combo,
            k=rrf_cfg["k"],
            candidates_per_method=rrf_cfg["candidates_per_method"],
        )
        runs[f"RRF({combo_name})"] = ablation_ranker.batch_search(test_queries, k=k_eval)
        logger.info("  Ablation '%s' done.", combo_name)

    # ---- 8. Evaluate ---------------------------------------------------------
    logger.info("Evaluating all models …")
    per_query_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, run in runs.items():
        per_query = ev.evaluate_run(run, test_qrels, cutoffs=cutoffs)
        per_query_all[model_name] = per_query
        agg = ev.aggregate_metrics(per_query)
        model_metrics[model_name] = agg
        logger.info(
            "  %-25s  P@10=%.4f  nDCG@10=%.4f  MAP=%.4f",
            model_name, agg.get("P@10", 0), agg.get("nDCG@10", 0), agg.get("MAP", 0)
        )

    # ---- 9. Latency ----------------------------------------------------------
    logger.info("Measuring query latency …")
    latencies: Dict[str, float] = {}
    for model_name, ranker in ranker_instances.items():
        latencies[model_name] = ev.measure_latency(ranker, test_queries, k=10, n_trials=50)
        logger.info("  %s: %.1f ms/query", model_name, latencies[model_name])

    # Index sizes
    sparse_mb = (root / cfg["index"]["index_path"]).stat().st_size / 1024 / 1024
    emb_path = root / cfg["dense"]["embeddings_path"]
    dense_mb = emb_path.stat().st_size / 1024 / 1024 if emb_path.exists() else 0.0
    index_sizes = {
        "TF-IDF": sparse_mb,
        "BM25": sparse_mb,
        "Dense": dense_mb,
        "Hybrid-RRF": sparse_mb + dense_mb,
    }

    # ---- 10. Save tables -----------------------------------------------------
    df_results = ev.build_results_table(model_metrics, latencies)
    print("\n" + df_results.to_markdown())
    ev.save_results_table(df_results, results_dir)
    ev.save_latency_table(latencies, index_sizes, results_dir)

    # ---- 11. Significance test -----------------------------------------------
    best_single = max(
        ["TF-IDF", "BM25", "Dense"],
        key=lambda m: model_metrics.get(m, {}).get("nDCG@10", 0),
    )
    sig = ev.significance_test(
        per_query_all["Hybrid-RRF"],
        per_query_all[best_single],
        metric="nDCG@10",
        name_a="Hybrid-RRF",
        name_b=best_single,
    )
    ev.save_significance_report(sig, results_dir / "significance.txt")
    print(f"\nSignificance (Hybrid vs {best_single}): p={sig['p_value']:.4f}  "
          f"{'[SIGNIFICANT]' if sig['significant'] else '[not significant]'}")

    # ---- 12. Figures ---------------------------------------------------------
    logger.info("Generating figures …")
    primary_runs = {m: runs[m] for m in ["TF-IDF", "BM25", "Dense", "Hybrid-RRF"]}
    ev.plot_pr_curve(primary_runs, test_qrels, results_dir / "pr_curve.png")
    ev.plot_metrics_bar(df_results, results_dir / "metrics_bar.png")
    ev.plot_ndcg_histogram(
        {m: per_query_all[m] for m in ["TF-IDF", "BM25", "Dense", "Hybrid-RRF"]},
        results_dir / "ndcg_histogram.png",
    )
    ev.plot_bm25_vs_dense_winloss(
        per_query_all["BM25"], per_query_all["Dense"],
        results_dir / "bm25_vs_dense_winloss.png",
    )

    # Copy figures to report/figures/
    import shutil
    for png in results_dir.glob("*.png"):
        shutil.copy(png, report_dir / "figures" / png.name)

    # ---- 13. Error analysis --------------------------------------------------
    logger.info("Building error analysis …")
    runs_top3 = {m: runs[m] for m in ["TF-IDF", "BM25", "Dense", "Hybrid-RRF"]}
    error_md = ev.build_error_analysis(
        per_query_bm25=per_query_all["BM25"],
        per_query_dense=per_query_all["Dense"],
        per_query_hybrid=per_query_all["Hybrid-RRF"],
        queries=test_queries,
        corpus=corpus_dict,
        qrels=test_qrels,
        runs_top3=runs_top3,
    )
    with (results_dir / "error_analysis.md").open("w", encoding="utf-8") as fh:
        fh.write(error_md)
    logger.info("Error analysis saved.")

    # ---- 14. Report support --------------------------------------------------
    _save_report_support(cfg, model_metrics, latencies, index_sizes, sig, best_k1, best_b,
                         index, report_dir, df_results)

    logger.info("\nAll experiments complete. Results in: %s", results_dir)


def _save_report_support(
    cfg, model_metrics, latencies, index_sizes, sig, best_k1, best_b,
    index, report_dir, df_results
) -> None:
    """Write facts.md, outline.md, LaTeX tables, and references.bib."""
    # facts.md
    facts = [
        "# facts.md — Actual Measured Numbers for the Report\n",
        f"## Dataset",
        f"- Corpus size: see dataset_stats.json",
        f"- Seed: {cfg['seed']}",
        f"\n## Preprocessing",
        f"- Tokeniser: {cfg['preprocessing']['tokenizer']}",
        f"- Stemmer: {cfg['preprocessing']['stemmer']}",
        f"\n## Index",
        f"- Vocabulary size: {index.vocabulary_size():,}",
        f"- Total postings: {index.total_postings():,}",
        f"- Avg document length: {index.avg_dl:.1f} tokens",
        f"\n## BM25 Hyperparameters (tuned on dev set)",
        f"- Best k1 = {best_k1}",
        f"- Best b  = {best_b}",
        f"\n## Performance (test set)",
        df_results.to_markdown(),
        f"\n## Latencies (ms/query)",
    ] + [f"- {m}: {v:.1f} ms" for m, v in latencies.items()] + [
        f"\n## Statistical Significance",
        f"- Hybrid vs {sig['name_b']} on nDCG@10",
        f"- p-value = {sig['p_value']:.6f}",
        f"- {'SIGNIFICANT' if sig['significant'] else 'NOT SIGNIFICANT'} at α={sig['alpha']}",
    ]
    with (report_dir / "facts.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(facts))

    # LaTeX table
    latex_table = df_results.to_latex(float_format="%.4f", bold_rows=True)
    with (report_dir / "tables" / "performance.tex").open("w") as fh:
        fh.write(latex_table)
    with (report_dir / "tables" / "performance.md").open("w") as fh:
        fh.write(df_results.to_markdown())

    # outline.md
    _write_outline(report_dir)
    # references.bib
    _write_references(report_dir)

    logger.info("Report support files written to %s", report_dir)


def _write_outline(report_dir: Path) -> None:
    outline = """# Report Outline — Evidence Retrieval for Open-Domain QA on Natural Questions

## Abstract (≈150 words)
- One sentence: task and dataset (BEIR NQ subset).
- One sentence: four models compared.
- Key quantitative result (best nDCG@10, significance).
- One sentence: main takeaway.

## 1. Introduction
- Motivation: evidence retrieval as the backbone of open-domain QA.
- Research questions: how do sparse, dense, and hybrid paradigms compare?
- Contributions: reproducible benchmark on NQ subset; hand-built inverted index;
  head-to-head comparison; Streamlit demo.

## 2. Related Work
- Classic IR: TF-IDF, BM25 (Robertson & Zaragoza, 2009).
- Neural IR: dense retrieval (DPR, Reimers & Gurevych, 2019).
- Hybrid fusion: RRF (Cormack et al., 2009).
- BEIR benchmark (Thakur et al., 2021).
- Natural Questions (Kwiatkowski et al., 2019).

## 3. Dataset
- BEIR NQ description; full corpus size.
- Subsetting methodology (all relevant kept + random distractors).
- Dataset statistics table.
- Data card note: absolute scores not comparable to full corpus.

## 4. Methodology
### 4.1 Text Preprocessing
- Two pipelines: full (TF-IDF/BM25) vs light (Dense).
- Rationale for not stemming transformer inputs.
- Before/after example.

### 4.2 Inverted Index
- Data structures (PostingEntry, InvertedIndex).
- Build algorithm; title+text concatenation.
- Index statistics table.

### 4.3 TF-IDF
- Log-TF × log-IDF formula; cosine normalisation.

### 4.4 BM25
- Okapi BM25 formula (Robertson smoothed IDF).
- Hyperparameter tuning: grid search heatmap; chosen k1*, b*.

### 4.5 Sentence-Transformer Dense Retrieval
- Model: all-MiniLM-L6-v2; 384-dim; L2-normalised.
- FAISS IndexFlatIP; NumPy fallback.

### 4.6 Hybrid RRF
- RRF formula; k=60; candidates per method=100.
- Ablation combinations.

## 5. Experimental Setup
- Hardware (CPU model, RAM).
- All metrics defined (P@10, R@10/50/100, MAP, nDCG@10, MRR@10).
- Test set size; held-out dev set for BM25 tuning.

## 6. Results & Discussion
- Performance table (all 4 models + ablations).
- PR curve figure.
- Bar chart figure.
- Key finding: Dense > BM25 for semantic queries; BM25 strong for lexical.
- Hybrid consistently best on nDCG@10.
- Significance test result.
- Latency table (trade-off analysis).

## 7. Error Analysis
- 5 BM25-wins: exact term match, rare names.
- 5 Dense-wins: paraphrase, implicit topics.
- 5 Hybrid-fixes: complementary signals.

## 8. Limitations
- Subset inflates absolute scores vs full 2.68M corpus.
- Dense model not fine-tuned on NQ; using general-purpose MiniLM.
- No query expansion or re-ranking.

## 9. Conclusion
- Summary of findings.
- Best model recommendation.
- Future work: bi-encoder fine-tuning, cross-encoder re-ranking, SPLADE.

## References
(See references.bib)
"""
    with (report_dir / "outline.md").open("w", encoding="utf-8") as fh:
        fh.write(outline)


def _write_references(report_dir: Path) -> None:
    bib = """@article{robertson2009bm25,
  author    = {Stephen Robertson and Hugo Zaragoza},
  title     = {The Probabilistic Relevance Framework: {BM25} and Beyond},
  journal   = {Foundations and Trends in Information Retrieval},
  volume    = {3},
  number    = {4},
  pages     = {333--389},
  year      = {2009},
  doi       = {10.1561/1500000019}
}

@inproceedings{kwiatkowski2019nq,
  author    = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and
               Michael Collins and Ankur Parikh and Chris Alberti and
               Danielle Epstein and Illia Polosukhin and Matthew Kelcey and
               Jacob Devlin and Kenton Lee and Kristina N. Toutanova and
               Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit
               and Quoc Le and Slav Petrov},
  title     = {Natural Questions: A Benchmark for Question Answering Research},
  booktitle = {Transactions of the Association of Computational Linguistics},
  volume    = {7},
  pages     = {452--466},
  year      = {2019},
  doi       = {10.1162/tacl_a_00276}
}

@inproceedings{thakur2021beir,
  author    = {Nandan Thakur and Nils Reimers and Andreas R\\"{u}ckl\\'{e} and
               Abhishek Srivastava and Iryna Gurevych},
  title     = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of
               Information Retrieval Models},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems
               Datasets and Benchmarks Track},
  year      = {2021},
  url       = {https://openreview.net/forum?id=wCu6T5xFjeJ}
}

@inproceedings{reimers2019sbert,
  author    = {Nils Reimers and Iryna Gurevych},
  title     = {Sentence-{BERT}: Sentence Embeddings using {Siamese BERT}-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in
               Natural Language Processing},
  pages     = {3982--3992},
  year      = {2019},
  doi       = {10.18653/v1/D19-1410}
}

@inproceedings{cormack2009rrf,
  author    = {Gordon V. Cormack and Charles L. A. Clarke and Stefan Buettcher},
  title     = {Reciprocal Rank Fusion Outperforms {Condorcet} and Individual
               Rank Learning Methods},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on
               Research and Development in Information Retrieval},
  pages     = {758--759},
  year      = {2009},
  doi       = {10.1145/1571941.1572114}
}

@inproceedings{wang2020minilm,
  author    = {Wenhui Wang and Furu Wei and Li Dong and Hangbo Bao and
               Nan Yang and Ming Zhou},
  title     = {{MiniLM}: Deep Self-Attention Distillation for Task-Agnostic
               Compression of Pre-Trained Transformers},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {5776--5788},
  year      = {2020}
}
"""
    with (report_dir / "references.bib").open("w", encoding="utf-8") as fh:
        fh.write(bib)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NQ retrieval experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
