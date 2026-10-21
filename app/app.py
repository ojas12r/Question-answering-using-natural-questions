"""
app/app.py — Streamlit interactive demo for NQ Evidence Retrieval.

Tabs
----
1. Search  — free text or sample query, model selector, top-k slider,
             result cards with highlighting, gold-relevance badges.
2. Evaluation — performance table + PR curve.
3. RRF Explainer — rank breakdown table for the last Hybrid query.

Start with: streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---- path setup ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, get_logger
from index import load_index
from rankers.tfidf import TFIDFRanker
from rankers.bm25 import BM25Ranker
from rankers.dense import DenseRanker
from rankers.hybrid import HybridRRFRanker

logger = get_logger(__name__)

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="NQ Evidence Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.result-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #89b4fa; }

.rank-badge {
    background: #89b4fa;
    color: #1e1e2e;
    border-radius: 50%;
    width: 28px; height: 28px;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    margin-right: 8px;
}
.gold-badge {
    background: #f9e2af;
    color: #1e1e2e;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.score-badge {
    background: #313244;
    color: #cdd6f4;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
}
.title-text { font-weight: 600; font-size: 1rem; color: #cdd6f4; }
.passage-text { font-size: 0.88rem; color: #a6adc8; line-height: 1.6; }
mark { background: #f9e2af55; color: #f9e2af; padding: 1px 2px; border-radius: 3px; }

.metric-pill {
    display: inline-block;
    background: #313244;
    border-radius: 8px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.82rem;
    color: #cdd6f4;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Cached resource loading
# ============================================================================

@st.cache_resource(show_spinner="Loading indexes and models …")
def load_all(config_path: str = "config.yaml"):
    """Load config, index, and all rankers once at startup."""
    cfg = load_config(ROOT / config_path)
    data_dir = ROOT / cfg["dataset"]["data_dir"]

    # Corpus & qrels for gold-relevance badges
    corpus_df = pd.read_parquet(data_dir / "corpus.parquet")
    query_df  = pd.read_parquet(data_dir / "queries.parquet")
    qrels_df  = pd.read_parquet(data_dir / "qrels.parquet")

    corpus_dict: Dict[str, Dict] = {
        row["id"]: {"title": row.get("title", ""), "text": row["text"]}
        for _, row in corpus_df.iterrows()
    }
    qrels_dict: Dict[str, Dict[str, int]] = {}
    for row in qrels_df.itertuples():
        qrels_dict.setdefault(row.query_id, {})[row.corpus_id] = int(row.score)

    sample_queries = query_df.sample(
        n=min(cfg["app"]["sample_queries_count"], len(query_df)), random_state=42
    )[["id", "text"]].to_dict("records")

    # Index
    index = load_index(ROOT / cfg["index"]["index_path"])

    # Rankers
    tfidf_ranker = TFIDFRanker(index,
                               tf_scheme=cfg["tfidf"]["tf_scheme"],
                               normalize=cfg["tfidf"]["normalize"])
    bm25_ranker = BM25Ranker(index, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])

    dense_cfg = cfg["dense"]
    dense_ranker = DenseRanker(
        model_name=dense_cfg["model_name"],
        embeddings_path=str(ROOT / dense_cfg["embeddings_path"]),
        doc_ids_path=str(ROOT / dense_cfg["doc_ids_path"]),
        batch_size=dense_cfg["batch_size"],
        normalize_embeddings=dense_cfg["normalize_embeddings"],
        use_faiss=dense_cfg["use_faiss"],
        max_seq_length=cfg["preprocessing"]["max_transformer_tokens"],
    )
    dense_ranker.load()

    rrf_cfg = cfg["rrf"]
    hybrid_ranker = HybridRRFRanker(
        rankers={"tfidf": tfidf_ranker, "bm25": bm25_ranker, "dense": dense_ranker},
        k=rrf_cfg["k"],
        candidates_per_method=rrf_cfg["candidates_per_method"],
    )

    rankers = {
        "TF-IDF": tfidf_ranker,
        "BM25": bm25_ranker,
        "Dense": dense_ranker,
        "Hybrid RRF": hybrid_ranker,
    }

    return cfg, rankers, corpus_dict, qrels_dict, sample_queries


# ============================================================================
# Utilities
# ============================================================================

def highlight_query_terms(text: str, query: str) -> str:
    """Wrap query terms in <mark> tags for HTML display."""
    import re
    tokens = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
    for token in set(tokens):
        text = re.sub(
            rf"(?i)\b({re.escape(token)})\b",
            r"<mark>\1</mark>",
            text,
        )
    return text


def render_result_card(
    rank: int,
    doc_id: str,
    score: float,
    corpus_dict: Dict[str, Dict],
    query: str,
    is_gold: bool,
    max_chars: int = 400,
) -> None:
    """Render a single result as an HTML card."""
    doc = corpus_dict.get(doc_id, {"title": doc_id, "text": "(passage not found)"})
    title = doc.get("title", "") or doc_id
    text  = doc.get("text", "")[:max_chars] + ("…" if len(doc.get("text", "")) > max_chars else "")
    highlighted = highlight_query_terms(text, query)

    gold_html = '<span class="gold-badge">⭐ Gold Relevant</span>' if is_gold else ""
    html = f"""
    <div class="result-card">
      <div>
        <span class="rank-badge">{rank}</span>
        <span class="title-text">{title}</span>
        {gold_html}
        <span class="score-badge" style="float:right">{score:.4f}</span>
      </div>
      <div class="passage-text" style="margin-top:0.5rem">{highlighted}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def run_search(ranker_name: str, rankers, query: str, k: int):
    """Run search and return (results, elapsed_ms)."""
    ranker = rankers[ranker_name]
    t0 = time.perf_counter()
    if ranker_name == "Hybrid RRF":
        results, breakdown = ranker.search_with_breakdown(query, k=k)
        st.session_state["last_rrf_breakdown"] = breakdown
        st.session_state["last_rrf_query"] = query
    else:
        results = ranker.search(query, k=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar(cfg):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google-favicon-2015.png/64px-Google-favicon-2015.png", width=40)
        st.title("NQ Evidence Retrieval")
        st.caption("BEIR · TF-IDF · BM25 · Dense · Hybrid RRF")
        st.divider()
        st.markdown("**Model**")
        model_choice = st.radio(
            "Select model",
            ["TF-IDF", "BM25", "Dense", "Hybrid RRF", "Compare All"],
            label_visibility="collapsed",
        )
        st.divider()
        k = st.slider(
            "Top-K results",
            min_value=cfg["app"]["min_top_k"],
            max_value=cfg["app"]["max_top_k"],
            value=cfg["app"]["default_top_k"],
        )
        st.divider()
        st.markdown("**About**")
        st.markdown(
            "University IR project comparing four retrieval paradigms on "
            "the [Natural Questions](https://ai.google.com/research/NaturalQuestions) dataset."
        )
    return model_choice, k


# ============================================================================
# Tabs
# ============================================================================

def tab_search(cfg, rankers, corpus_dict, qrels_dict, sample_queries, model_choice, k):
    st.header("🔍 Search")

    # Query input
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g.  who sang the national anthem at super bowl 2016",
            key="search_query",
        )
    with col_sample:
        sample_labels = ["— sample queries —"] + [q["text"][:60] for q in sample_queries]
        chosen = st.selectbox("Sample queries", sample_labels, key="sample_select")
        if chosen != "— sample queries —":
            query = chosen

    if not query.strip():
        st.info("Enter a question above or pick a sample query to get started.")
        return

    # Determine which queries have gold qrels
    qid_lookup = {q["text"]: q["id"] for q in sample_queries}
    query_id = qid_lookup.get(query)
    gold_ids = set()
    if query_id and query_id in qrels_dict:
        gold_ids = {d for d, s in qrels_dict[query_id].items() if s > 0}

    # Run retrieval
    if model_choice == "Compare All":
        cols = st.columns(4)
        model_names = ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]
        for col, mname in zip(cols, model_names):
            with col:
                results, ms = run_search(mname, rankers, query, k)
                st.subheader(f"{mname}")
                st.caption(f"⏱ {ms:.0f} ms")
                for rank, (doc_id, score) in enumerate(results, 1):
                    render_result_card(rank, doc_id, score, corpus_dict, query,
                                       is_gold=(doc_id in gold_ids))
    else:
        results, ms = run_search(model_choice, rankers, query, k)
        st.caption(f"**{model_choice}** · ⏱ {ms:.0f} ms · {len(results)} results")
        for rank, (doc_id, score) in enumerate(results, 1):
            render_result_card(rank, doc_id, score, corpus_dict, query,
                               is_gold=(doc_id in gold_ids))


def tab_evaluation():
    st.header("📊 Evaluation Results")
    results_dir = ROOT / "results"

    table_path = results_dir / "performance_table.csv"
    if table_path.exists():
        df = pd.read_csv(table_path, index_col=0)
        st.subheader("Performance Table")
        st.dataframe(df.style.highlight_max(axis=0, color="#a6e3a1"), use_container_width=True)
    else:
        st.warning("Run `python src/run_experiments.py` to generate results.")

    pr_path = results_dir / "pr_curve.png"
    if pr_path.exists():
        st.subheader("Precision-Recall Curve")
        st.image(str(pr_path), use_container_width=True)

    bar_path = results_dir / "metrics_bar.png"
    if bar_path.exists():
        st.subheader("Metrics Bar Chart")
        st.image(str(bar_path), use_container_width=True)

    ndcg_path = results_dir / "ndcg_histogram.png"
    if ndcg_path.exists():
        st.subheader("nDCG@10 Distribution")
        st.image(str(ndcg_path), use_container_width=True)

    winloss_path = results_dir / "bm25_vs_dense_winloss.png"
    if winloss_path.exists():
        st.subheader("BM25 vs Dense Win/Loss")
        st.image(str(winloss_path), use_container_width=True)

    sig_path = results_dir / "significance.txt"
    if sig_path.exists():
        st.subheader("Statistical Significance")
        st.code(sig_path.read_text(), language="text")


def tab_rrf_explainer():
    st.header("🔀 RRF Rank Fusion Explainer")

    breakdown = st.session_state.get("last_rrf_breakdown")
    query = st.session_state.get("last_rrf_query", "")

    if not breakdown:
        st.info("Run a **Hybrid RRF** search in the Search tab first.")
        return

    st.caption(f"Query: *{query}*")
    rows = []
    for entry in breakdown:
        row = {"doc_id": entry["doc_id"][:20] + "…", "RRF Score": f"{entry['rrf_score']:.5f}"}
        for method, rank in entry["ranks"].items():
            row[f"Rank ({method})"] = rank if rank is not None else "—"
            row[f"1/(k+rank) ({method})"] = (
                f"{entry['contributions'][method]:.5f}"
                if entry["contributions"][method] > 0 else "0"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.markdown(
        "**Formula**: RRF_score(d) = Σ<sub>m</sub> 1/(60 + rank<sub>m</sub>(d))",
        unsafe_allow_html=True,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    try:
        cfg, rankers, corpus_dict, qrels_dict, sample_queries = load_all()
    except FileNotFoundError as e:
        st.error(
            f"**Setup incomplete.** {e}\n\n"
            "Please run the full pipeline first:\n```\npython run_all.py\n```"
        )
        return

    model_choice, k = render_sidebar(cfg)

    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Evaluation", "🔀 RRF Explainer"])

    with tab1:
        tab_search(cfg, rankers, corpus_dict, qrels_dict, sample_queries, model_choice, k)
    with tab2:
        tab_evaluation()
    with tab3:
        tab_rrf_explainer()


if __name__ == "__main__":
    main()
