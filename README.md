# Question Answering using Natural Questions

> **IR Project** | Python 3.10+ | CPU-friendly | Fully reproducible

Retrieve supporting evidence passages from Wikipedia for real Google search queries
and compare four ranking paradigms:
**TF-IDF** · **BM25** · **Sentence-Transformer dense retrieval** · **Hybrid RRF**

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# 4. Run the full pipeline  (≈ 20–40 min on a laptop CPU, cached after first run)
python run_all.py

# 5. Launch the interactive demo
streamlit run app/app.py
```

### Running individual phases

```bash
python src/data.py          # Phase 1 – dataset
python src/index.py         # Phase 3 – inverted index
python src/run_experiments.py  # Phase 4+5 – rankers + evaluation
pytest tests/ -v            # unit tests
```

---

## Repository Structure

```
nq-retrieval/
├── config.yaml            ← all tuneable parameters
├── requirements.txt
├── Makefile               ← make all | make data | make test ...
├── run_all.py             ← Python equivalent of make all
├── README.md
├── data/
│   ├── raw/               ← downloaded BEIR NQ (git-ignored)
│   └── processed/         ← parquet + index + embeddings (git-ignored)
├── src/
│   ├── utils.py           ← shared config/logging helpers
│   ├── data.py            ← dataset download & subsetting
│   ├── preprocess.py      ← cleaning → tokenise → stopwords → stem
│   ├── index.py           ← hand-written inverted index
│   ├── rankers/
│   │   ├── base.py        ← abstract Ranker interface
│   │   ├── tfidf.py       ← TF-IDF (log tf × log idf, cosine)
│   │   ├── bm25.py        ← Okapi BM25 (k1=1.5, b=0.75)
│   │   ├── dense.py       ← Sentence-Transformer + FAISS
│   │   └── hybrid.py      ← Reciprocal Rank Fusion
│   ├── fusion.py          ← RRF core math
│   ├── evaluate.py        ← metrics + figures + significance
│   └── run_experiments.py ← orchestrator
├── app/app.py             ← Streamlit demo
├── tests/                 ← pytest suite
├── results/               ← CSVs, PNGs, error_analysis.md
└── report/                ← facts.md, outline.md, references.bib, tables/
```

---

## Data Card

### Dataset: BEIR Natural Questions (NQ) — Retrieval Subset

**Source**: [BEIR benchmark](https://github.com/beir-cellardog/beir), `beir/nq` split, 
originally derived from [Natural Questions](https://ai.google.com/research/NaturalQuestions)
(Kwiatkowski et al., 2019).

| Property | Value |
|----------|-------|
| Full corpus | ~2.68 M Wikipedia passages |
| Subset corpus | ~100 000 passages (configurable) |
| Test queries used | up to 1 000 (configurable) |
| Subset construction seed | 42 |

**Subsetting methodology**:
1. Identify every corpus passage that is marked relevant in the test qrels.
2. Fill the remaining budget with randomly sampled *distractor* passages (seed 42).
3. Save as Parquet.

**Why this is methodologically acceptable**: Every relevant passage is retained, so
Recall@K values are meaningful. Precision-based metrics (P@K, MAP, nDCG) may be
slightly inflated compared to the full corpus because the distractor pool is smaller,
making relevant passages proportionally easier to surface. All results are reported
on this fixed subset consistently, so relative comparisons between models are valid.
Absolute scores should **not** be compared directly to papers evaluated on the full
2.68 M corpus.

---

## Preprocessing Design Decisions

Two preprocessing modes are used deliberately:

| Mode | Used for | Steps |
|------|----------|-------|
| **Full pipeline** | TF-IDF, BM25 | Unicode normalise → strip HTML → remove URLs → lowercase → tokenise (NLTK) → remove stopwords → Snowball stem |
| **Light cleaning** | Sentence-Transformer | Unicode normalise → strip HTML → remove URLs → normalise whitespace |

**Rationale**: Stemming and stop-word removal benefit sparse models (reducing vocabulary
mismatch) but *hurt* transformer-based models. Transformers are pre-trained on natural
text with full morphology; their sub-word tokenisers rely on word-form signals, and stop
words carry syntactic context that attention heads use. Applying the full pipeline to
transformer input degrades retrieval quality.

---

## Configuring the Run

All parameters live in `config.yaml`:
- `dataset.target_corpus_size` — change subset size (default 100 000)
- `dataset.max_queries` — limit test queries (default 1 000)
- `bm25.k1`, `bm25.b` — BM25 hyperparameters (overwritten by tuning run)
- `dense.model_name` — swap the Sentence-Transformer model
- `rrf.k` — RRF fusion constant (default 60)

---

## Reproducing Results

```bash
# Full pipeline from scratch:
python run_all.py

# Skip already-built steps:
python run_all.py --skip-data --skip-index   # only re-run experiments
```

All intermediate artefacts (index, embeddings, run files) are cached to disk.
Re-running skips expensive steps automatically if output files exist.

---

## References

- Kwiatkowski et al. (2019). *Natural Questions: A Benchmark for QA Research.*
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation.*
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
- Cormack, Clarke & Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.*
