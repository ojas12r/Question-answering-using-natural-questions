# Presentation & Project Defense Guide: Evidence Retrieval for Open-Domain QA on Natural Questions

> **Project Title**: *Comparative Performance Evaluation of Sparse, Dense, and Hybrid (RRF) Retrieval for Evidence Retrieval on Natural Questions*  
> **Course / Context**: University Information Retrieval (IR) & Machine Learning Capstone  
> **Repository**: `nq-retrieval/`  
> **Live Interface**: Streamlit Demo (`http://localhost:8501`)  
> **Core Document**: Based on [`IR_Report_NQ_Project3.docx`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/IR_Report_NQ_Project3.docx) and the complete codebase.

---

## Table of Contents
1. [Executive Summary & Elevator Pitch](#1-executive-summary--elevator-pitch)
2. [Complete Slide-by-Slide Presentation Deck](#2-complete-slide-by-slide-presentation-deck)
   - [Slide 1: Title & Project Overview](#slide-1-title--project-overview)
   - [Slide 2: Motivation — The Retrieval Bottleneck in QA / RAG](#slide-2-motivation--the-retrieval-bottleneck-in-qa--rag)
   - [Slide 3: The Dataset — Natural Questions (BEIR Benchmark)](#slide-3-the-dataset--natural-questions-beir-benchmark)
   - [Slide 4: System Architecture & Dual Preprocessing Pipeline](#slide-4-system-architecture--dual-preprocessing-pipeline)
   - [Slide 5: Hand-Built Inverted Index & Sparse Rankers (TF-IDF & BM25)](#slide-5-hand-built-inverted-index--sparse-rankers-tf-idf--bm25)
   - [Slide 6: BM25 Hyperparameter Optimization Grid](#slide-6-bm25-hyperparameter-optimization-grid)
   - [Slide 7: Dense Semantic Retrieval (`all-MiniLM-L6-v2`)](#slide-7-dense-semantic-retrieval-all-minilm-l6-v2)
   - [Slide 8: Hybrid Reciprocal Rank Fusion (RRF) Math & Philosophy](#slide-8-hybrid-reciprocal-rank-fusion-rrf-math--philosophy)
   - [Slide 9: Benchmark Results & Comprehensive Performance Table](#slide-9-benchmark-results--comprehensive-performance-table)
   - [Slide 10: Critical Scientific Finding: Why 3-Way Fusion Underperformed](#slide-10-critical-scientific-finding-why-3-way-fusion-underperformed)
   - [Slide 11: Error Analysis — Qualitative Win/Loss Case Studies](#slide-11-error-analysis--qualitative-winloss-case-studies)
   - [Slide 12: Streamlit Interactive Interface & Live Demo](#slide-12-streamlit-interactive-interface--live-demo)
   - [Slide 13: Latency vs. Accuracy Trade-Offs & Conclusion](#slide-13-latency-vs-accuracy-trade-offs--conclusion)
3. [The Streamlit Interactive Interface Guide](#3-the-streamlit-interactive-interface-guide)
   - [Architecture & Performance Engineering](#architecture--performance-engineering)
   - [Tab 1: Search & Multi-Model Comparator](#tab-1-search--multi-model-comparator)
   - [Tab 2: Evaluation Analytics & Visualizations](#tab-2-evaluation-analytics--visualizations)
   - [Tab 3: RRF Rank Breakdown & Mathematical Explainer](#tab-3-rrf-rank-breakdown--mathematical-explainer)
   - [2-Minute Live Demonstration Script (Step-by-Step)](#2-minute-live-demonstration-script-step-by-step)
4. [In-Depth Report & Codebase Walkthrough](#4-in-depth-report--codebase-walkthrough)
   - [Structure of `IR_Report_NQ_Project3.docx`](#structure-of-ir_report_nq_project3docx)
   - [Mathematical Foundations of the 4 Ranking Paradigms](#mathematical-foundations-of-the-4-ranking-paradigms)
   - [Ablation Analysis: The "Weak Ranker Dilution" Effect](#ablation-analysis-the-weak-ranker-dilution-effect)
5. [Comprehensive Viva & Oral Defense Q&A Preparation](#5-comprehensive-viva--oral-defense-qa-preparation)
   - [Category A: Information Retrieval Theory & Formulations](#category-a-information-retrieval-theory--formulations)
   - [Category B: Dense Embeddings & Neural Retrieval](#category-b-dense-embeddings--neural-retrieval)
   - [Category C: Fusion Mechanics & Hybrid Search](#category-c-fusion-mechanics--hybrid-search)
   - [Category D: Evaluation Metrics & Statistical Rigor](#category-d-evaluation-metrics--statistical-rigor)
   - [Category E: Software Architecture & Reproducibility](#category-e-software-architecture--reproducibility)
   - [Category F: Critical Thinking, Flaws & Future Work](#category-f-critical-thinking-flaws--future-work)

---

## 1. Executive Summary & Elevator Pitch

### What is this project?
An end-to-end, reproducible Information Retrieval benchmark designed to solve the **evidence retrieval stage of Open-Domain Question Answering (ODQA)**. For real user queries issued to Google Search (from Google’s **Natural Questions** benchmark), the system retrieves the most relevant Wikipedia evidence passage among a 100,000-passage corpus.

### What are we comparing?
We evaluate four foundational information retrieval paradigms:
1. **TF-IDF Vector Space Model** (custom term frequency and cosine normalization built directly from our index).
2. **Okapi BM25** (probabilistic retrieval with tuned term saturation $k_1=0.9$ and document length normalization $b=0.4$).
3. **Sentence-Transformer Dense Retrieval** (`all-MiniLM-L6-v2`, 384-dimensional dense semantic vector space).
4. **Hybrid Reciprocal Rank Fusion (RRF)** (an unsupervised rank aggregation algorithm that merges rankings without score calibration).

### What is the core surprising takeaway?
Contrary to popular marketing claims that *"hybrid search is always better than single-model search"*, our empirical benchmark proves that **unweighted Reciprocal Rank Fusion (RRF) is susceptible to a "weak ranker dilution effect"**. 
* Dense retrieval is the single strongest paradigm ($\text{nDCG@10} = 0.8249$, $\text{MAP} = 0.7838$).
* When BM25 ($\text{nDCG@10} = 0.6806$) and TF-IDF ($\text{nDCG@10} = 0.4903$) are fused with Dense using equal rank weights, TF-IDF’s errors outvote Dense's correct top ranks, dragging hybrid performance down to $\text{nDCG@10} = 0.7488$ ($p < 0.0001$).
* However, two-way fusion of **BM25 + Dense** eliminates the dilution and achieves the benchmark's highest deep recall ($\text{Recall@100} = 99.29\%$), proving that fusion must only combine strong, mutually complementary rankers.

---

## 2. Complete Slide-by-Slide Presentation Deck

---

### Slide 1: Title & Project Overview
* **Visual**: Large minimalist dark theme title card with badges: `Python 3.12`, `Natural Questions Benchmark`, `100k Documents`, `Streamlit UI`.
* **Slide Content**:
  * **Title**: Evidence Retrieval for Open-Domain Question Answering on Natural Questions
  * **Subtitle**: Comparative Performance Evaluation of TF-IDF, Okapi BM25, Sentence-Transformers, and Reciprocal Rank Fusion
  * **Author**: Information Retrieval Capstone Project
* **Speaker Notes**:
  > *"Good morning. Today I am presenting our end-to-end Information Retrieval project. In modern question-answering systems and RAG pipelines, generation models are completely dependent on the retrieval phase. If the retriever fails to find the correct Wikipedia passage, the downstream reader will hallucinate. We built, benchmarked, and deployed a system from scratch comparing lexical, neural, and hybrid ranking paradigms across 800 test queries."*

---

### Slide 2: Motivation — The Retrieval Bottleneck in QA / RAG
* **Visual**: Diagram showing `User Question` $\to$ `[Retriever Engine]` $\to$ `Top-K Passages` $\to$ `[LLM / Reader]` $\to$ `Final Answer`. A red bottleneck marker highlights the Retriever.
* **Slide Content**:
  * **The Problem**: Open-Domain Question Answering (ODQA) requires searching millions of documents without knowing in advance which article contains the answer.
  * **The Retrieval Upper Bound**: A reader model can only extract answers from passages it is provided. Recall and precision at rank $K$ directly bound end-to-end QA accuracy.
  * **The Classical Conflict**:
    * *Lexical Retrieval (BM25)*: Excellent for exact numbers, acronyms, and rare entities; fails on synonyms and paraphrasing (*vocabulary mismatch problem*).
    * *Dense Retrieval (Transformers)*: Captures semantic intent and contextual meaning; susceptible to hallucinated semantic drift on exact numeric or named queries.
    * *The Hybrid Hypothesis*: Can Reciprocal Rank Fusion (RRF) combine their strengths without supervised training?
* **Speaker Notes**:
  > *"Why is evidence retrieval so critical? In open-domain QA, retrieval accuracy is an absolute ceiling on performance. Lexical search excels at exact strings, while dense embeddings excel at semantic matching. Conventional wisdom assumes that fusing both into a hybrid pipeline is guaranteed to win. In this project, we put that hypothesis to rigorous empirical test."*

---

### Slide 3: The Dataset — Natural Questions (BEIR Benchmark)
* **Visual**: Data card table and sample query breakdown.
* **Slide Content**:
  * **Origin**: Real, anonymized Google Search queries paired with human-annotated Wikipedia answer passages (Kwiatkowski et al., 2019; BEIR benchmark, Thakur et al., 2021).
  * **Reproducible Subset Design (Seed 42)**:
    * Full BEIR corpus has 2.68M passages—prohibitive for standard local execution.
    * We engineered a reproducible 100,000-passage benchmark: retaining **100% of all relevant passages (1,219 passages)** for 1,000 queries, filled with 98,781 randomly sampled distractors.
    * Split into 200 held-out development queries (for BM25 tuning) and 800 test queries.
  * **Key Challenge**: Average relevant passages per query is only **1.22** (sparse gold labels).

| Dataset Metric | Full BEIR NQ | Our Project Subset |
| :--- | :--- | :--- |
| Corpus Size | 2,681,468 passages | **100,000 passages** (Parquet: 34.3 MB) |
| Test Split | 3,452 queries | **800 test queries** |
| Dev Split | — | **200 tuning queries** |
| Mean Passage Length | ~76 words | **48.9 tokens** (post-processed) |

* **Speaker Notes**:
  > *"We used the BEIR formulation of Google's Natural Questions. To enable fully reproducible execution on laptop hardware, we created a 100,000-passage subset using random seed 42. Crucially, we guaranteed zero label leakage: every single gold answer passage was preserved. Because there are only ~1.2 relevant passages per query, Precision@10 naturally maxes out around 0.12, which makes nDCG@10, MAP, and MRR our primary evaluation metrics."*

---

### Slide 4: System Architecture & Dual Preprocessing Pipeline
* **Visual**: Architectural flowchart illustrating the split between the Sparse Pipeline and Dense Pipeline.
* **Slide Content**:
  * **Architectural Flowchart**:
```
                        [ Raw User Query ]
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
  [ Sparse Pipeline ]                              [ Dense Pipeline ]
  • NFKD Normalization                             • NFKD Normalization
  • Lowercasing & Punctuation Strip                • HTML / URL Strip
  • NLTK Word Tokenization                         • Preserves Casing
  • Stop-Word Removal (179 words)                  • Preserves Full Words & Syntax
  • Snowball Stemming (lossy)                      • Sub-word WordPiece/BPE
        │                                                │
        ▼                                                ▼
  [ Inverted Index ]                         [ Transformer Encoder ]
  (TF-IDF & Tuned BM25)                      (all-MiniLM-L6-v2, 384-d)
        │                                                │
        └───────────────────────┬────────────────────────┘
                                ▼
                 [ Reciprocal Rank Fusion (k=60) ]
                                │
                                ▼
                   [ Top-K Evidence Passages ]
```
  * **Core Engineering Rule**: Never stem transformer inputs! Stemming destroys sub-word tokenization morphology, and stop-word removal strips attention context.
* **Speaker Notes**:
  > *"Notice the deliberate divergence in preprocessing. For TF-IDF and BM25, vocabulary mismatch is reduced by aggressive tokenization, English stop-word filtering, and Snowball stemming. But for the Sentence-Transformer, stemming is catastrophic because pre-trained sub-word tokenizers rely on exact word forms and attention heads need syntax. We enforce two dedicated pipelines."*

---

### Slide 5: Hand-Built Inverted Index & Sparse Rankers (TF-IDF & BM25)
* **Visual**: Code snippets / structure showing `PostingEntry(doc_id, tf)` and posting lists.
* **Slide Content**:
  * **Zero External IR Black-Boxes**: The inverted index was built completely from scratch in Python (`src/index.py`).
  * **Internal Data Structures**:
    * `postings: dict[str, list[PostingEntry]]` — term to list of `(doc_id, tf)`
    * `df: dict[str, int]` — document frequency per term
    * `doc_lengths: dict[str, int]` — exact preprocessed token length per document
    * Statistics: **176,623 unique vocabulary terms** and **3,856,444 posting entries**, serialized to a 17.2 MB disk cache.
  * **TF-IDF Math**:
    $$\text{TF-Weight}(t, d) = 1 + \ln(\text{tf}(t, d)) \quad (\text{if } \text{tf} > 0)$$
    $$\text{IDF}(t) = \ln\left(\frac{N}{\text{df}(t)}\right)$$
    $$\text{Score}_{\text{TF-IDF}}(q, d) = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\|w_q\|_2 \cdot \|w_d\|_2}$$
* **Speaker Notes**:
  > *"Instead of importing Lucene or Anserini, we wrote the inverted index from scratch. Documents are indexed as concatenated title and passage text. Document frequency and lengths are tracked dynamically. The resulting index holds 3.85 million postings and unpickles in seconds."*

---

### Slide 6: BM25 Hyperparameter Optimization Grid
* **Visual**: Heatmap image (`results/bm25_heatmap.png`) showing $k_1$ on Y-axis vs $b$ on X-axis.
* **Slide Content**:
  * **Okapi BM25 Formulation**:
    $$\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}_{\text{Robertson}}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
    $$\text{IDF}_{\text{Robertson}}(t) = \ln\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$
  * **Grid Search on 200 Dev Queries**:
    * $k_1 \in \{0.9, 1.2, 1.5, 2.0\}$ (term saturation parameter)
    * $b \in \{0.4, 0.6, 0.75, 0.9\}$ (document length normalization penalty)
  * **Discovery**: Standard defaults ($k_1=1.5, b=0.75$) score $0.6244$. The optimal tuned setting is **$k_1 = 0.9, b = 0.4$**, reaching **$\text{nDCG@10} = 0.6629$** (+6.2% improvement).
  * **Physical Intuition**: Because Wikipedia evidence passages are short and homogeneous (mean 48.9 tokens), heavy length penalties ($b \ge 0.75$) unjustly penalize passages that contain more detailed answers.
* **Speaker Notes**:
  > *"We tuned BM25 on a 200-query dev set across 16 combinations. Standard textbook defaults for BM25 are k1=1.5 and b=0.75. However, our grid search proved that for short Wikipedia evidence passages, a lower b of 0.4 and lower k1 of 0.9 performs best. Why? When passages are relatively short and uniform, aggressive length normalization penalizes good answers."*

---

### Slide 7: Dense Semantic Retrieval (`all-MiniLM-L6-v2`)
* **Visual**: Diagram showing bi-encoder architecture and 384-dimensional inner product search.
* **Slide Content**:
  * **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, 6 layers, 384 hidden dimensions).
  * **Efficiency & Footprint**: Pre-trained via deep self-attention distillation from larger models; runs comfortably on CPU laptops in ~20 ms per query.
  * **Corpus Encoding**:
    * 100,000 passages encoded offline with batch size 64.
    * All vectors row-wise $L_2$-normalized: $\|\mathbf{v}\|_2 = 1.0$.
    * Serialized to disk as [`corpus_embeddings.npy`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/data/processed/corpus_embeddings.npy) ($100,000 \times 384 \times 4 \text{ bytes} \approx 146.5 \text{ MB}$).
  * **Online Scoring**: Inner product matrix multiplication ($\mathbf{E} \cdot \mathbf{q}^T$) exactly equals Cosine Similarity:
    $$\text{Score}_{\text{Dense}}(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d) = \mathbf{e}_q \cdot \mathbf{e}_d$$
  * **Sub-50ms CPU Querying**: Query vector encoding takes ~18 ms, vector dot product takes ~3 ms.
* **Speaker Notes**:
  > *"For dense retrieval, we chose all-MiniLM-L6-v2. It produces 384-dimensional embeddings. The 100k corpus was encoded once into a 146 MB NumPy array. Because all passage embeddings are pre-normalized to unit length, at runtime retrieval is simply an inner product matrix multiplication that runs in 21 milliseconds on a standard laptop CPU."*

---

### Slide 8: Hybrid Reciprocal Rank Fusion (RRF) Math & Philosophy
* **Visual**: Formula breakdown showing how rank 1 vs rank 50 contributes to score.
* **Slide Content**:
  * **Why RRF instead of Linear Score Combination?**
    * BM25 produces unbounded positive scores ($[0, 45+]$).
    * Dense retrieval produces cosine similarities ($[-1, 1]$).
    * Min-max normalization depends on query-specific outliers and requires hyperparameter calibration.
  * **The Cormack et al. (2009) Formulation**:
    $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    * $M$: The ensemble of ranking systems.
    * $r_m(d)$: The 1-based rank position of document $d$ in system $m$'s top 100.
    * $k$: Smoothing constant (standard $k = 60$).
  * **Mechanics**:
    * A document ranked #1 receives $\frac{1}{60 + 1} = 0.01639$.
    * A document ranked #100 receives $\frac{1}{60 + 100} = 0.00625$.
    * Documents missing from a model's top 100 receive zero.
* **Speaker Notes**:
  > *"Why Reciprocal Rank Fusion? Unlike linear score combination, RRF is completely parameter-free. You don't need to calibrate BM25 scores against cosine similarities. The smoothing constant k=60 prevents high ranks from dominating completely while ensuring that documents appearing across multiple systems rise to the top."*

---

### Slide 9: Benchmark Results & Comprehensive Performance Table
* **Visual**: Highlighted performance table with bar chart (`results/metrics_bar.png`) and PR curve (`results/pr_curve.png`).
* **Slide Content**:

| Ranking Model | P@10 | Recall@10 | Recall@50 | Recall@100 | MAP | nDCG@10 | MRR@10 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF** | 0.0816 | 0.6771 | 0.8712 | 0.9223 | 0.4352 | 0.4903 | 0.4416 | 24.8 ms |
| **BM25** (Tuned) | 0.0989 | 0.8244 | 0.9262 | 0.9490 | 0.6332 | 0.6806 | 0.6462 | 20.5 ms |
| **Dense** (MiniLM) | **0.1135** | **0.9412** | 0.9842 | 0.9921 | **0.7838** | **0.8249** | **0.7946** | **21.3 ms** |
| **Hybrid RRF** (3-Way) | 0.1075 | 0.8923 | 0.9725 | 0.9910 | 0.7000 | 0.7488 | 0.7141 | 69.6 ms |
| **RRF (BM25 + Dense)** | 0.1131 | 0.9404 | **0.9890** | **0.9929** | 0.7572 | 0.8044 | 0.7707 | — |
| **RRF (TF-IDF + BM25)** | 0.0958 | 0.7971 | 0.9206 | 0.9471 | 0.5839 | 0.6351 | 0.5933 | — |
| **RRF (TF-IDF + Dense)** | 0.1075 | 0.8948 | 0.9835 | 0.9917 | 0.6433 | 0.7051 | 0.6545 | — |

* **Speaker Notes**:
  > *"Here are the core numbers from our evaluation across 800 test queries. Notice the hierarchy: Dense is the strongest single model across every metric: nDCG@10 of 0.825 and MAP of 0.784. BM25 is a solid second with 0.681 nDCG@10. But notice the surprise: 3-way Hybrid RRF lands at 0.7488. It beat both lexical baselines, but it scored lower than Dense alone. Let's examine why."*

---

### Slide 10: Critical Scientific Finding: Why 3-Way Fusion Underperformed
* **Visual**: Scatter plot (`results/bm25_vs_dense_winloss.png`) and distribution histogram (`results/ndcg_histogram.png`).
* **Slide Content**:
  * **The Hypothesis**: Standard IR literature frequently assumes $M_{\text{Hybrid}} \ge \max(M_{\text{Dense}}, M_{\text{Sparse}})$.
  * **The Statistical Reality**:
    * Paired $t$-test: $t = -7.3584, p = 4.2 \times 10^{-13}$ (Dense significantly beats Hybrid-RRF).
  * **Why Did Fusion Hurt Top-10 Ranking? (The Weak Ranker Dilution Effect)**:
    1. **Asymmetric Strength**: Dense retrieval won on 323 queries (40.4%), BM25 won on 112 queries (14.0%), and they tied on 365 queries (45.6%). Dense was superior almost 3-to-1.
    2. **Equal Weighting Trap**: RRF gives equal voting power to all constituent lists. When TF-IDF ranks a non-relevant passage at #1 and Dense ranks the gold passage at #1, TF-IDF + BM25 can combine to outvote Dense's correct top choice.
    3. **Ablation Proof**:
       * Adding TF-IDF to BM25 **dropped** nDCG@10 from $0.6806 \to 0.6351$.
       * Adding TF-IDF to Dense **dropped** nDCG@10 from $0.8249 \to 0.7051$.
       * Only **BM25 + Dense** maintained competitive performance ($0.8044$) while maximizing deep recall ($\text{Recall@100} = 99.29\%$).
* **Speaker Notes**:
  > *"This is the most academically interesting finding of our project. Equal-weight RRF assumes all rankers have comparable baseline quality. But here, TF-IDF is substantially weaker. When TF-IDF and BM25 place an irrelevant document at rank 4, their combined reciprocal scores outvote Dense which had the gold passage at rank 1. Adding a weak model dilutes the ranking."*

---

### Slide 11: Error Analysis — Qualitative Win/Loss Case Studies
* **Visual**: Two contrasting query result cards showing gold passage placement.
* **Slide Content**:
  * **Case 1: BM25 Wins Over Dense (Exact Lexical Term Advantage)**
    * *Query*: `"indian cricketer who has the highest test batting average in cricket"`
    * *Gold Answer*: **Vinod Kambli** (played last test at 23, highest career average of 54).
    * **BM25 Rank**: **#1** ($\text{Score} = 34.11, \text{nDCG@10} = \mathbf{1.000}$) $\to$ Exact multi-word phrase overlap.
    * **Dense Rank**: **Not in top-10** ($\text{nDCG@10} = \mathbf{0.000}$) $\to$ Distracted by generic lists (*Kumar Sangakkara*, *List of Test cricket records*).
    * **Hybrid Rescue**: Ranked **#2** ($\text{nDCG@10} = \mathbf{0.631}$) $\to$ Hybrid successfully salvaged a complete Dense failure!
  * **Case 2: Dense Wins Over BM25 (Semantic Paraphrase Advantage)**
    * *Query*: `"what does a high white blood cell count mean"`
    * *Gold Answer*: **Leukocytosis** (medical term for elevated white blood cells).
    * **Dense Rank**: **#1** ($\text{nDCG@10} = \mathbf{1.000}$) $\to$ Encodes semantic synonymy between "high white blood cell count" and "leukocytosis".
    * **BM25 Rank**: **#7** ($\text{nDCG@10} = \mathbf{0.356}$) $\to$ Exact vocabulary mismatch on medical terminology.
* **Speaker Notes**:
  > *"These two cases demonstrate the distinct personalities of the models. For Vinod Kambli, Dense failed completely because it gravitated towards famous cricketers like Sangakkara. BM25 nailed it at Rank 1 due to exact token overlap. Hybrid RRF put it at Rank 2, preventing a total failure. Conversely, when users ask medical questions in layman terms, Dense identifies 'Leukocytosis' instantly while BM25 struggles with vocabulary mismatch."*

---

### Slide 12: Streamlit Interactive Interface & Live Demo
* **Visual**: Screenshot of the Streamlit application running on `http://localhost:8501`.
* **Slide Content**:
  * **Interactive Search Tab**:
    * Test with free text or 20 pre-loaded Natural Questions queries.
    * Model Selector: `TF-IDF`, `BM25`, `Dense`, `Hybrid RRF`, or `Compare All` (side-by-side 4-column layout).
    * Query term `<mark>` highlighting in yellow.
    * ⭐ **Gold Badges**: Instantly shows if a retrieved passage matches the ground-truth qrels.
  * **Evaluation Analytics Tab**:
    * Live interactive performance table and embedded 300 DPI Precision-Recall curves.
  * **RRF Explainer Tab**:
    * Full transparency table showing how rank positions from each model combine into the final score: $\sum \frac{1}{60 + r_m}$.
* **Speaker Notes**:
  > *"We packaged the entire project into a modern Streamlit web application. It features real-time search across all 100k passages, keyword highlighting, gold ground-truth indicators, a 4-model side-by-side comparison view, and a dedicated RRF Explainer tab that breaks down the math behind rank fusion."*

---

### Slide 13: Latency vs. Accuracy Trade-Offs & Conclusion
* **Visual**: Scatter chart of Latency (X-axis) vs. nDCG@10 (Y-axis).
* **Slide Content**:
  * **Efficiency Profile (Single CPU Laptop)**:
    * `TF-IDF`: 24.8 ms | 17.2 MB index
    * `BM25`: 20.5 ms | 17.2 MB index
    * `Dense`: 21.3 ms | 146.5 MB embeddings
    * `Hybrid RRF`: 69.6 ms (sum of all three pipelines + fusion overhead)
  * **Conclusions**:
    1. **Dense is king on semantic benchmarks**: Single-encoder MiniLM provides the best accuracy-latency trade-off (21 ms, 0.825 nDCG@10).
    2. **BM25 is the optimal sparse fallback**: Tuning $k_1$ and $b$ provides a +38.8% gain over TF-IDF at zero extra runtime cost.
    3. **Selective Hybridization**: Never fuse weak rankers. Fusing **BM25 + Dense** yields 99.29% Recall@100, ideal as a candidate generator for a downstream Cross-Encoder re-ranker.
* **Speaker Notes**:
  > *"To conclude: Dense retrieval is the clear winner for Natural Questions in both latency and accuracy. However, if your goal is maximal recall before a second-stage cross-encoder, a two-way fusion of BM25 and Dense achieves 99.3% Recall@100. Thank you, and I am now ready for your questions."*

---

## 3. The Streamlit Interactive Interface Guide

The web application is located in [`app/app.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/app/app.py) and runs locally via Streamlit.

### Architecture & Performance Engineering
* **Instant Startup via `@st.cache_resource`**: All heavy data structures—the 100,000-document inverted index, pre-computed dense embeddings ($146.5\text{ MB}$), corpus text mappings, and the Sentence-Transformer model—are cached in memory on initial load.
* **Asynchronous Warmup**: Subsequent user interactions execute in **under 25 milliseconds**.
* **Clean Dark-Mode Design**: Styled with custom CSS utilizing modern typography (Google Fonts *Inter*), high-contrast rank pills, and glassmorphic card borders.

```
┌────────────────────────────────────────────────────────────────────────┐
│  🔍 Natural Questions Evidence Retrieval Demo                          │
│  [ Search & Compare ]  [ Benchmark Evaluation ]  [ RRF Explainer ]     │
├────────────────────────────────────────────────────────────────────────┤
│  Query: [ who was the first president of the united states       ▼ ]  │
│  Model: ( ) TF-IDF  ( ) BM25  ( ) Dense  ( ) Hybrid  (●) Compare All   │
│  Top-K: [====|========= 10 ]                                           │
├───────────────┬───────────────┬────────────────┬───────────────────────┤
│    TF-IDF     │     BM25      │     Dense      │      Hybrid RRF       │
│  1. [Score]   │  1. [Score]   │  1. [Score] ⭐ │  1. [Score] ⭐        │
│  George Wash. │  George Wash. │  George Wash.  │  George Washington    │
│  "first pres" │  "first pres" │  "first pres"  │  "first pres"         │
└───────────────┴───────────────┴────────────────┴───────────────────────┘
```

---

### Tab 1: Search & Multi-Model Comparator
1. **Query Input Options**:
   * **Free-Text Search**: Users can type any open-domain question.
   * **Sample Query Dropdown**: 20 pre-loaded Natural Questions test queries with verified ground-truth answers in Wikipedia.
2. **Model Selection**:
   * Radios for `TF-IDF`, `BM25`, `Dense`, and `Hybrid RRF`.
   * **"Compare All" View**: Renders 4 parallel Streamlit columns (`st.columns(4)`), displaying the top-$K$ results of all four paradigms side-by-side for direct visual inspection.
3. **Card Annotations**:
   * **Rank Badge**: Visual circular badge (`1` through `K`).
   * **Score Badge**: Raw cosine similarity, BM25 score, or RRF aggregated score.
   * **Keyword Highlighting**: Exact query tokens in the passage are dynamically wrapped in `<mark>` tags.
   * **Gold Star Badge (⭐)**: Cross-references the document ID against `qrels.parquet`. If the passage contains the human-verified answer, a gold badge is rendered: `⭐ Gold Answer Passage`.

---

### Tab 2: Evaluation Analytics & Visualizations
* **Automated Table Rendering**: Automatically reads [`results/performance_table.md`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/results/performance_table.md) and displays it with formatted floating-point precision.
* **Visual Charts**:
  * Displays the high-resolution 11-point Precision-Recall Curve (`results/pr_curve.png`).
  * Displays the Win/Loss Scatter Plot (`results/bm25_vs_dense_winloss.png`).
  * Displays the BM25 Dev Grid Search Heatmap (`results/bm25_heatmap.png`).

---

### Tab 3: RRF Rank Breakdown & Mathematical Explainer
* Designed specifically for viva examinations and jury demonstrations.
* When a query is run with **Hybrid RRF**, this tab expands to reveal a mathematical breakdown table for the top 10 documents:
  
| Document ID | Title | TF-IDF Rank ($r_{\text{tfidf}}$) | BM25 Rank ($r_{\text{bm25}}$) | Dense Rank ($r_{\text{dense}}$) | RRF Score ($\sum \frac{1}{60 + r_m}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `doc1173136` | George Washington | 1 | 1 | 1 | $\frac{1}{61} + \frac{1}{61} + \frac{1}{61} = \mathbf{0.04918}$ |
| `doc589337` | President of the US | 4 | 2 | 12 | $\frac{1}{64} + \frac{1}{62} + \frac{1}{72} = \mathbf{0.04564}$ |

* Clearly proves how a document appearing in multiple lists accumulates reciprocal rank weight to rise to the top.

---

### 2-Minute Live Demonstration Script (Step-by-Step)

If an examiner asks you to demo your project live, follow this exact 120-second walkthrough:

1. **Step 1 (0:00 - 0:30) — Launch & Architecture**:
   * *Action*: Open terminal and run `streamlit run app/app.py`. Open browser at `http://localhost:8501`.
   * *Spoken*: *"Here is our interactive search interface running over the 100,000 Wikipedia passages. All models, inverted indexes, and 384-dimensional embeddings are cached in memory using `st.cache_resource`, giving us sub-25 millisecond query response times."*
2. **Step 2 (0:30 - 1:00) — Demonstrating Dense Semantic Superiority**:
   * *Action*: Select query `"what does a high white blood cell count mean"` and choose `Compare All`.
   * *Spoken*: *"Notice this query. The user asks in plain English about 'high white blood cell count'. Dense retrieval instantly ranks 'Leukocytosis' at Rank 1 with a gold star badge. BM25 ranks it down at Rank 7 because of vocabulary mismatch—the medical article doesn't repeat the word 'high' frequently enough."*
3. **Step 3 (1:00 - 1:30) — Demonstrating BM25 Exact Match Advantage**:
   * *Action*: Select query `"indian cricketer who has the highest test batting average in cricket"` and click `Compare All`.
   * *Spoken*: *"Now notice the opposite case. For exact named entities and statistics, BM25 scores the gold answer passage—Vinod Kambli—at Rank 1. Dense retrieval fails completely here, hallucinating generic lists and Kumar Sangakkara. But look at Column 4: Hybrid RRF rescues this query and ranks Vinod Kambli at Rank 2."*
4. **Step 4 (1:30 - 2:00) — Explaining the Fusion Math & Evaluation**:
   * *Action*: Click over to the **RRF Explainer** tab and show the table, then switch to the **Evaluation** tab.
   * *Spoken*: *"In the RRF Explainer tab, you can see how the reciprocal rank math $\frac{1}{60 + r}$ fuses the lists without needing score normalization. Finally, our Evaluation tab displays our full benchmark over 800 test queries, where Dense achieves an nDCG@10 of 0.825 and BM25 achieves 0.681."*

---

## 4. In-Depth Report & Codebase Walkthrough

### Structure of `IR_Report_NQ_Project3.docx`
The university technical report is structured into 5 standard academic sections:
1. **Section 1: Introduction**: Sets up Open-Domain QA, the retrieval ceiling hypothesis, vocabulary mismatch, and the contributions of our from-scratch implementation.
2. **Section 2: About Dataset**: Details the BEIR Natural Questions formulation, reproducible sampling with seed 42, passage length distributions, and explains why Precision@10 is naturally capped at ~0.12 due to single-answer annotations.
3. **Section 3: Methodology**:
   * Architectural pipeline (Figure 1 in report).
   * Dual preprocessing pipelines.
   * Hand-written Inverted Index data structures (`PostingEntry`, document lengths, document frequencies).
   * Exact scoring formulations for TF-IDF, Okapi BM25, and Sentence-Transformers.
   * Reciprocal Rank Fusion formulation with $k=60$.
4. **Section 4: Experimental Setup and Performance Analysis**:
   * Full metrics table across all 7 configurations.
   * 11-point Precision-Recall curves.
   * BM25 parameter grid surface heatmap ($k_1 \times b$).
   * Per-query bimodal distribution analysis (Figure 5).
   * Per-query BM25 vs Dense scatter win/loss breakdown (Figure 6).
   * Latency and index storage trade-offs.
   * Statistical significance testing with paired Student's t-test ($p < 0.001$).
5. **Section 5: Conclusion & Limitations**:
   * Summarizes the "weak ranker dilution" finding.
   * Documents limitations (subsampled corpus, CPU inference, unweighted fusion).
   * Outlines future directions (cross-encoder re-ranking, weighted fusion, learned sparse SPLADE).

---

### Mathematical Foundations of the 4 Ranking Paradigms

#### Paradigm 1: TF-IDF Vector Space Model
* Built directly from index postings in [`src/rankers/tfidf.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/src/rankers/tfidf.py).
* Sub-linear term frequency scaling:
  $$w_{t, d} = 1 + \ln(\text{tf}(t, d)) \quad \text{if } \text{tf}(t, d) > 0 \quad (\text{else } 0)$$
* Standard inverse document frequency:
  $$\text{idf}(t) = \ln\left(\frac{N}{\text{df}(t)}\right)$$
* Cosine normalization ensures long documents with repetitive terms do not dominate:
  $$\text{Score}_{\text{TF-IDF}}(q, d) = \frac{\sum_{t \in q \cap d} w_{t, q} \cdot w_{t, d}}{\sqrt{\sum_{t \in q} w_{t, q}^2} \cdot \sqrt{\sum_{t \in d} w_{t, d}^2}}$$

#### Paradigm 2: Okapi BM25
* Implemented in [`src/rankers/bm25.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/src/rankers/bm25.py).
* Solves TF-IDF's two main flaws:
  1. *Unbounded term frequency*: BM25 caps the gain from repeated occurrences via the term saturation hyperparameter $k_1$.
  2. *Excessive length penalization*: Document length is normalized relative to the average collection document length $\text{avgdl}$ scaled by parameter $b$.
* Formula:
  $$\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \ln\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

#### Paradigm 3: Dense Semantic Vector Search
* Implemented in [`src/rankers/dense.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/src/rankers/dense.py).
* Maps queries and passages to dense vectors: $\mathbf{e}_q = f_\theta(q), \mathbf{e}_d = f_\theta(d) \in \mathbb{R}^{384}$.
* With $L_2$-normalized vectors ($\|\mathbf{e}\|_2 = 1.0$), Cosine Similarity simplifies to an inner product:
  $$\text{Score}_{\text{Dense}}(q, d) = \mathbf{e}_q^T \mathbf{e}_d = \sum_{i=1}^{384} e_{q, i} \cdot e_{d, i}$$
* Eliminates the vocabulary mismatch problem by projecting synonyms into proximal geometric clusters.

#### Paradigm 4: Hybrid Reciprocal Rank Fusion (RRF)
* Implemented in [`src/fusion.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/src/fusion.py) and [`src/rankers/hybrid.py`](file:///c:/Users/ASUS%20PC/Downloads/ir_project/nq-retrieval/src/rankers/hybrid.py).
* Bypasses the need for score calibration:
  $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{TF-IDF}, \text{BM25}, \text{Dense}\}} \frac{1}{60 + r_m(d)}$$

---

### Ablation Analysis: The "Weak Ranker Dilution" Effect

The ablation experiments in Section 4 of the report provide crucial scientific insight:

1. **Full 3-Way Ensemble (`TF-IDF + BM25 + Dense`)**:
   * $\text{nDCG@10} = 0.7488$.
   * Better than TF-IDF ($0.4903$) and BM25 ($0.6806$), but **worse than Dense alone ($0.8249$)**.
2. **Ablation 1: `BM25 + Dense` (Dropping TF-IDF)**:
   * $\text{nDCG@10}$ jumps up to **$0.8044$** (+5.6 points over 3-way hybrid).
   * Maximizes deep recall: $\text{Recall@50} = \mathbf{98.90\%}$ and $\text{Recall@100} = \mathbf{99.29\%}$ (beating Dense alone).
3. **Ablation 2: `TF-IDF + BM25` (Lexical-Only Fusion)**:
   * $\text{nDCG@10}$ drops to **$0.6351$**, significantly lower than BM25 alone ($0.6806$).
4. **Ablation 3: `TF-IDF + Dense` (Weak Sparse + Dense)**:
   * $\text{nDCG@10} = 0.7051$ (a catastrophic 12-point drop from Dense alone).

**Conclusion**: In unweighted rank fusion, adding an inferior model is not harmless noise; it actively damages precision by artificially elevating lower-quality documents into the top-$K$.

---

## 5. Comprehensive Viva & Oral Defense Q&A Preparation

---

### Category A: Information Retrieval Theory & Formulations

#### Q1: What is the vocabulary mismatch problem, and how does it manifest in Natural Questions?
**Answer**:
The vocabulary mismatch problem occurs when a query and an answer-bearing document describe the exact same concept using different terminology. In Natural Questions, queries are natural human Google searches (e.g., *"who wrote the declaration of independence"*), while Wikipedia documents use formal encyclopedic language (e.g., *"The document was drafted principally by Thomas Jefferson"*). Lexical systems like BM25 fail to match *"wrote"* with *"drafted"* because their word stems differ (`write` vs `draft`). Dense retrievers solve this by projecting semantic equivalents into nearby points in vector space.

#### Q2: Explain the role of the parameters $k_1$ and $b$ in BM25. Why did our tuning choose $k_1=0.9$ and $b=0.4$ instead of the textbook defaults ($k_1=1.5, b=0.75$)?
**Answer**:
* $k_1$ controls **term frequency saturation**. As $\text{tf} \to \infty$, the term's contribution asymptotically approaches $(k_1 + 1)$. A smaller $k_1$ means the score saturates more rapidly, preventing passages that repeat a keyword many times from dominating.
* $b$ controls **document length penalization**. When $b=1$, the score is fully penalized in proportion to document length; when $b=0$, document length is completely ignored.
* Our grid search on 200 dev queries found $k_1=0.9, b=0.4$ optimal because our collection consists of short Wikipedia evidence passages (mean length 48.9 tokens) rather than full multi-page articles. Length variations are small, and a heavy length penalty ($b=0.75$) unfairly penalizes informative passages that are slightly longer simply because they contain the complete answer.

#### Q3: Why is sub-linear TF scaling ($1 + \ln(\text{tf})$) used in TF-IDF instead of raw term frequency?
**Answer**:
Raw term frequency implies that a document mentioning a query term 20 times is twenty times more relevant than a document mentioning it once. In reality, relevance follows the law of diminishing marginal utility. Sub-linear scaling ensures that the first occurrence carries the highest informational weight, while subsequent occurrences provide progressively smaller score increments.

---

### Category B: Dense Embeddings & Neural Retrieval

#### Q4: Why did we use `all-MiniLM-L6-v2` instead of a larger model like `all-mpnet-base-v2` or a modern 7B LLM embedding?
**Answer**:
1. **Computational Feasibility**: `all-MiniLM-L6-v2` has only 22.7M parameters and produces 384-dimensional embeddings. It encodes 100,000 passages in ~15 minutes on CPU and queries in ~18 ms, satisfying our strict project constraint to run efficiently on standard laptop hardware without a dedicated GPU.
2. **Memory Footprint**: 100,000 float32 vectors in 384 dimensions take only ~146.5 MB of RAM, allowing direct in-memory NumPy matrix operations without crashing or requiring distributed infrastructure.
3. **Strong Semantic Baseline**: Despite its small size, MiniLM was trained via knowledge distillation from large cross-encoders, achieving an exceptional $\text{nDCG@10} = 0.8249$ on our benchmark.

#### Q5: Why is stemming and stop-word removal forbidden before passing text into Sentence-Transformers?
**Answer**:
1. Pre-trained transformer models use sub-word tokenizers (such as WordPiece or Byte-Pair Encoding). They were trained on grammatically correct natural English text. Stemming (e.g., converting `"running"` to `"run"` or `"cities"` to `"citi"`) corrupts the morphological roots that sub-word tokenizers rely on.
2. Self-attention mechanisms in transformers utilize stop-words (such as prepositions and pronouns) to resolve syntactic dependencies, negation, and relationship directionality. Removing them destroys context.

#### Q6: Why did our dense retrieval use direct NumPy matrix multiplication instead of FAISS?
**Answer**:
FAISS is an optimized C++ library for approximate nearest neighbor (ANN) search. However, on Windows platforms, compiling and running `faiss-cpu` can encounter binary incompatibility issues. Because our benchmark subset is 100,000 vectors of dimension 384, an exact inner product matrix multiplication ($\mathbf{E} \cdot \mathbf{q}^T$) takes only **2 to 3 milliseconds in NumPy** via optimized BLAS/LAPACK routines. Exact search in NumPy eliminated external C++ dependencies while guaranteeing $100\%$ precision without approximate search recall loss.

---

### Category C: Fusion Mechanics & Hybrid Search

#### Q7: What is Reciprocal Rank Fusion (RRF), and what does the smoothing parameter $k=60$ do?
**Answer**:
RRF is an unsupervised rank aggregation method that combines multiple ranked lists using only document rank positions:
$$\text{Score}_{\text{RRF}}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
The parameter $k$ prevents the highest-ranked documents from dominating the score disproportionately. If $k=0$, rank 1 gets score $1.0$, while rank 2 gets $0.5$ (a massive 50% drop). With $k=60$, rank 1 receives $\frac{1}{61} \approx 0.01639$ and rank 2 receives $\frac{1}{62} \approx 0.01613$ (a gentle 1.6% drop). Cormack et al. (2009) determined empirically that $k=60$ provides the ideal balance across diverse retrieval systems.

#### Q8: Why did 3-way Hybrid RRF perform worse than Dense retrieval alone? Doesn't hybrid search always improve performance?
**Answer**:
Hybrid search does *not* always improve performance; this is a common misconception. Unweighted RRF grants equal voting power to all constituent rankers. In our benchmark:
* Dense retrieval had $\text{nDCG@10} = 0.8249$.
* BM25 had $\text{nDCG@10} = 0.6806$.
* TF-IDF had $\text{nDCG@10} = 0.4903$.
Because TF-IDF is substantially weaker, its rankings are noisy. When TF-IDF and BM25 both rank an irrelevant document at rank 4, their combined reciprocal score ($\frac{1}{64} + \frac{1}{64} = 0.0312$) can outscore a gold document that Dense placed at rank 1 ($\frac{1}{61} = 0.0164$). This is the **Weak Ranker Dilution Effect**. When we ran the ablation dropping TF-IDF (`BM25 + Dense`), nDCG jumped back up to $0.8044$ and achieved the highest top-100 recall ($99.29\%$).

#### Q9: When would you recommend using Hybrid BM25 + Dense over Dense alone in production?
**Answer**:
You should use Hybrid BM25 + Dense when building a **two-stage retrieval pipeline**:
1. **Stage 1 (Candidate Generation)**: Hybrid BM25 + Dense achieves $\text{Recall@100} = 99.29\%$ (higher than Dense alone at $99.21\%$). It guarantees that neither semantic paraphrases nor exact named entities (like *Vinod Kambli*) are missed.
2. **Stage 2 (Re-Ranking)**: A heavy Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`) scores all 100 candidates with full cross-attention. This captures the best of both worlds: maximum candidate recall from hybrid fusion, and top-1 precision from the re-ranker.

---

### Category D: Evaluation Metrics & Statistical Rigor

#### Q10: Why did we report nDCG@10, MAP, and MRR instead of just Precision@10?
**Answer**:
In Natural Questions, each query has an average of only **1.22 relevant passages** in the ground-truth qrels. Because Precision@10 is defined as $\frac{|\text{Relevant} \cap \text{Top-10}|}{10}$, even a perfect retriever that places the gold document at Rank 1 can achieve a maximum Precision@10 of only $\frac{1}{10} = 0.10$. Therefore, Precision@10 is mathematically bounded near 0.12. In contrast:
* **nDCG@10** normalizes against the Ideal DCG, properly rewarding systems that place the answer at Rank 1.
* **MRR@10** measures the reciprocal rank of the *first* correct answer, matching user satisfaction in question answering.
* **MAP** evaluates the entire precision-recall curve.

#### Q11: How was statistical significance tested, and what was the result?
**Answer**:
We performed a **paired, two-tailed Student's t-test** (`scipy.stats.ttest_rel`) comparing the per-query nDCG@10 scores of the best single model (Dense) against the Hybrid-RRF model across all 800 test queries. The test yielded $t = -7.3584$ and $p = 4.2 \times 10^{-13}$. Because $p \ll 0.05$, we rejected the null hypothesis and confirmed that the performance difference is statistically significant and not an artifact of random query sampling.

#### Q12: Why are the 11-point Precision-Recall curves in our results relatively flat?
**Answer**:
In typical ad-hoc web search (e.g., TREC ad-hoc), a query might have 50 or 100 relevant documents, causing precision to steadily decay as recall increases. In Natural Questions, most queries have exactly **one** relevant passage. Consequently, the query either achieves 100% recall at Rank 1 (giving high precision across all recall levels) or it fails completely. The flat curves reflect the binary nature of single-answer evidence retrieval.

---

### Category E: Software Architecture & Reproducibility

#### Q13: How did we ensure complete reproducibility across different machines?
**Answer**:
1. **Fixed Random Seed**: Seed `42` was hard-coded and passed into NumPy (`np.random.default_rng(42)`) and Pandas sampling routines for dataset subsetting, query splitting, and BM25 dev split selection.
2. **Zero Hard-Coded Paths**: Project paths are resolved dynamically relative to `Path(__file__).resolve().parent.parent`.
3. **Single Entry Point**: All data download, indexing, grid search, ranking, evaluation, and reporting steps can be run end-to-end via a single command: `python run_all.py` or `make all`.
4. **Automated Test Suite**: 61 unit tests (`pytest tests/`) test every formula (P@K, R@K, AP, nDCG, MRR), text preprocessing step, and inverted index integrity before experiments run.

#### Q14: How does the inverted index handle postings serialization without memory bloat?
**Answer**:
Instead of storing full document strings, the index stores a list of lightweight `PostingEntry(doc_id, tf)` dataclass instances. Vocabulary terms map directly to posting lists. The index is serialized with `joblib.dump(compress=3)`, compressing 3.85 million postings into a compact 17.2 MB file that loads into memory in under 45 seconds.

---

### Category F: Critical Thinking, Flaws & Future Work

#### Q15: What are the main limitations of our benchmark?
**Answer**:
1. **Subsampled Collection**: While 100,000 passages is a massive benchmark for local CPU evaluation, absolute performance numbers are higher than on the full 2.68M Wikipedia corpus because the distractor space is smaller.
2. **Single Dense Encoder**: We evaluated `all-MiniLM-L6-v2`. Larger bi-encoders fine-tuned specifically on MS-MARCO or NQ (such as `bge-base-en-v1.5` or `GritLM`) would yield even higher semantic scores.
3. **Unweighted Fusion**: We used standard unweighted Cormack RRF. Learning query-dependent fusion weights $w_m$ or down-weighting TF-IDF would mitigate the dilution effect.

#### Q16: If you were given a month and GPU budget to expand this project, what would you implement next?
**Answer**:
1. **Learned Sparse Retrieval (SPLADE)**: Instead of classic BM25, use SPLADE (Sparse Lexical and Expansion Model), which predicts term weights and expansions using BERT vocabulary heads while maintaining inverted index efficiency.
2. **Multi-Vector Dense Retrieval (ColBERT)**: Implement token-level late interaction via ColBERT, capturing fine-grained phrase alignments without full cross-encoder latency.
3. **Two-Stage Re-Ranking**: Deploy a Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the RRF candidates to resolve subtle factual distinctions.

---

## 6. Quick Reference: Reproduction & Commands

```bash
# 1. Run all unit tests (61 tests)
pytest tests/ -v

# 2. Re-run complete pipeline from scratch (Tests -> Data -> Index -> Experiments)
python run_all.py

# 3. Launch Streamlit interactive web demo
streamlit run app/app.py

# 4. View performance summary
cat results/performance_table.md
cat results/significance.txt
```
