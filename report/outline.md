# Report Outline — Evidence Retrieval for Open-Domain QA on Natural Questions

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
