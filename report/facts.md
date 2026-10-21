# facts.md — Actual Measured Numbers for the Report

## Dataset
- Corpus size: see dataset_stats.json
- Seed: 42

## Preprocessing
- Tokeniser: nltk
- Stemmer: snowball

## Index
- Vocabulary size: 176,623
- Total postings: 3,856,444
- Avg document length: 48.9 tokens

## BM25 Hyperparameters (tuned on dev set)
- Best k1 = 0.9
- Best b  = 0.4

## Performance (test set)
| Model            |   P@10 |   R@10 |   R@50 |   R@100 |    MAP |   nDCG@10 |   MRR@10 |   Latency(ms) |
|:-----------------|-------:|-------:|-------:|--------:|-------:|----------:|---------:|--------------:|
| TF-IDF           | 0.0816 | 0.6771 | 0.8712 |  0.9223 | 0.4352 |    0.4903 |   0.4416 |       24.8225 |
| BM25             | 0.0989 | 0.8244 | 0.9262 |  0.949  | 0.6332 |    0.6806 |   0.6462 |       20.4816 |
| Dense            | 0.1135 | 0.9412 | 0.9842 |  0.9921 | 0.7838 |    0.8249 |   0.7946 |       21.268  |
| Hybrid-RRF       | 0.1075 | 0.8923 | 0.9725 |  0.991  | 0.7    |    0.7488 |   0.7141 |       69.6447 |
| RRF(bm25+dense)  | 0.1131 | 0.9404 | 0.989  |  0.9929 | 0.7572 |    0.8044 |   0.7707 |        0      |
| RRF(tfidf+bm25)  | 0.0958 | 0.7971 | 0.9206 |  0.9471 | 0.5839 |    0.6351 |   0.5933 |        0      |
| RRF(tfidf+dense) | 0.1075 | 0.8948 | 0.9835 |  0.9917 | 0.6433 |    0.7051 |   0.6545 |        0      |

## Latencies (ms/query)
- TF-IDF: 24.8 ms
- BM25: 20.5 ms
- Dense: 21.3 ms
- Hybrid-RRF: 69.6 ms

## Statistical Significance
- Hybrid vs Dense on nDCG@10
- p-value = 0.000000
- SIGNIFICANT at α=0.05