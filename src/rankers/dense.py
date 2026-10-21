"""
src/rankers/dense.py — Sentence-Transformer dense retrieval ranker.

Model: sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim, CPU-friendly).
Retrieval: FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine).
Fallback: NumPy matmul if FAISS is not installed.

Corpus embeddings are encoded in batches, L2-normalised, and cached to .npy.
Query encoding happens at search time (< 5 ms per query on CPU).

Design note: clean_for_transformer() is used instead of the full preprocessing
pipeline. Stemming and stop-word removal hurt transformer quality because:
  1. Sub-word tokenisers (WordPiece/BPE) rely on word-form signals.
  2. Stop words carry syntactic context exploited by attention heads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import BaseRanker
from preprocess import clean_for_transformer

import logging
logger = logging.getLogger(__name__)


class DenseRanker(BaseRanker):
    """
    Dense retrieval using Sentence-Transformers + FAISS (or NumPy fallback).

    Parameters
    ----------
    model_name          : HuggingFace model identifier
    embeddings_path     : path to cached corpus embeddings (.npy)
    doc_ids_path        : path to cached doc-id ordering (.json)
    batch_size          : encoding batch size (default 64)
    normalize_embeddings: L2-normalise embeddings before indexing (default True)
    use_faiss           : try FAISS first; fall back to NumPy if unavailable
    max_seq_length      : max tokens for transformer input (default 256)
    """

    name = "Dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings_path: Optional[str] = None,
        doc_ids_path: Optional[str] = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        use_faiss: bool = True,
        max_seq_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.doc_ids_path = Path(doc_ids_path) if doc_ids_path else None
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_faiss = use_faiss
        self.max_seq_length = max_seq_length

        self._model = None         # loaded lazily
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self._doc_ids: Optional[List[str]] = None
        self._faiss_index = None   # FAISS index or None
        self._use_numpy_fallback = False

    # ------------------------------------------------------------------
    # Model / embedding loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading model: %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
            logger.info("Model loaded.")
        return self._model

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def encode_corpus(
        self,
        corpus_texts: List[str],
        doc_ids: List[str],
    ) -> None:
        """
        Encode and cache corpus embeddings.

        Parameters
        ----------
        corpus_texts : list of raw passage strings (title + text)
        doc_ids      : corresponding document ids (parallel list)
        """
        if self.embeddings_path and self.embeddings_path.exists() and \
                self.doc_ids_path and self.doc_ids_path.exists():
            logger.info("Cached embeddings found — loading from disk.")
            self._load_embeddings()
            return

        model = self._get_model()

        # Light cleaning (no stemming/stopwords)
        cleaned = [clean_for_transformer(t) for t in tqdm(corpus_texts, desc="Cleaning", ncols=80)]

        logger.info(
            "Encoding %d passages (batch_size=%d) — this may take 10–30 min on CPU …",
            len(cleaned),
            self.batch_size,
        )
        t0 = time.perf_counter()
        embeddings = model.encode(
            cleaned,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise below
        )
        elapsed = time.perf_counter() - t0
        logger.info("Encoding done in %.1f s.", elapsed)

        embeddings = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            embeddings = self._l2_normalize(embeddings)

        self._embeddings = embeddings
        self._doc_ids = list(doc_ids)

        # Cache to disk
        if self.embeddings_path:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self.embeddings_path), embeddings)
            logger.info("Embeddings saved: %s (%.1f MB)", self.embeddings_path,
                        self.embeddings_path.stat().st_size / 1024 / 1024)
        if self.doc_ids_path:
            self.doc_ids_path.parent.mkdir(parents=True, exist_ok=True)
            with self.doc_ids_path.open("w") as fh:
                json.dump(self._doc_ids, fh)
            logger.info("Doc IDs saved: %s", self.doc_ids_path)

        self._build_search_index()

    def _load_embeddings(self) -> None:
        """Load cached embeddings and doc ids from disk."""
        logger.info("Loading embeddings from %s …", self.embeddings_path)
        self._embeddings = np.load(str(self.embeddings_path)).astype(np.float32)
        with self.doc_ids_path.open("r") as fh:
            self._doc_ids = json.load(fh)
        logger.info(
            "Embeddings loaded: shape=%s, doc_ids=%d",
            self._embeddings.shape,
            len(self._doc_ids),
        )
        self._build_search_index()

    def load(self) -> None:
        """Public method to trigger loading of cached embeddings."""
        if self._embeddings is None:
            if (self.embeddings_path and self.embeddings_path.exists() and
                    self.doc_ids_path and self.doc_ids_path.exists()):
                self._load_embeddings()
            else:
                raise FileNotFoundError(
                    "Embeddings not found. Call encode_corpus() first."
                )

    def _build_search_index(self) -> None:
        """Build FAISS index (or set numpy fallback flag)."""
        if self._embeddings is None:
            return
        dim = self._embeddings.shape[1]

        if self.use_faiss:
            try:
                import faiss  # type: ignore
                index = faiss.IndexFlatIP(dim)
                index.add(self._embeddings)
                self._faiss_index = index
                logger.info("FAISS IndexFlatIP built with %d vectors (dim=%d).", index.ntotal, dim)
                return
            except ImportError:
                logger.warning("FAISS not installed — using NumPy fallback.")

        # NumPy fallback
        self._use_numpy_fallback = True
        logger.info("NumPy matmul fallback active (%d × %d).", *self._embeddings.shape)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to a normalised embedding vector."""
        model = self._get_model()
        cleaned = clean_for_transformer(query)
        vec = model.encode(
            [cleaned],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        if self.normalize_embeddings:
            vec = self._l2_normalize(vec)
        return vec[0]

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k passages for the query using dense similarity.

        Parameters
        ----------
        query : str  — raw query text
        k     : int  — number of results

        Returns
        -------
        List[Tuple[str, float]] — (doc_id, score), sorted descending.
        """
        if self._embeddings is None:
            raise RuntimeError("Call encode_corpus() or load() before searching.")

        q_vec = self._encode_query(query)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec[np.newaxis, :], k)
            scores = scores[0]
            indices = indices[0]
            results = [
                (self._doc_ids[idx], float(scores[i]))
                for i, idx in enumerate(indices)
                if idx >= 0
            ]
        else:
            # NumPy matmul fallback: O(N * D) — slower but always works
            scores = self._embeddings @ q_vec  # (N,)
            top_k_idx = np.argpartition(scores, -min(k, len(scores)))[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            results = [
                (self._doc_ids[idx], float(scores[idx]))
                for idx in top_k_idx
            ]

        return results
