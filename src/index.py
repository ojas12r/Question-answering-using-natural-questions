"""
src/index.py — Hand-written Inverted Index for NQ Evidence Retrieval.

This module implements an inverted index from scratch (no external IR library).
The index stores:
  - postings:    term → list of PostingEntry(doc_id, tf)
  - doc_lengths: doc_id → number of (preprocessed) tokens in the document
  - df:          term → document frequency (number of docs containing term)
  - N:           total number of indexed documents
  - avg_dl:      average document length (tokens)

Design choices
--------------
- Documents are indexed as "title + ' ' + text" concatenation to give the
  title extra weight without a separate field weighting scheme.
- Preprocessing (stem + stopwords) is applied via src/preprocess.py.
- Serialised with joblib for speed and disk efficiency.
- Building is iterative (one doc at a time) to avoid large memory spikes.

Usage
-----
    python src/index.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, set_seed, ensure_dir, get_project_root
from preprocess import preprocess

logger = get_logger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class PostingEntry:
    """A single entry in a posting list."""
    doc_id: str
    tf: int  # raw term frequency within the document


@dataclass
class InvertedIndex:
    """
    In-memory inverted index.

    Attributes
    ----------
    postings   : dict mapping term (str) → list of PostingEntry
    doc_lengths: dict mapping doc_id (str) → token count (int)
    df         : dict mapping term (str) → document frequency (int)
    N          : total number of indexed documents
    avg_dl     : average document length in tokens
    doc_ids    : ordered list of all doc_ids (for dense retrieval alignment)
    """
    postings: Dict[str, List[PostingEntry]] = field(default_factory=dict)
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    df: Dict[str, int] = field(default_factory=dict)
    N: int = 0
    avg_dl: float = 0.0
    doc_ids: List[str] = field(default_factory=list)

    def get_postings(self, term: str) -> List[PostingEntry]:
        """Return posting list for a term, or empty list if not found."""
        return self.postings.get(term, [])

    def get_doc_length(self, doc_id: str) -> int:
        """Return the token count for a document."""
        return self.doc_lengths.get(doc_id, 0)

    def vocabulary_size(self) -> int:
        """Return the number of distinct terms in the index."""
        return len(self.postings)

    def total_postings(self) -> int:
        """Return the total number of (term, doc) postings entries."""
        return sum(len(pl) for pl in self.postings.values())


# ===========================================================================
# Index builder
# ===========================================================================

def build_index(
    corpus_df: pd.DataFrame,
    stem: bool = True,
    remove_stopwords: bool = True,
    field: str = "title_text",
    show_progress: bool = True,
) -> InvertedIndex:
    """
    Build an inverted index from a corpus DataFrame.

    Parameters
    ----------
    corpus_df        : DataFrame with columns [id, title, text]
    stem             : apply Snowball stemming (default True)
    remove_stopwords : apply stop-word removal (default True)
    field            : "title_text" = title + " " + text (recommended)
                       "text"       = body only
    show_progress    : display tqdm progress bar

    Returns
    -------
    InvertedIndex
        Fully populated inverted index.
    """
    index = InvertedIndex()

    # temp structure: term → {doc_id: tf}
    tf_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    logger.info(
        "Building inverted index over %d documents (stem=%s, stopwords=%s) …",
        len(corpus_df),
        stem,
        remove_stopwords,
    )

    iterator = tqdm(corpus_df.itertuples(), total=len(corpus_df), disable=not show_progress,
                    desc="Indexing", ncols=80)

    for row in iterator:
        doc_id: str = str(row.id)

        # Build the text to index
        if field == "title_text":
            raw = f"{getattr(row, 'title', '')} {getattr(row, 'text', '')}".strip()
        else:
            raw = str(getattr(row, "text", ""))

        tokens = preprocess(raw, stem=stem, remove_stopwords=remove_stopwords)
        doc_len = len(tokens)

        # Count term frequencies for this document
        local_tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            local_tf[token] += 1

        # Update global structures
        for term, tf in local_tf.items():
            tf_map[term][doc_id] = tf

        index.doc_lengths[doc_id] = doc_len
        index.doc_ids.append(doc_id)
        index.N += 1

    # ---- Build postings and df from tf_map ----------------------------------
    logger.info("Compressing postings lists …")
    for term, doc_tf_dict in tf_map.items():
        index.postings[term] = [
            PostingEntry(doc_id=did, tf=tf) for did, tf in doc_tf_dict.items()
        ]
        index.df[term] = len(doc_tf_dict)

    # ---- Compute avg_dl -----------------------------------------------------
    total_tokens = sum(index.doc_lengths.values())
    index.avg_dl = total_tokens / index.N if index.N > 0 else 0.0

    logger.info(
        "Index built: %d terms, %d postings, avg_dl=%.1f",
        index.vocabulary_size(),
        index.total_postings(),
        index.avg_dl,
    )
    return index


# ===========================================================================
# Persistence
# ===========================================================================

def save_index(index: InvertedIndex, path: Path) -> None:
    """Serialize the index to disk using joblib."""
    ensure_dir(path.parent)
    joblib.dump(index, path, compress=3)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info("Index saved to %s (%.1f MB)", path, size_mb)


def load_index(path: Path) -> InvertedIndex:
    """Deserialize the index from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    logger.info("Loading index from %s …", path)
    t0 = time.perf_counter()

    # Ensure __main__ has PostingEntry and InvertedIndex for unpickler compatibility
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        if not hasattr(main_mod, "PostingEntry"):
            setattr(main_mod, "PostingEntry", PostingEntry)
        if not hasattr(main_mod, "InvertedIndex"):
            setattr(main_mod, "InvertedIndex", InvertedIndex)

    index = joblib.load(path)
    logger.info("Index loaded in %.1f s.", time.perf_counter() - t0)
    return index


def index_exists(path: Path) -> bool:
    """Return True if the serialized index file exists."""
    return path.exists()


# ===========================================================================
# Statistics printer
# ===========================================================================

def print_index_stats(index: InvertedIndex, build_time: float, index_path: Path) -> dict:
    """Print and return a dict of index statistics."""
    size_mb = index_path.stat().st_size / 1024 / 1024 if index_path.exists() else 0.0

    stats = {
        "vocabulary_size": index.vocabulary_size(),
        "total_postings": index.total_postings(),
        "num_documents": index.N,
        "avg_document_length": round(index.avg_dl, 2),
        "build_time_s": round(build_time, 2),
        "disk_size_mb": round(size_mb, 2),
    }

    print("\n" + "=" * 55)
    print("  INVERTED INDEX STATISTICS")
    print("=" * 55)
    print(f"  {'Vocabulary size (unique terms)':<35} {stats['vocabulary_size']:>10,}")
    print(f"  {'Total postings (term-doc pairs)':<35} {stats['total_postings']:>10,}")
    print(f"  {'Documents indexed':<35} {stats['num_documents']:>10,}")
    print(f"  {'Avg document length (tokens)':<35} {stats['avg_document_length']:>10.1f}")
    print(f"  {'Build time (s)':<35} {stats['build_time_s']:>10.1f}")
    print(f"  {'Index disk size (MB)':<35} {stats['disk_size_mb']:>10.1f}")
    print("=" * 55 + "\n")

    # Spot-check
    for spot_term in ["president", "university", "war", "science"]:
        postings = index.get_postings(spot_term)
        if postings:
            logger.info(
                "Spot-check '%s': df=%d, first posting=(doc=%s, tf=%d)",
                spot_term, len(postings), postings[0].doc_id, postings[0].tf
            )

    return stats


# ===========================================================================
# Main entry point
# ===========================================================================

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    root = get_project_root()
    data_dir = root / cfg["dataset"]["data_dir"]
    index_path = root / cfg["index"]["index_path"]

    # ---- Skip if already built -----------------------------------------------
    if index_exists(index_path):
        logger.info("Index already exists at %s — loading and printing stats.", index_path)
        index = load_index(index_path)
        print_index_stats(index, 0.0, index_path)
        return

    # ---- Load processed corpus -----------------------------------------------
    corpus_parquet = data_dir / "corpus.parquet"
    if not corpus_parquet.exists():
        raise FileNotFoundError(
            f"Processed corpus not found at {corpus_parquet}. Run `python src/data.py` first."
        )
    corpus_df = pd.read_parquet(corpus_parquet)
    logger.info("Loaded corpus: %d passages.", len(corpus_df))

    # ---- Build index ---------------------------------------------------------
    t0 = time.perf_counter()
    index = build_index(
        corpus_df=corpus_df,
        stem=True,
        remove_stopwords=True,
        field=cfg["index"]["field"],
        show_progress=True,
    )
    build_time = time.perf_counter() - t0

    # ---- Save ----------------------------------------------------------------
    save_index(index, index_path)

    # ---- Print stats ---------------------------------------------------------
    print_index_stats(index, build_time, index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the inverted index.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
