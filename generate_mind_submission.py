#!/usr/bin/env python
"""Generate a Codabench-ready MINDlarge submission.

Supports three scoring methods:
  --method bm25      lexical only (Q2's approach)
  --method semantic   embeddings only (Q3's approach)
  --method hybrid     alpha * bm25_norm + (1-alpha) * semantic_norm,
                       per-impression min-max normalized before blending
                       (BM25 and cosine similarity live on different scales,
                       so raw addition would be meaningless)

Vectorization notes (measured, not assumed -- see conversation for the
actual benchmarks that led here):
  - BM25 (sparse): batching ALL candidates across a whole chunk into ONE
    sparse operation IS faster (~3-4x) -- sparse rows are cheap to combine.
  - Semantic (dense): the same cross-impression batching trick actually
    made things SLOWER and memory-hungrier (dense vectors don't compress).
    The real fix there is simpler: one matrix-vector product PER IMPRESSION
    (ANNIndex.score_candidates) instead of per-candidate dot products in an
    inner loop -- still ~2.6x faster, with no extra memory risk.
  - Both methods cache each user's query (text or embedding) ONCE, reused
    across all their impressions in the test set.

Usage:
    python generate_mind_submission.py --method hybrid --alpha 0.5 \
        --train-dir data/raw/mind_large/train \
        --dev-dir data/raw/mind_large/dev \
        --test-dir data/raw/mind_large/test
"""
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import compute_embeddings, save_embeddings, load_embeddings, embeddings_exist
from src.query_builder import build_query, build_user_embedding

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
N_RECENT = 5
CHUNK_SIZE = 10_000  # kept conservative -- this is what fixed the OOM you hit earlier


def load_news(directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / "news.tsv", sep="\t", header=None, names=NEWS_COLS)


def build_corpus(train_dir: Path, dev_dir: Path, test_dir: Path) -> pd.DataFrame:
    print("  loading news.tsv from train/dev/test and deduplicating...")
    news = pd.concat([load_news(train_dir), load_news(dev_dir), load_news(test_dir)],
                      ignore_index=True).drop_duplicates("news_id")
    news["text"] = (news["title"].fillna("") + " " + news["abstract"].fillna("")).str.strip()
    print(f"  full corpus: {len(news)} unique articles")
    return news


def build_bm25(news: pd.DataFrame):
    index = BM25Index()
    index.fit(news["news_id"].tolist(), news["text"].tolist())
    return index


def build_semantic(news: pd.DataFrame):
    if embeddings_exist("mind_large"):
        article_ids, embeddings = load_embeddings("mind_large")
    else:
        print("  computing embeddings for full large corpus (one-time, cached after)...")
        article_ids, embeddings = compute_embeddings(news["news_id"].tolist(), news["text"].tolist())
        save_embeddings("mind_large", article_ids, embeddings)
    return ANNIndex().fit(article_ids, embeddings)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class UserQueryCache:
    """Caches BOTH the BM25 text query and the semantic mean-pooled
    embedding per user, built once, reused across all their impressions."""
    def __init__(self, article_lookup: dict, id_to_embedding: dict):
        self.article_lookup = article_lookup
        self.id_to_embedding = id_to_embedding
        self._bm25_cache = {}
        self._semantic_cache = {}

    def get_bm25_query(self, user_id, history):
        if user_id not in self._bm25_cache:
            self._bm25_cache[user_id] = build_query(history, self.article_lookup, n_recent=N_RECENT)
        return self._bm25_cache[user_id]

    def get_semantic_vec(self, user_id, history):
        if user_id not in self._semantic_cache:
            self._semantic_cache[user_id] = build_user_embedding(history, self.id_to_embedding, n_recent=N_RECENT)
        return self._semantic_cache[user_id]


def process_chunk(chunk: pd.DataFrame, method: str, alpha: float,
                   bm25_index, semantic_index, query_cache: UserQueryCache, out_f):
    for row in chunk.itertuples(index=False):
        candidates = str(row.impressions).split()
        history = str(row.history).split() if pd.notna(row.history) and row.history else []

        if method in ("bm25", "hybrid"):
            query_text = query_cache.get_bm25_query(row.user_id, history)
            if query_text:
                bm25_scores_dict = bm25_index.score_docs(candidates, query_text)
                bm25_scores = np.array([bm25_scores_dict[c] for c in candidates])
            else:
                bm25_scores = np.zeros(len(candidates))

        if method in ("semantic", "hybrid"):
            user_vec = query_cache.get_semantic_vec(row.user_id, history)
            sem_scores = semantic_index.score_candidates(candidates, user_vec) \
                if user_vec is not None else np.zeros(len(candidates))

        if method == "bm25":
            final_scores = bm25_scores
        elif method == "semantic":
            final_scores = sem_scores
        else:  # hybrid
            final_scores = alpha * minmax_normalize(bm25_scores) + (1 - alpha) * minmax_normalize(sem_scores)

        order = np.argsort(-final_scores, kind="stable")
        ranked_order = [candidates[i] for i in order]
        rank_of = {cid: i + 1 for i, cid in enumerate(ranked_order)}
        ranks_in_original_order = [rank_of[c] for c in candidates]
        rank_str = "[" + ",".join(str(r) for r in ranks_in_original_order) + "]"
        out_f.write(f"{row.impression_id} {rank_str}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bm25", "semantic", "hybrid"], default="hybrid")
    ap.add_argument("--alpha", type=float, default=0.5,
                     help="hybrid weight on BM25 (1-alpha goes to semantic); tune on your val split's AUC first")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", default="prediction.txt")
    ap.add_argument("--zip", default="submission.zip")
    args = ap.parse_args()

    train_dir, dev_dir, test_dir = Path(args.train_dir), Path(args.dev_dir), Path(args.test_dir)

    news = build_corpus(train_dir, dev_dir, test_dir)
    article_lookup = dict(zip(news["news_id"], news["text"]))

    bm25_index = build_bm25(news) if args.method in ("bm25", "hybrid") else None
    semantic_index = build_semantic(news) if args.method in ("semantic", "hybrid") else None
    id_to_embedding = dict(zip(*load_embeddings("mind_large"))) if semantic_index is not None else {}

    query_cache = UserQueryCache(article_lookup, id_to_embedding)

    test_behaviors_path = test_dir / "behaviors.tsv"
    print(f"Method: {args.method}" + (f" (alpha={args.alpha})" if args.method == "hybrid" else ""))
    print(f"Streaming {test_behaviors_path} in chunks of {CHUNK_SIZE:,}...")

    out_path = Path(args.out)
    n_written = 0
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for chunk in pd.read_csv(test_behaviors_path, sep="\t", header=None,
                                  names=BEH_COLS, chunksize=CHUNK_SIZE):
            process_chunk(chunk, args.method, args.alpha, bm25_index, semantic_index, query_cache, out_f)
            n_written += len(chunk)
            elapsed = time.time() - t_start
            rate = n_written / elapsed
            print(f"  {n_written:,} impressions processed... ({rate:.0f}/sec, {elapsed/60:.1f} min elapsed)")

    print(f"wrote {n_written:,} lines -> {out_path} in {(time.time()-t_start)/60:.1f} min")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)
    print(f"zipped -> {args.zip} (ready to upload)")


if __name__ == "__main__":
    main()