#!/usr/bin/env python
"""Generate a Codabench-ready EB-NeRD submission.

Supports --method {bm25, semantic, hybrid}, same as MIND's generator.
Given your val-split findings (BM25 AUC~0.49 = near-random for EB-NeRD,
semantic AUC~0.54 = weak but real signal), pure semantic or a
semantic-favoring hybrid is likely your best bet here -- run tune_alpha.py
--dataset ebnerd first to confirm rather than guessing.

Usage:
    python generate_ebnerd_submission.py --method hybrid --alpha 0.2 \
        --articles data/raw/ebnerd/large/articles.parquet \
        --test-dir data/raw/ebnerd/testset
"""
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import compute_embeddings, save_embeddings, load_embeddings, embeddings_exist
from src.query_builder import build_query, build_user_embedding

N_RECENT = 5
BATCH_SIZE = 10_000


def build_articles(articles_path: Path) -> pd.DataFrame:
    articles = pd.read_parquet(articles_path)
    articles["text"] = (articles["title"].fillna("") + " " + articles["subtitle"].fillna("")).str.strip()
    print(f"  corpus: {len(articles)} articles")
    return articles


def build_bm25(articles: pd.DataFrame):
    index = BM25Index()
    index.fit(articles["article_id"].tolist(), articles["text"].tolist())
    return index


def build_semantic(articles: pd.DataFrame):
    if embeddings_exist("ebnerd_large"):
        article_ids, embeddings = load_embeddings("ebnerd_large")
    else:
        print("  computing embeddings for full large corpus (one-time, cached after)...")
        article_ids, embeddings = compute_embeddings(articles["article_id"].tolist(), articles["text"].tolist())
        save_embeddings("ebnerd_large", article_ids, embeddings)
    return ANNIndex().fit(article_ids, embeddings)


def build_user_history_lookup(history_path: Path) -> dict:
    print(f"  loading user history from {history_path}...")
    hist = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed"])
    lookup = dict(zip(hist["user_id"], hist["article_id_fixed"]))
    print(f"  {len(lookup):,} users with history")
    return lookup


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class UserQueryCache:
    def __init__(self, article_lookup: dict, id_to_embedding: dict, user_history: dict):
        self.article_lookup = article_lookup
        self.id_to_embedding = id_to_embedding
        self.user_history = user_history
        self._bm25_cache = {}
        self._semantic_cache = {}

    def get_bm25_query(self, user_id):
        if user_id not in self._bm25_cache:
            history = self.user_history.get(user_id, [])
            self._bm25_cache[user_id] = build_query(history, self.article_lookup, n_recent=N_RECENT)
        return self._bm25_cache[user_id]

    def get_semantic_vec(self, user_id):
        if user_id not in self._semantic_cache:
            history = self.user_history.get(user_id, [])
            self._semantic_cache[user_id] = build_user_embedding(history, self.id_to_embedding, n_recent=N_RECENT)
        return self._semantic_cache[user_id]


def process_batch(batch: pd.DataFrame, method: str, alpha: float,
                   bm25_index, semantic_index, query_cache: UserQueryCache, out_f):
    for row in batch.itertuples(index=False):
        candidates = list(row.article_ids_inview)

        if method in ("bm25", "hybrid"):
            query_text = query_cache.get_bm25_query(row.user_id)
            bm25_scores = (np.array(list(bm25_index.score_docs(candidates, query_text).values()))
                           if query_text else np.zeros(len(candidates)))

        if method in ("semantic", "hybrid"):
            user_vec = query_cache.get_semantic_vec(row.user_id)
            sem_scores = (semantic_index.score_candidates(candidates, user_vec)
                          if user_vec is not None else np.zeros(len(candidates)))

        if method == "bm25":
            final_scores = bm25_scores
        elif method == "semantic":
            final_scores = sem_scores
        else:
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
                     help="hybrid weight on BM25 (1-alpha goes to semantic); tune on val split's AUC first")
    ap.add_argument("--articles", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", default="predictions.txt")  # note: PLURAL per EB-NeRD's spec
    ap.add_argument("--zip", default="submission.zip")
    args = ap.parse_args()

    print("Building index over EB-NeRD article corpus...")
    articles = build_articles(Path(args.articles))
    article_lookup = dict(zip(articles["article_id"], articles["text"]))

    bm25_index = build_bm25(articles) if args.method in ("bm25", "hybrid") else None
    semantic_index = build_semantic(articles) if args.method in ("semantic", "hybrid") else None
    id_to_embedding = dict(zip(*load_embeddings("ebnerd_large"))) if semantic_index is not None else {}

    test_dir = Path(args.test_dir)
    user_history = build_user_history_lookup(test_dir / "history.parquet")
    query_cache = UserQueryCache(article_lookup, id_to_embedding, user_history)

    test_behaviors_path = test_dir / "behaviors.parquet"
    pf = pq.ParquetFile(test_behaviors_path)
    total_rows = pf.metadata.num_rows
    print(f"Method: {args.method}" + (f" (alpha={args.alpha})" if args.method == "hybrid" else ""))
    print(f"Streaming {test_behaviors_path} in batches of {BATCH_SIZE:,} ({total_rows:,} impressions)...")

    out_path = Path(args.out)
    n_written = 0
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for record_batch in pf.iter_batches(
                batch_size=BATCH_SIZE, columns=["impression_id", "user_id", "article_ids_inview"]):
            batch = record_batch.to_pandas()
            process_batch(batch, args.method, args.alpha, bm25_index, semantic_index, query_cache, out_f)
            n_written += len(batch)
            elapsed = time.time() - t_start
            rate = n_written / elapsed
            eta_min = (total_rows - n_written) / rate / 60 if rate > 0 else float("nan")
            print(f"  {n_written:,}/{total_rows:,} processed... ({rate:.0f}/sec, "
                  f"{elapsed/60:.1f} min elapsed, ~{eta_min:.1f} min remaining)")

    print(f"wrote {n_written:,} lines -> {out_path} in {(time.time()-t_start)/60:.1f} min")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)
    print(f"zipped -> {args.zip} (ready to upload)")


if __name__ == "__main__":
    main()