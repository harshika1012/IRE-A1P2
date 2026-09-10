#!/usr/bin/env python
"""Sweep hybrid alpha (bm25 weight) on the val split to find the best blend,
using AUC as the primary selection metric (matches what the leaderboard
likely weighs most). Builds both indices ONCE, scores every impression with
BOTH methods ONCE, then tries many alpha values cheaply (just re-blending
already-computed scores, not re-running retrieval).

Usage:
    python tune_alpha.py --dataset mind --split val --max-impressions -1
"""
import argparse
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR
from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import load_embeddings, embeddings_exist, compute_embeddings, save_embeddings
from src.query_builder import build_query, build_user_embedding
from src.ranking_metrics import auc, mrr, ndcg_at_k

ALPHA_GRID = np.arange(0.0, 1.01, 0.1)  # 0.0 = pure semantic, 1.0 = pure bm25


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def build_bm25_index(articles, dataset):
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    index = BM25Index()
    index.fit(sub["article_id"].tolist(), sub["text"].tolist())
    return index, dict(zip(sub["article_id"], sub["text"]))


def build_semantic_index(articles, dataset):
    if embeddings_exist(dataset):
        article_ids, embeddings = load_embeddings(dataset)
    else:
        sub = articles[articles["dataset"] == dataset].copy()
        sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
        article_ids, embeddings = compute_embeddings(sub["article_id"].tolist(), sub["text"].tolist())
        save_embeddings(dataset, article_ids, embeddings)
    index = ANNIndex().fit(article_ids, embeddings)
    return index, dict(zip(article_ids, embeddings))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-impressions", type=int, default=5000)
    args = ap.parse_args()
    max_imp = None if args.max_impressions == -1 else args.max_impressions

    print(f"=== tuning alpha for {args.dataset} / {args.split} ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet")
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet")

    bm25_index, bm25_lookup = build_bm25_index(articles, args.dataset)
    sem_index, sem_lookup = build_semantic_index(articles, args.dataset)

    uf = user_features[user_features["dataset"] == args.dataset]
    click_hist = uf.set_index("user_id")["click_history"].to_dict()

    imp = impressions[(impressions["dataset"] == args.dataset) & (impressions["split"] == args.split)]
    grouped = imp.groupby("impression_id")
    impression_ids = list(grouped.groups.keys())
    if max_imp and len(impression_ids) > max_imp:
        rng = np.random.default_rng(42)
        impression_ids = list(rng.choice(impression_ids, size=max_imp, replace=False))
    print(f"scoring {len(impression_ids)} impressions with BOTH methods (once each)...")

    # score every impression ONCE per method, store normalized arrays + labels
    records = []
    n_skipped = 0
    for iid in impression_ids:
        grp = grouped.get_group(iid)
        uid = grp["user_id"].iloc[0]
        candidates = grp["article_id"].tolist()
        labels = np.array(grp["clicked"].tolist())
        if len(candidates) < 2 or labels.sum() == 0:
            n_skipped += 1
            continue

        history = click_hist.get(uid, [])
        bm25_query = build_query(history, bm25_lookup)
        sem_vec = build_user_embedding(history, sem_lookup)
        if not bm25_query and sem_vec is None:
            n_skipped += 1
            continue

        bm25_scores = (np.array(list(bm25_index.score_docs(candidates, bm25_query).values()))
                       if bm25_query else np.zeros(len(candidates)))
        sem_scores = (sem_index.score_candidates(candidates, sem_vec)
                      if sem_vec is not None else np.zeros(len(candidates)))

        records.append((minmax_normalize(bm25_scores), minmax_normalize(sem_scores), labels))

    print(f"skipped {n_skipped}, usable: {len(records)}\n")

    print(f"{'alpha':>6} | {'AUC':>8} | {'MRR':>8} | {'nDCG@5':>8} | {'nDCG@10':>8}")
    best_alpha, best_auc = None, -1
    for alpha in ALPHA_GRID:
        aucs, mrrs, n5s, n10s = [], [], [], []
        for bm25_norm, sem_norm, labels in records:
            scores = alpha * bm25_norm + (1 - alpha) * sem_norm
            a, m, n5, n10 = auc(scores, labels), mrr(scores, labels), ndcg_at_k(scores, labels, 5), ndcg_at_k(scores, labels, 10)
            if a is not None: aucs.append(a)
            if m is not None: mrrs.append(m)
            if n5 is not None: n5s.append(n5)
            if n10 is not None: n10s.append(n10)
        mean_auc = np.mean(aucs)
        print(f"{alpha:6.1f} | {mean_auc:8.4f} | {np.mean(mrrs):8.4f} | {np.mean(n5s):8.4f} | {np.mean(n10s):8.4f}")
        if mean_auc > best_auc:
            best_auc, best_alpha = mean_auc, alpha

    print(f"\nBest alpha by AUC: {best_alpha:.1f} (AUC={best_auc:.4f})")
    print(f"Use this in generate_mind_submission.py: --method hybrid --alpha {best_alpha:.1f}")


if __name__ == "__main__":
    main()