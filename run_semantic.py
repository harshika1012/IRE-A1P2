#!/usr/bin/env python
"""Q3: semantic candidate generation (embeddings) + recall@K.

Usage:
    python run_semantic.py --dataset mind --split val --max-impressions 5000
    python run_semantic.py --dataset all --split val --max-impressions -1

First run computes and caches embeddings to data/features/embeddings_<dataset>.npz
(slow, downloads+runs the sentence-transformer model). Subsequent runs load the cache.
"""
import argparse
import random
import json
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.embeddings import compute_embeddings, save_embeddings, load_embeddings, embeddings_exist
from src.ann_index import ANNIndex
from src.query_builder import build_user_embedding

K_VALUES = [50, 100, 200]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_or_build_embeddings(articles: pd.DataFrame, dataset: str):
    if embeddings_exist(dataset):
        print(f"  loading cached embeddings for {dataset}...")
        return load_embeddings(dataset)

    print(f"  computing embeddings for {dataset} (one-time, cached after this)...")
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    article_ids, embeddings = compute_embeddings(sub["article_id"].tolist(), sub["text"].tolist())
    save_embeddings(dataset, article_ids, embeddings)
    return article_ids, embeddings


def evaluate(dataset: str, split: str, max_impressions, seed: int = 42):
    print(f"\n=== {dataset} / {split} (semantic) ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet")
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet")

    article_ids, embeddings = get_or_build_embeddings(articles, dataset)
    id_to_embedding = dict(zip(article_ids, embeddings))

    index = ANNIndex().fit(article_ids, embeddings)

    uf = user_features[user_features["dataset"] == dataset].set_index("user_id")["click_history"].to_dict()
    n_clicks_lookup = user_features[user_features["dataset"] == dataset].set_index("user_id")["n_clicks"].to_dict()

    imp = impressions[(impressions["dataset"] == dataset) & (impressions["split"] == split)]
    truth = imp[imp["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)
    imp_users = imp.drop_duplicates("impression_id").set_index("impression_id")["user_id"]

    impression_ids = list(truth.index)
    if max_impressions and len(impression_ids) > max_impressions:
        random.seed(seed)
        impression_ids = random.sample(impression_ids, max_impressions)
    print(f"  evaluating {len(impression_ids)} impressions "
          f"(of {len(truth)} total with a ground-truth click)")

    hits = {k: 0 for k in K_VALUES}
    max_k = max(K_VALUES)
    n_scored = 0
    n_skipped = 0
    per_impression_rows = []

    for iid in impression_ids:
        uid = imp_users.loc[iid]
        history = uf.get(uid, [])
        user_vec = build_user_embedding(history, id_to_embedding)
        if user_vec is None:
            n_skipped += 1
            continue
        n_scored += 1
        results = index.top_k(user_vec, k=max_k)
        retrieved_ids = [aid for aid, _ in results]
        gt = truth.loc[iid]
        row = {"impression_id": iid, "user_id": uid, "n_clicks": n_clicks_lookup.get(uid, 0)}
        for k in K_VALUES:
            hit = bool(gt & set(retrieved_ids[:k]))
            hits[k] += hit
            row[f"hit_{k}"] = hit
        per_impression_rows.append(row)

    print(f"  skipped {n_skipped} impressions (no click history / no embedded articles -> cold-start)")
    print(f"  {'K':>5} | {'Recall@K':>10}")
    report = {}
    for k in K_VALUES:
        recall = hits[k] / n_scored if n_scored else 0.0
        report[k] = recall
        print(f"  {k:>5} | {recall:>10.4f}")

    pd.DataFrame(per_impression_rows).to_parquet(
        RESULTS_DIR / f"semantic_{dataset}_{split}_impressions.parquet", index=False)
    with open(RESULTS_DIR / f"semantic_{dataset}_{split}_summary.json", "w") as f:
        json.dump({"method": "semantic", "dataset": dataset, "split": split,
                    "n_scored": n_scored, "n_skipped": n_skipped, "recall": report}, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--max-impressions", type=int, default=5000)
    args = ap.parse_args()

    max_imp = None if args.max_impressions == -1 else args.max_impressions
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]

    all_reports = {}
    for ds in datasets:
        all_reports[ds] = evaluate(ds, args.split, max_imp)

    print("\n=== Summary ===")
    for ds, report in all_reports.items():
        for k, r in report.items():
            print(f"{ds:8s} recall@{k:<4d} = {r:.4f}")


if __name__ == "__main__":
    main()