#!/usr/bin/env python
"""Q2: BM25 lexical candidate generation + recall@K.

Usage:
    python run_bm25.py --dataset mind --split val
    python run_bm25.py --dataset ebnerd --split val
    python run_bm25.py --dataset all --split val --max-impressions 5000
"""
import argparse
import random
import json
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.bm25_index import BM25Index
from src.query_builder import build_query

K_VALUES = [50, 100, 200]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_index(articles: pd.DataFrame, dataset: str) -> BM25Index:
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    idx = BM25Index()
    idx.fit(sub["article_id"].tolist(), sub["text"].tolist())
    return idx


def evaluate(dataset: str, split: str, max_impressions: int | None, seed: int = 42):
    print(f"\n=== {dataset} / {split} ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet")
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet")

    articles_ds = articles[articles["dataset"] == dataset]
    article_lookup = dict(zip(
        articles_ds["article_id"],
        (articles_ds["title"].fillna("") + " " + articles_ds["abstract"].fillna(""))
    ))

    uf = user_features[user_features["dataset"] == dataset].set_index("user_id")["click_history"].to_dict()
    n_clicks_lookup = user_features[user_features["dataset"] == dataset].set_index("user_id")["n_clicks"].to_dict()

    imp = impressions[(impressions["dataset"] == dataset) & (impressions["split"] == split)]
    # ground truth: clicked article per impression_id
    truth = imp[imp["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)
    imp_users = imp.drop_duplicates("impression_id").set_index("impression_id")["user_id"]

    impression_ids = list(truth.index)
    if max_impressions and len(impression_ids) > max_impressions:
        random.seed(seed)
        impression_ids = random.sample(impression_ids, max_impressions)
    print(f"  evaluating {len(impression_ids)} impressions "
          f"(of {len(truth)} total with a ground-truth click)")

    print("  building BM25 index...")
    index = build_index(articles, dataset)

    hits = {k: 0 for k in K_VALUES}
    max_k = max(K_VALUES)
    n_scored = 0  # impressions where the user actually had a usable history/query
    n_skipped = 0
    per_impression_rows = []

    for iid in impression_ids:
        uid = imp_users.loc[iid]
        history = uf.get(uid, [])
        query = build_query(history, article_lookup)
        if not query:
            n_skipped += 1
            continue
        n_scored += 1
        results = index.top_k(query, k=max_k)
        retrieved_ids = [aid for aid, _ in results]
        gt = truth.loc[iid]
        row = {"impression_id": iid, "user_id": uid, "n_clicks": n_clicks_lookup.get(uid, 0)}
        for k in K_VALUES:
            hit = bool(gt & set(retrieved_ids[:k]))
            hits[k] += hit
            row[f"hit_{k}"] = hit
        per_impression_rows.append(row)

    print(f"  skipped {n_skipped} impressions (no click history / OOV query -> cold-start)")
    print(f"  {'K':>5} | {'Recall@K':>10}")
    report = {}
    for k in K_VALUES:
        recall = hits[k] / n_scored if n_scored else 0.0
        report[k] = recall
        print(f"  {k:>5} | {recall:>10.4f}")

    # persist for the lexical-vs-semantic comparison in Q3.5
    pd.DataFrame(per_impression_rows).to_parquet(
        RESULTS_DIR / f"bm25_{dataset}_{split}_impressions.parquet", index=False)
    with open(RESULTS_DIR / f"bm25_{dataset}_{split}_summary.json", "w") as f:
        json.dump({"method": "bm25", "dataset": dataset, "split": split,
                    "n_scored": n_scored, "n_skipped": n_skipped, "recall": report}, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--max-impressions", type=int, default=5000,
                     help="Sample size for feasibility. Use -1 for the full split.")
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