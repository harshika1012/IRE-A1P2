#!/usr/bin/env python
"""Q2: two-stage retrieve-then-rank pipeline.

1. Retrieval: score each impression's official candidate set with BM25 +
   semantic similarity (Assignment 1's candidate generators, queried from
   Assignment 2 Q1's point-in-time click history) and keep the top-K.
2. Re-rank: train a LightGBM LambdaMART ranker over Q1's engineered
   behavioural features plus the two retrieval scores.
3. Report AUC/MRR/nDCG@5/nDCG@10 before (Stage 1's own score) and after
   (the trained reranker's score), with bootstrap 95% CI, on val and test.

Requires data/features/behavioral_features.parquet and
data/features/click_history_features.parquet (Assignment 2 Q1) --
run build_behavioral_features.py first.

Usage:
    python run_reranker.py --dataset mind
    python run_reranker.py --dataset all --k 150 --primary-method semantic
"""
import argparse
import gc
import json
import random
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.reranker import (
    build_bm25_index, build_semantic_index, add_retrieval_scores, cap_top_k,
    train_ranker, score_ranker, evaluate_scores,
)
from src.bootstrap import bootstrap_ci

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sample_impression_ids(ids, max_impressions, seed: int = 42):
    ids = list(ids)
    if max_impressions and len(ids) > max_impressions:
        random.seed(seed)
        ids = random.sample(ids, max_impressions)
    return ids


def load_sampled_candidates(dataset: str, max_train_impressions: int, max_eval_impressions: int):
    """Picks which impressions to use BEFORE reading their features, then
    reads behavioral_features.parquet / click_history_features.parquet with
    a pyarrow filter on that exact impression_id list.

    behavioral_features.parquet holds every candidate row for BOTH datasets
    (9M+ total; MIND alone is ~8.6M) -- reading it in full and sampling down
    to a few thousand impressions afterwards, as an earlier version of this
    script did, means paying MIND's full memory cost before ever discarding
    anything. A lightweight 2-column projection (just impression_id + split)
    is cheap enough to read in full, so sampling can happen on that instead,
    and the real (23-column) read only ever touches the sampled rows.
    """
    proj = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet",
                            columns=["impression_id", "split"],
                            filters=[("dataset", "==", dataset)])
    proj = proj.drop_duplicates("impression_id")

    sampled_ids = []
    for split_name, n in [("train", max_train_impressions),
                           ("val", max_eval_impressions), ("test", max_eval_impressions)]:
        ids = proj.loc[proj["split"] == split_name, "impression_id"].tolist()
        sampled_ids.extend(sample_impression_ids(ids, n))
    del proj

    row_filter = [("dataset", "==", dataset), ("impression_id", "in", sampled_ids)]
    behavioral = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet", filters=row_filter)
    click_hist = pd.read_parquet(FEATURES_DIR / "click_history_features.parquet", filters=row_filter)
    return behavioral, click_hist


def report_metric(name, series):
    vals = series.dropna().values
    mean, lo, hi = bootstrap_ci(vals)
    if mean is None:
        print(f"    {name:<8} = n/a (no valid data)")
        return None
    print(f"    {name:<8} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}


def run(dataset: str, k: int, primary_method: str, max_train_impressions: int, max_eval_impressions: int):
    print(f"\n=== Q2 reranker: {dataset} (K={k}, Stage-1 method={primary_method}) ===")
    combined_raw, click_hist = load_sampled_candidates(dataset, max_train_impressions, max_eval_impressions)
    split_counts = combined_raw["split"].value_counts()
    print(f"  sampled {combined_raw['impression_id'].nunique()} impressions "
          f"({split_counts.get('train', 0)} train / {split_counts.get('val', 0)} val / "
          f"{split_counts.get('test', 0)} test candidate rows)")

    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=[("dataset", "==", dataset)])
    print("  building BM25 + semantic indexes...")
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    semantic_index, id_to_embedding = build_semantic_index(articles, dataset)

    print("  Stage 1: scoring each impression's candidates with both retrievers...")
    scored = add_retrieval_scores(combined_raw, click_hist, bm25_index, bm25_lookup,
                                   semantic_index, id_to_embedding)
    # only `scored` is needed from here on -- these were sizeable full copies
    # (one row per candidate) and Stage 2 (LightGBM) needs its own headroom
    del combined_raw, click_hist
    gc.collect()

    primary_col = "bm25_score" if primary_method == "bm25" else "semantic_score"
    capped = cap_top_k(scored, primary_col, k)
    print(f"  kept top-{k}: {len(scored)} -> {len(capped)} candidate rows")
    del scored
    gc.collect()

    train_df = capped[capped["split"] == "train"]
    val_df = capped[capped["split"] == "val"]
    test_df = capped[capped["split"] == "test"]
    del capped
    gc.collect()

    print(f"  Stage 2: training LightGBM LambdaMART on {train_df['impression_id'].nunique()} "
          f"train impressions ({len(train_df)} rows)...")
    ranker = train_ranker(train_df)
    ranker.booster_.save_model(str(RESULTS_DIR / f"reranker_{dataset}_model.txt"))
    del train_df
    gc.collect()

    report = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if split_df.empty:
            continue
        split_df = split_df.copy()
        split_df["reranker_score"] = score_ranker(ranker, split_df)

        print(f"\n  -- {split_name} ({split_df['impression_id'].nunique()} impressions) --")
        print(f"  BEFORE re-ranking ({primary_method} score alone):")
        before = evaluate_scores(split_df, primary_col)
        before_metrics = {m: report_metric(m, before[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        print("  AFTER re-ranking (LightGBM LambdaMART):")
        after = evaluate_scores(split_df, "reranker_score")
        after_metrics = {m: report_metric(m, after[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        report[split_name] = {"before": before_metrics, "after": after_metrics, "n_impressions": len(before)}

    with open(RESULTS_DIR / f"reranker_{dataset}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/reranker_{dataset}_summary.json, "
          f"results/reranker_{dataset}_model.txt")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--k", type=int, default=150, help="top-K kept after Stage 1 retrieval (spec: 100-200)")
    ap.add_argument("--primary-method", choices=["bm25", "semantic"], default="semantic",
                     help="which Assignment 1 candidate generator is Stage 1 / the 'before' baseline")
    ap.add_argument("--max-train-impressions", type=int, default=20000)
    ap.add_argument("--max-eval-impressions", type=int, default=5000)
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(ds, args.k, args.primary_method, args.max_train_impressions, args.max_eval_impressions)


if __name__ == "__main__":
    main()
