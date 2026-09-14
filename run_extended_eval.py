#!/usr/bin/env python
"""Q5: extended evaluation of the FULL two-stage pipeline (Q2's BM25/semantic
retrieval + top-K capping + LightGBM re-rank) -- every accuracy metric
(AUC, MRR, nDCG@5, nDCG@10) plus beyond-accuracy (intra-list diversity,
novelty, catalog coverage), sliced by cold-start-vs-warm AND head-vs-tail,
with bootstrap 95% CI on every reported number.

This deliberately runs on the same labeled small/demo val+test splits used
throughout Q1-Q4 (Assignment 1's Codabench test sets ship with NO labels,
so AUC/MRR/nDCG are undefined there by construction -- see A1's own brief:
"the test set has 13.5M/2.37M impressions with no click labels"). The
Codabench SUBMISSION itself is a separate deliverable (predictions on the
large, unlabeled official test set) -- see generate_codabench_submission.py.

Requires results/reranker_<dataset>_model.txt (Q2's saved model) and
data/features/{behavioral_features,click_history_features}.parquet (Q1) --
run build_behavioral_features.py and run_reranker.py first.

Usage:
    python run_extended_eval.py --dataset mind
    python run_extended_eval.py --dataset all --max-eval-impressions 3000
"""
import argparse
import json
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.reranker import build_bm25_index, build_semantic_index, add_retrieval_scores, cap_top_k, FEATURE_COLS
from src.ranking_metrics import auc, mrr, ndcg_at_k
from src.beyond_accuracy import intra_list_diversity, novelty, catalog_coverage
from src.bootstrap import bootstrap_ci

RESULTS_DIR = ROOT / "results"
RECLIST_K = 10
# same per-dataset cold-start thresholds A1's run_eval_harness.py used (its own
# Q3 25th-percentile analysis), kept identical here so the two are comparable
COLD_START_THRESHOLD = {"mind": 8, "ebnerd": 35}
HEAD_PERCENTILE = 0.8  # articles at/above this train-click-count percentile = "head"


def sample_ids(ids, max_n, seed=42):
    ids = list(ids)
    if max_n and len(ids) > max_n:
        random.seed(seed)
        ids = random.sample(ids, max_n)
    return ids


def load_sampled(dataset: str, split: str, max_impressions: int, seed: int = 42):
    """Same memory-safe pattern as run_reranker.py: sample impression_ids
    from a cheap 2-column projection first, then read the full feature
    files filtered to just those ids."""
    proj = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet",
                            columns=["impression_id", "split"], filters=[("dataset", "==", dataset)])
    ids = sample_ids(proj.loc[proj["split"] == split, "impression_id"].drop_duplicates(), max_impressions, seed)
    del proj

    row_filter = [("dataset", "==", dataset), ("impression_id", "in", ids)]
    behavioral = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet", filters=row_filter)
    click_hist = pd.read_parquet(FEATURES_DIR / "click_history_features.parquet", filters=row_filter,
                                  columns=["impression_id", "recent_article_ids"])
    return behavioral, click_hist


def load_train_popularity(dataset: str):
    """Narrow projection (article_id/clicked/split only) so this doesn't
    pay impressions.parquet's full ~10-column cost for a single aggregate --
    see run_serving_benchmark.py's identical fix for why that matters."""
    imp = pd.read_parquet(PROCESSED_DIR / "impressions.parquet", filters=[("dataset", "==", dataset)],
                           columns=["article_id", "clicked", "split"])
    train = imp[imp["split"] == "train"]
    popularity = train.groupby("article_id")["clicked"].sum().to_dict()
    total_train_clicks = int(train["clicked"].sum())
    return popularity, total_train_clicks


def prepare_X(df: pd.DataFrame, feature_cols=FEATURE_COLS) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in ("is_cold_start", "category_match"):
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X


def evaluate_full_pipeline(df: pd.DataFrame, popularity: dict, total_train_clicks: int,
                            id_to_embedding: dict, cold_start_threshold: int):
    """One row per impression: accuracy metrics over ALL its (capped)
    candidates, beyond-accuracy over its top-RECLIST_K reranked list, plus
    the two slice keys (is_cold_start, is_head)."""
    head_cutoff = np.quantile(list(popularity.values()), HEAD_PERCENTILE) if popularity else 0
    rows = []
    all_reclists = []
    for iid, grp in df.groupby("impression_id", observed=True):
        scores = grp["reranker_score"].to_numpy()
        labels = grp["clicked"].to_numpy()
        if len(scores) < 2 or labels.sum() == 0:
            continue

        order = np.argsort(-scores)
        reclist = grp["article_id"].to_numpy()[order][:RECLIST_K].tolist()
        all_reclists.append(reclist)

        clicked_article = grp.loc[grp["clicked"] == 1, "article_id"].iloc[0]
        is_head = popularity.get(clicked_article, 0) >= head_cutoff

        rows.append({
            "impression_id": iid,
            "auc": auc(scores, labels), "mrr": mrr(scores, labels),
            "ndcg5": ndcg_at_k(scores, labels, 5), "ndcg10": ndcg_at_k(scores, labels, 10),
            "diversity": intra_list_diversity(reclist, id_to_embedding),
            "novelty": novelty(reclist, popularity, total_train_clicks),
            "is_cold_start": bool(grp["n_clicks_before"].iloc[0] <= cold_start_threshold),
            "is_head": bool(is_head),
        })
    return pd.DataFrame(rows), all_reclists


def report_metric(name, series, indent="    "):
    vals = series.dropna().values
    mean, lo, hi = bootstrap_ci(vals)
    if mean is None:
        print(f"{indent}{name:<10} = n/a (no valid data, n=0)")
        return None
    print(f"{indent}{name:<10} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}


def report_slice(label, sub_df, indent="    "):
    print(f"{indent}{label} (n={len(sub_df)}):")
    metrics = {}
    for name in ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]:
        metrics[name] = report_metric(name, sub_df[name], indent=indent + "  ")
    return metrics


def run(dataset: str, k: int, primary_method: str, max_eval_impressions: int, catalog_size: int):
    print(f"\n=== Q5 extended evaluation: {dataset} ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=[("dataset", "==", dataset)])
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    semantic_index, id_to_embedding = build_semantic_index(articles, dataset)
    popularity, total_train_clicks = load_train_popularity(dataset)
    booster = lgb.Booster(model_file=str(RESULTS_DIR / f"reranker_{dataset}_model.txt"))
    primary_col = "bm25_score" if primary_method == "bm25" else "semantic_score"

    report = {}
    for split in ("val", "test"):
        behavioral, click_hist = load_sampled(dataset, split, max_eval_impressions)
        if behavioral.empty:
            continue
        scored = add_retrieval_scores(behavioral, click_hist, bm25_index, bm25_lookup,
                                       semantic_index, id_to_embedding)
        capped = cap_top_k(scored, primary_col, k)
        capped = capped.copy()
        capped["reranker_score"] = booster.predict(prepare_X(capped))

        per_imp, all_reclists = evaluate_full_pipeline(
            capped, popularity, total_train_clicks, id_to_embedding,
            COLD_START_THRESHOLD.get(dataset, 8))

        print(f"\n  -- {split} ({len(per_imp)} scored impressions) --")
        print("  Overall:")
        overall = report_slice("overall", per_imp)
        coverage = catalog_coverage(all_reclists, catalog_size)
        print(f"      {'coverage':<10} = {coverage:.4f}  (catalog: {catalog_size} articles)")

        print("\n  Slice: cold-start vs warm")
        cold = report_slice("cold-start", per_imp[per_imp["is_cold_start"]])
        warm = report_slice("warm", per_imp[~per_imp["is_cold_start"]])

        print("\n  Slice: head vs tail (by the ground-truth clicked article's train popularity)")
        head = report_slice("head", per_imp[per_imp["is_head"]])
        tail = report_slice("tail", per_imp[~per_imp["is_head"]])

        report[split] = {
            "n_impressions": len(per_imp), "overall": overall, "coverage": coverage,
            "cold_start_vs_warm": {"cold_start": cold, "warm": warm},
            "head_vs_tail": {"head": head, "tail": tail},
        }

    with open(RESULTS_DIR / f"extended_eval_{dataset}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/extended_eval_{dataset}_summary.json")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--k", type=int, default=150, help="top-K kept after Stage 1 retrieval (spec: 100-200)")
    ap.add_argument("--primary-method", choices=["bm25", "semantic"], default="semantic")
    ap.add_argument("--max-eval-impressions", type=int, default=3000)
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        catalog_size = pd.read_parquet(
            PROCESSED_DIR / "articles.parquet", columns=["article_id"], filters=[("dataset", "==", ds)]
        ).shape[0]
        run(ds, args.k, args.primary_method, args.max_eval_impressions, catalog_size)


if __name__ == "__main__":
    main()
