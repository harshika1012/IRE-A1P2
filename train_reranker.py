#!/usr/bin/env python
"""Assignment 2, Q2 -- Stage 2: train a LightGBM re-ranker over the Q1
behavioural features + Q2 retrieval features, and report before/after
ranking metrics.

"Before" = retrieval_score alone (the tuned hybrid blend from tune_alpha.py /
candidate_retrieval.py) -- this is your reproduced baseline (Q3 point 1).
"After"  = the trained re-ranker's predicted score -- this is your improvement
(Q3 point 2), and the CI on the paired per-impression metric delta (via
bootstrap.py) is what Q3 point 4 asks for ("claimed gains must ship a paired
bootstrap 95% CI that excludes zero").

Trains ONE model PER DATASET (mind, ebnerd) since their feature distributions
and schemas differ substantially (e.g. freshness_hours is all-NaN for MIND) --
matches how you'll submit to the two Codabench leaderboards separately anyway.

Also trains a second model WITHOUT candidate_position, to satisfy Q9's
"report metrics with and without features unavailable at serving time":
candidate_position is the ORIGINAL platform's own display order for that
candidate. Including it risks the model learning "this was shown first by
the old system" (position bias) rather than genuine relevance -- it's not a
feature your own retrieval stage produces, so a fair from-scratch re-ranker
shouldn't lean on it. We report both so the difference is visible rather than
silently baked in.

Usage:
    python train_reranker.py --dataset mind
    python train_reranker.py --dataset ebnerd
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import PROCESSED_DIR
from src.ranking_metrics import auc, mrr, ndcg_at_k
from src.bootstrap import bootstrap_ci

NUMERIC_FEATURES = [
    "n_clicks_before", "recency_weighted_hist", "days_since_last_click",
    "popularity_before", "freshness_hours", "category_match",
    "session_length", "session_impression_count_before",
    "session_click_count_before", "time_since_previous_click",
    "retrieval_score", "retrieval_rank",
]
CATEGORICAL_FEATURES = ["category"]
POSITION_FEATURE = "candidate_position"  # ablated separately -- see module docstring


REQUIRED_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [POSITION_FEATURE,
                 "clicked", "split", "impression_id", "dataset"]


def _load(dataset: str) -> pd.DataFrame:
    # Only pull the columns the ranker actually needs -- impressions_full.parquet
    # also carries recent_article_ids / history_category_distribution /
    # history_embedding, which are heavy Python list/dict objects per row (huge
    # in-memory overhead vs. on-disk size) and unused here. Reading the whole
    # file was what OOM-killed the previous run.
    df = pd.read_parquet(PROCESSED_DIR / "impressions_full.parquet", columns=REQUIRED_COLS)
    df = df[df["dataset"] == dataset].copy()
    df["category"] = df["category"].astype("category")
    return df


def _prepare(df: pd.DataFrame, feature_cols):
    df = df.sort_values("impression_id")
    X = df[feature_cols].copy()
    y = df["clicked"].astype(int).to_numpy()
    groups = df.groupby("impression_id", sort=False).size().to_numpy()
    impression_ids = df["impression_id"].to_numpy()
    return X, y, groups, impression_ids


def _per_impression_metrics(scores: np.ndarray, y: np.ndarray, impression_ids: np.ndarray):
    """Returns dict of metric_name -> array of per-impression values (paired
    across models since impression_ids/order match if computed on the same df)."""
    out = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}
    order = np.argsort(impression_ids, kind="stable")
    scores, y, impression_ids = scores[order], y[order], impression_ids[order]
    boundaries = np.flatnonzero(np.r_[True, impression_ids[1:] != impression_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        s, l = scores[start:end], y[start:end]
        a, m = auc(s, l), mrr(s, l)
        n5, n10 = ndcg_at_k(s, l, 5), ndcg_at_k(s, l, 10)
        out["auc"].append(a); out["mrr"].append(m)
        out["ndcg5"].append(n5); out["ndcg10"].append(n10)
    return {k: np.array([v for v in vals if v is not None]) for k, vals in out.items()}


def _report(name: str, metrics: dict):
    print(f"  {name:>22}: "
          f"AUC={metrics['auc'].mean():.4f}  MRR={metrics['mrr'].mean():.4f}  "
          f"nDCG@5={metrics['ndcg5'].mean():.4f}  nDCG@10={metrics['ndcg10'].mean():.4f}  "
          f"(n={len(metrics['auc'])})")


def train_and_eval(df: pd.DataFrame, dataset: str, use_position: bool):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ([POSITION_FEATURE] if use_position else [])

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    X_tr, y_tr, g_tr, _ = _prepare(train_df, feature_cols)
    X_val, y_val, g_val, iid_val = _prepare(val_df, feature_cols)

    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
    )
    model.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_val, y_val)], eval_group=[g_val], eval_at=[5, 10],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    val_sorted = val_df.sort_values("impression_id")
    baseline_scores = val_sorted["retrieval_score"].to_numpy()
    reranked_scores = model.predict(X_val.loc[val_sorted.index])

    baseline_metrics = _per_impression_metrics(baseline_scores, y_val, iid_val)
    reranked_metrics = _per_impression_metrics(reranked_scores, y_val, iid_val)

    tag = "with candidate_position" if use_position else "without candidate_position"
    print(f"\n=== {dataset} -- {tag} ===")
    _report("baseline (retrieval)", baseline_metrics)
    _report("re-ranked (LightGBM)", reranked_metrics)

    n = min(len(baseline_metrics["auc"]), len(reranked_metrics["auc"]))
    delta_auc = reranked_metrics["auc"][:n] - baseline_metrics["auc"][:n]
    mean, lo, hi = bootstrap_ci(delta_auc)
    excludes_zero = (lo is not None) and (lo > 0 or hi < 0)
    print(f"  paired AUC delta: mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
          f"excludes zero: {excludes_zero}")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    args = ap.parse_args()
    df = _load(args.dataset)
    train_and_eval(df, args.dataset, use_position=True)
    train_and_eval(df, args.dataset, use_position=False)


if __name__ == "__main__":
    main()