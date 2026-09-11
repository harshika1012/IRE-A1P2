#!/usr/bin/env python
"""Assignment 2, Part I Q1: click-history & session features.

Builds on Assignment 1's processed data (data/processed/*.parquet) --
run `python build_pipeline.py --dataset all --skip-download` first if
those don't exist yet, or if you change parse_mind.py / parse_ebnerd.py.

Usage:
    python build_behavioral_features.py --dataset all
"""
import argparse
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR
from src.behavioral_features import build_and_save_behavioral_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = ap.parse_args()

    # a pyarrow predicate filter avoids ever materializing the OTHER dataset's
    # rows in memory, instead of loading everything and then discarding most
    # of it -- matters most for --dataset ebnerd (mind is ~15x bigger).
    row_filter = None if args.dataset == "all" else [("dataset", "==", args.dataset)]
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=row_filter)
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet", filters=row_filter)
    history = pd.read_parquet(PROCESSED_DIR / "history.parquet", filters=row_filter)

    print(f"== building behavioural features for {impressions['impression_id'].nunique()} "
          f"impressions ({len(impressions)} candidate rows) ==")
    behavioral, click_hist, session = build_and_save_behavioral_features(
        impressions, history, articles, FEATURES_DIR)

    print("\n== summary by dataset ==")
    summary_cols = ["n_clicks_before", "recency_weighted_click_count", "is_cold_start",
                     "popularity_prior_ctr", "freshness_hours", "category_match",
                     "session_click_count_before", "avg_dwell_time_before", "position_ctr_prior"]
    print(behavioral.groupby("dataset", observed=True)[summary_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
