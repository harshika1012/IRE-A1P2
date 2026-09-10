#!/usr/bin/env python
"""One-command rebuild: raw files -> feature store.

Usage:
    python build_pipeline.py --dataset all      # MIND + EB-NeRD
    python build_pipeline.py --dataset mind
    python build_pipeline.py --dataset ebnerd
    python build_pipeline.py --dataset all --skip-download   # raw files already present
"""
import argparse
import pandas as pd

from src import download
from src.parse_mind import load_mind
from src.parse_ebnerd import load_ebnerd
from src.temporal_split import temporal_split
from src.feature_store import build_article_features, build_user_features, save
from src.config import PROCESSED_DIR
from src.click_features import (
        add_click_history_features, add_recent_article_ids,
        add_session_features, add_article_dynamic_features,
        add_category_match_features, assert_no_future_leakage,
    )
from src.candidate_retrieval import add_retrieval_features
# from src.full_corpus_retrieval import build_full_corpus_candidates
from src.config import HYBRID_ALPHA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]

    if not args.skip_download:
        print("== Step 1: download ==")
        download.run(datasets)

    print("== Step 2: parse into unified schema ==")
    articles_list, impressions_list, history_list = [], [], []
    if "mind" in datasets:
        a, i, h = load_mind()
        articles_list.append(a); impressions_list.append(i); history_list.append(h)
    if "ebnerd" in datasets:
        a, i, h = load_ebnerd()
        articles_list.append(a); impressions_list.append(i); history_list.append(h)

    articles = pd.concat(articles_list, ignore_index=True)
    impressions = pd.concat(impressions_list, ignore_index=True)
    history = pd.concat(history_list, ignore_index=True)

    articles.to_parquet(PROCESSED_DIR / "articles.parquet", index=False)
    history.to_parquet(PROCESSED_DIR / "history.parquet", index=False)
    print(f"  articles: {len(articles)}, history rows: {len(history)}")

    print("== Step 3: temporal split ==")
    impressions = temporal_split(impressions)
    impressions.to_parquet(PROCESSED_DIR / "impressions.parquet", index=False)

    print("== Step 4: feature store ==")
    article_features = build_article_features(articles)
    impressions_train = impressions[impressions["split"] == "train"]
    user_features = build_user_features(history, impressions_train)
    save(article_features, user_features)

    print("\nDone. Processed data in data/processed/, features in data/features/")
    
    print("== Step 5: Q1 behavioural features ==")
    impressions = add_click_history_features(impressions, history)
    # impressions = add_recent_article_ids(impressions, history, k=20)
    impressions = add_recent_article_ids(impressions, history, articles=articles, k=20)
    impressions = add_session_features(impressions)
    impressions = add_article_dynamic_features(impressions, articles)
    impressions = add_category_match_features(impressions, history, articles)
    assert_no_future_leakage(impressions, history)
    impressions.to_parquet(PROCESSED_DIR / "impressions_with_features.parquet", index=False)

    print("== Step 6: Q2 stage-1 retrieval features ==")
    for ds in datasets:
        impressions = add_retrieval_features(impressions, articles, dataset=ds, alpha=HYBRID_ALPHA[ds])
    impressions.to_parquet(PROCESSED_DIR / "impressions_full.parquet", index=False)

if __name__ == "__main__":
    main()