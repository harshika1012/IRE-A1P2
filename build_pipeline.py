#!/usr/bin/env python
"""One-command rebuild: raw files -> feature store.

Usage:
    python build_pipeline.py --dataset all      # MIND + EB-NeRD
    python build_pipeline.py --dataset mind
    python build_pipeline.py --dataset ebnerd
    python build_pipeline.py --dataset all --skip-download   # raw files already present
    python build_pipeline.py --dataset all --with-behavioral # also run Assignment 2 Q1
                                                               # (heaviest step: ~5GB peak RSS
                                                               # on the full mind+ebnerd data,
                                                               # off by default for that reason)
"""
import argparse
import gc
import pandas as pd

from src import download
from src.parse_mind import load_mind
from src.parse_ebnerd import load_ebnerd
from src.temporal_split import temporal_split
from src.feature_store import build_article_features, build_user_features, save
from src.behavioral_features import build_and_save_behavioral_features
from src.config import PROCESSED_DIR, FEATURES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--with-behavioral", action="store_true",
                     help="Also build Assignment 2 Q1 behavioural features (click-history, "
                          "session, popularity/freshness). Off by default: it's the heaviest "
                          "step here (~5GB peak RSS on the full mind+ebnerd data) and only "
                          "needed once you're past Assignment 1's Q2/Q3 retrieval work. Run "
                          "build_behavioral_features.py separately instead if you'd rather "
                          "not slow down every pipeline rebuild.")
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
    # the per-dataset frames concat just copied out of are otherwise dead weight
    # for the rest of the run (this matters once --with-behavioral is heavy on RAM)
    del articles_list, impressions_list, history_list
    gc.collect()

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
    # nothing past this point needs these -- free them before Step 5, which is
    # the memory-heavy step, rather than let them sit alongside it unused
    del article_features, user_features, impressions_train
    gc.collect()

    if args.with_behavioral:
        print("== Step 5: behavioural features (Assignment 2, Part I Q1) ==")
        build_and_save_behavioral_features(impressions, history, articles, FEATURES_DIR)

    print("\nDone. Processed data in data/processed/, features in data/features/")


if __name__ == "__main__":
    main()