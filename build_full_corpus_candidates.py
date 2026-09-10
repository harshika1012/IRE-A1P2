#!/usr/bin/env python
"""Assignment 2, Q2 step 1 (strict/literal version) -- run this AFTER
build_pipeline.py has already produced impressions_with_features.parquet
(i.e. after Q1's step 5 has run, since we need `recent_article_ids` and the
raw `history`/`articles` tables to exist).

This does NOT belong inside build_pipeline.py's normal flow: full-corpus
top-K retrieval is expensive (a brute-force search over the whole article
corpus PER IMPRESSION), so you want to control `--max-impressions` while
developing rather than pay this cost on every pipeline rebuild.

What it does:
    1. Loads the Q1-featured impressions + raw history/articles.
    2. Runs build_full_corpus_candidates() -- REAL full-corpus retrieval,
       not a re-ranking of the given candidate list (see
       full_corpus_retrieval.py's docstring for why this is a different,
       stricter thing than candidate_retrieval.add_retrieval_features).
    3. Feeds the resulting candidate table back through the SAME Q1 feature
       functions from click_features.py -- this works unchanged because the
       new table has the identical shape (impression_id, user_id, dataset,
       timestamp, article_id, clicked, position) that those functions expect.
    4. Saves to impressions_full_corpus.parquet.

Point train_reranker.py at this file instead of impressions_full.parquet to
train/evaluate on genuine full-corpus retrieved candidates.

Usage:
    python build_full_corpus_candidates.py --dataset mind --max-impressions 20000
    python build_full_corpus_candidates.py --dataset ebnerd --max-impressions 20000
    python build_full_corpus_candidates.py --dataset mind --max-impressions -1   # full dataset (slow)
"""
import argparse
import pandas as pd

from src.config import PROCESSED_DIR, HYBRID_ALPHA
from src.parse_mind import load_mind
from src.parse_ebnerd import load_ebnerd
from src.full_corpus_retrieval import build_full_corpus_candidates
from src.click_features import (
    add_click_history_features, add_recent_article_ids, add_session_features,
    add_article_dynamic_features, add_category_match_features, assert_no_future_leakage,
)


def _load_raw_history_and_articles(dataset: str):
    if dataset == "mind":
        _, _, h = load_mind()
    else:
        _, _, h = load_ebnerd()
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
    return h, articles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--max-impressions", type=int, default=20000,
                     help="-1 for the full dataset (slow -- full-corpus search per impression)")
    args = ap.parse_args()
    max_imp = None if args.max_impressions == -1 else args.max_impressions

    print(f"== loading Q1-featured impressions for {args.dataset} ==")
    imp = pd.read_parquet(PROCESSED_DIR / "impressions_with_features.parquet",
                           columns=["impression_id", "user_id", "dataset", "timestamp",
                                    "article_id", "clicked", "split", "recent_article_ids"])
    imp = imp[imp["dataset"] == args.dataset]

    print(f"== loading raw history + articles for {args.dataset} ==")
    history, articles = _load_raw_history_and_articles(args.dataset)

    print(f"== stage 1: full-corpus top-K retrieval (k={args.k}) ==")
    candidates = build_full_corpus_candidates(
        imp, articles, dataset=args.dataset, alpha=HYBRID_ALPHA[args.dataset],
        k=args.k, max_impressions=max_imp,
    )
    if candidates.empty:
        print("no candidates produced -- check that recent_article_ids / embeddings exist")
        return

    print("== re-applying Q1 features to the retrieved candidate table ==")
    candidates = add_click_history_features(candidates, history)
    candidates = add_recent_article_ids(candidates, history, articles=articles, k=20)
    candidates = add_session_features(candidates)
    candidates = add_article_dynamic_features(candidates, articles)
    candidates = add_category_match_features(candidates, history, articles)
    assert_no_future_leakage(candidates, history)

    # carry the original impressions' split assignment across (retrieval
    # doesn't change which split an impression belongs to)
    split_lookup = imp.drop_duplicates("impression_id").set_index("impression_id")["split"]
    candidates["split"] = candidates["impression_id"].map(split_lookup)

    out_path = PROCESSED_DIR / f"impressions_full_corpus_{args.dataset}.parquet"
    candidates.to_parquet(out_path, index=False)
    print(f"saved {out_path} ({len(candidates)} candidate rows, "
          f"{candidates['impression_id'].nunique()} impressions)")


if __name__ == "__main__":
    main()