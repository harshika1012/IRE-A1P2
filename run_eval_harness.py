#!/usr/bin/env python
"""Q4: offline evaluation harness -- AUC, MRR, nDCG@5/10, beyond-accuracy,
cold-start/warm slicing, bootstrap 95% CI. Runs on either BM25 or semantic
scores, re-ranking each impression's OWN candidate set (unlike Q2/Q3's
full-corpus recall@K).

Usage:
    python run_eval_harness.py --method bm25 --dataset mind --split val
    python run_eval_harness.py --method semantic --dataset all --split val --max-impressions -1
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
from src.beyond_accuracy import intra_list_diversity, novelty, catalog_coverage
from src.bootstrap import bootstrap_ci

COLD_START_THRESHOLD = {"mind": 8, "ebnerd": 35}  # from Q3's per-dataset 25th-percentile analysis
RECLIST_K = 10  # length of the recommended list used for diversity/novelty/coverage


def build_bm25_index(articles, dataset):
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    index = BM25Index()
    index.fit(sub["article_id"].tolist(), sub["text"].tolist())
    article_lookup = dict(zip(sub["article_id"], sub["text"]))
    return index, article_lookup


def build_semantic_index(articles, dataset):
    if embeddings_exist(dataset):
        article_ids, embeddings = load_embeddings(dataset)
    else:
        sub = articles[articles["dataset"] == dataset].copy()
        sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
        article_ids, embeddings = compute_embeddings(sub["article_id"].tolist(), sub["text"].tolist())
        save_embeddings(dataset, article_ids, embeddings)
    index = ANNIndex().fit(article_ids, embeddings)
    id_to_embedding = dict(zip(article_ids, embeddings))
    return index, id_to_embedding


def compute_train_popularity(impressions, dataset):
    train = impressions[(impressions["dataset"] == dataset) &
                         (impressions["split"] == "train") & (impressions["clicked"] == 1)]
    counts = train["article_id"].value_counts().to_dict()
    return counts, int(train.shape[0])


def evaluate(method: str, dataset: str, split: str, max_impressions):
    print(f"\n=== {method} / {dataset} / {split} ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet")
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet")

    if method == "bm25":
        index, lookup = build_bm25_index(articles, dataset)
    elif method == "semantic":
        index, lookup = build_semantic_index(articles, dataset)  # lookup = id_to_embedding here
    else:
        raise ValueError(method)

    # for diversity we always want embeddings regardless of scoring method
    if embeddings_exist(dataset):
        emb_ids, emb_vecs = load_embeddings(dataset)
        id_to_embedding = dict(zip(emb_ids, emb_vecs))
    else:
        id_to_embedding = {}

    uf = user_features[user_features["dataset"] == dataset]
    click_hist = uf.set_index("user_id")["click_history"].to_dict()
    n_clicks_lookup = uf.set_index("user_id")["n_clicks"].to_dict()

    popularity, total_train_clicks = compute_train_popularity(impressions, dataset)
    catalog_size = (articles["dataset"] == dataset).sum()

    imp = impressions[(impressions["dataset"] == dataset) & (impressions["split"] == split)]
    grouped = imp.groupby("impression_id")
    impression_ids = list(grouped.groups.keys())
    if max_impressions and len(impression_ids) > max_impressions:
        rng = np.random.default_rng(42)
        impression_ids = list(rng.choice(impression_ids, size=max_impressions, replace=False))
    print(f"  evaluating {len(impression_ids)} impressions")

    rows = {"impression_id": [], "user_id": [], "n_clicks": [],
            "auc": [], "mrr": [], "ndcg5": [], "ndcg10": [],
            "diversity": [], "novelty": []}
    all_reclists = []
    n_skipped_no_query = 0
    n_skipped_degenerate = 0

    for iid in impression_ids:
        grp = grouped.get_group(iid)
        uid = grp["user_id"].iloc[0]
        candidates = grp["article_id"].tolist()
        labels = grp["clicked"].tolist()

        if len(candidates) < 2 or (sum(labels) == 0):
            n_skipped_degenerate += 1
            continue

        history = click_hist.get(uid, [])
        if method == "bm25":
            query = build_query(history, lookup)
            if not query:
                n_skipped_no_query += 1
                continue
            scores_dict = index.score_docs(candidates, query)
        else:
            user_vec = build_user_embedding(history, lookup)
            if user_vec is None:
                n_skipped_no_query += 1
                continue
            scores_dict = index.score_docs(candidates, user_vec)

        scores = [scores_dict.get(c, 0.0) for c in candidates]

        a = auc(scores, labels)
        m = mrr(scores, labels)
        n5 = ndcg_at_k(scores, labels, 5)
        n10 = ndcg_at_k(scores, labels, 10)

        order = np.argsort(-np.array(scores))
        reclist = [candidates[i] for i in order[:RECLIST_K]]
        div = intra_list_diversity(reclist, id_to_embedding)
        nov = novelty(reclist, popularity, total_train_clicks)
        all_reclists.append(reclist)

        rows["impression_id"].append(iid); rows["user_id"].append(uid)
        rows["n_clicks"].append(n_clicks_lookup.get(uid, 0))
        rows["auc"].append(a); rows["mrr"].append(m)
        rows["ndcg5"].append(n5); rows["ndcg10"].append(n10)
        rows["diversity"].append(div); rows["novelty"].append(nov)

    print(f"  skipped {n_skipped_no_query} (no query/embedding), "
          f"{n_skipped_degenerate} (degenerate: <2 candidates or all-same-label)")

    df = pd.DataFrame(rows)
    coverage = catalog_coverage(all_reclists, catalog_size)

    def report_metric(name, series):
        vals = series.dropna().values
        mean, lo, hi = bootstrap_ci(vals)
        if mean is None:
            print(f"  {name:<10} = n/a (no valid data for this metric)")
            return None, None, None
        print(f"  {name:<10} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
        return mean, lo, hi

    print("\n  Accuracy metrics:")
    metrics = {}
    for name in ["auc", "mrr", "ndcg5", "ndcg10"]:
        metrics[name] = report_metric(name, df[name])

    print("\n  Beyond-accuracy:")
    metrics["diversity"] = report_metric("diversity", df["diversity"])
    metrics["novelty"] = report_metric("novelty", df["novelty"])
    print(f"  {'coverage':<10} = {coverage:.4f}  (catalog: {catalog_size} articles)")

    print(f"\n  Slice: cold-start (<= {COLD_START_THRESHOLD[dataset]} train clicks)")
    cold = df[df["n_clicks"] <= COLD_START_THRESHOLD[dataset]]
    warm = df[df["n_clicks"] > COLD_START_THRESHOLD[dataset]]
    for label, sub in [("cold-start", cold), ("warm", warm)]:
        print(f"    {label} (n={len(sub)}):")
        for name in ["auc", "mrr", "ndcg5", "ndcg10"]:
            vals = sub[name].dropna().values
            if len(vals):
                mean, lo, hi = bootstrap_ci(vals)
                print(f"      {name:<10} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
            else:
                print(f"      {name:<10} = n/a (no data in this slice)")

    return df, metrics, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bm25", "semantic"], required=True)
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-impressions", type=int, default=5000)
    args = ap.parse_args()

    max_imp = None if args.max_impressions == -1 else args.max_impressions
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        evaluate(args.method, ds, args.split, max_imp)


if __name__ == "__main__":
    main()