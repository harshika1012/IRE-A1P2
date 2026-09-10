#!/usr/bin/env python
"""Q3.5: compare lexical (BM25) vs semantic retrieval, overall and by slice.

Run run_bm25.py and run_semantic.py first (same --dataset/--split) so their
results/*_impressions.parquet + *_summary.json files exist, then:

    python compare_results.py --dataset mind --split val
"""
import argparse
import json
import pandas as pd
from src.config import ROOT

RESULTS_DIR = ROOT / "results"
K_VALUES = [50, 100, 200]

# cold-start threshold: users at or below the 25th percentile of n_clicks are
# "cold-start". Defined per dataset (not one shared constant) since MIND and
# EB-NeRD have very different engagement distributions -- MIND's 25th
# percentile is ~8 clicks, EB-NeRD's is ~35. Recompute per dataset with
# check_clicks_distribution.py if you rebuild the pipeline with different data.
COLD_START_THRESHOLD = {
    "mind": 8,
    "ebnerd": 35,
}


def load_summary(method, dataset, split):
    path = RESULTS_DIR / f"{method}_{dataset}_{split}_summary.json"
    if not path.exists():
        print(f"  [missing] {path} -- run run_{method}.py --dataset {dataset} --split {split} first")
        return None
    with open(path) as f:
        return json.load(f)


def load_impressions(method, dataset, split):
    path = RESULTS_DIR / f"{method}_{dataset}_{split}_impressions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def compare(dataset: str, split: str):
    print(f"\n=== {dataset} / {split}: lexical vs semantic ===")
    bm25_summary = load_summary("bm25", dataset, split)
    sem_summary = load_summary("semantic", dataset, split)
    if bm25_summary is None or sem_summary is None:
        return

    print(f"\n  Overall recall@K:")
    print(f"  {'K':>5} | {'BM25':>10} | {'Semantic':>10} | {'Winner':>10}")
    for k in K_VALUES:
        b = bm25_summary["recall"][str(k)]
        s = sem_summary["recall"][str(k)]
        winner = "semantic" if s > b else ("bm25" if b > s else "tie")
        print(f"  {k:>5} | {b:>10.4f} | {s:>10.4f} | {winner:>10}")

    # slice comparison: cold-start (few train clicks) vs warm users
    bm25_imp = load_impressions("bm25", dataset, split)
    sem_imp = load_impressions("semantic", dataset, split)
    if bm25_imp is None or sem_imp is None:
        print("\n  (per-impression parquet files missing, skipping slice breakdown)")
        return

    for label, cond in [
        ("cold-start (<=%d train clicks)" % COLD_START_THRESHOLD[dataset],
         lambda df: df["n_clicks"] <= COLD_START_THRESHOLD[dataset]),
        ("warm (>%d train clicks)" % COLD_START_THRESHOLD[dataset],
         lambda df: df["n_clicks"] > COLD_START_THRESHOLD[dataset]),
    ]:
        print(f"\n  Slice: {label}")
        print(f"  {'K':>5} | {'BM25':>10} | {'Semantic':>10} | {'n (bm25/sem)':>14}")
        b_slice = bm25_imp[cond(bm25_imp)]
        s_slice = sem_imp[cond(sem_imp)]
        for k in K_VALUES:
            b_recall = b_slice[f"hit_{k}"].mean() if len(b_slice) else float("nan")
            s_recall = s_slice[f"hit_{k}"].mean() if len(s_slice) else float("nan")
            print(f"  {k:>5} | {b_recall:>10.4f} | {s_recall:>10.4f} | "
                  f"{len(b_slice):>6}/{len(s_slice):<6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        compare(ds, args.split)


if __name__ == "__main__":
    main()