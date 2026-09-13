#!/usr/bin/env python
"""Q3: reproduce the NRMS baseline, then beat it with one principled change.

1. Reproduce: title-only NRMS (word embedding -> multi-head self-attention
   -> additive attention, for both the news encoder and the user encoder;
   see src/nrms.py for the full architecture note).
2. Improve: category-aware news encoding (use_category=True) -- the only
   difference from the baseline model, everything else (architecture size,
   training data, hyperparameters, seed) held identical.
3. Ablation: both models are trained on the exact same sampled impressions
   and evaluated on the exact same val/test candidates, so the val/test
   metric gap is attributable to that one change.
4. Statistical significance: a PAIRED bootstrap over the per-impression
   metric differences (improved - baseline); the reported gain is only
   claimed as significant where that CI excludes zero.

Requires data/features/behavioral_features.parquet and
click_history_features.parquet (Assignment 2 Q1) -- run
build_behavioral_features.py first.

Usage:
    python run_nrms.py --dataset mind
    python run_nrms.py --dataset all --epochs 5 --max-train-impressions 5000
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.nrms import NRMS, Vocab, CategoryVocab, NRMSExampleDataset
from src.reranker import evaluate_scores
from src.bootstrap import bootstrap_ci

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_ids(ids, max_n, seed=42):
    ids = list(ids)
    if max_n and len(ids) > max_n:
        random.seed(seed)
        ids = random.sample(ids, max_n)
    return ids


def load_sampled(dataset: str, max_train_impressions: int, max_eval_impressions: int, seed: int = 42):
    """Same memory-safe pattern as run_reranker.py's load_sampled_candidates:
    sample impression_ids from a cheap 2-column projection FIRST, then read
    the full feature files filtered to just those ids -- MIND alone is
    8.6M+ candidate rows, and reading that in full before sampling is what
    OOM'd Q2's first version."""
    proj = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet",
                            columns=["impression_id", "split"], filters=[("dataset", "==", dataset)])
    proj = proj.drop_duplicates("impression_id")
    sampled_ids = []
    for split_name, n in [("train", max_train_impressions),
                           ("val", max_eval_impressions), ("test", max_eval_impressions)]:
        sampled_ids.extend(sample_ids(proj.loc[proj["split"] == split_name, "impression_id"], n, seed))
    del proj

    row_filter = [("dataset", "==", dataset), ("impression_id", "in", sampled_ids)]
    behavioral = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet", filters=row_filter,
                                  columns=["impression_id", "article_id", "category", "clicked", "split"])
    click_hist = pd.read_parquet(FEATURES_DIR / "click_history_features.parquet", filters=row_filter,
                                  columns=["impression_id", "recent_titles", "recent_categories"])
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=[("dataset", "==", dataset)],
                                columns=["article_id", "title", "category"])
    return behavioral, click_hist, articles


def build_rows(behavioral: pd.DataFrame, click_hist: pd.DataFrame, articles: pd.DataFrame,
               split: str, neg_k, seed: int = 42):
    """neg_k=None -> keep every candidate (eval). neg_k=int -> keep the
    positive + up to neg_k sampled negatives per impression (train), the
    standard lightweight alternative to NRMS's shared-softmax training."""
    hist_lookup = click_hist.set_index("impression_id")
    title_lookup = dict(zip(articles["article_id"].astype(str), articles["title"]))
    rng = random.Random(seed)

    rows = []
    sub = behavioral[behavioral["split"] == split]
    for iid, grp in sub.groupby("impression_id", observed=True):
        if iid not in hist_lookup.index:
            continue
        hrow = hist_lookup.loc[iid]
        history_titles = list(hrow["recent_titles"])
        history_categories = list(hrow["recent_categories"])

        cand_ids = grp["article_id"].astype(str).tolist()
        cand_cats = grp["category"].tolist()
        labels = grp["clicked"].tolist()

        pos_idx = [i for i, l in enumerate(labels) if l == 1]
        neg_idx = [i for i, l in enumerate(labels) if l == 0]
        if neg_k is None:
            keep_idx = range(len(cand_ids))
        else:
            if not pos_idx:
                continue  # nothing to learn from an impression with no positive
            keep_idx = pos_idx + rng.sample(neg_idx, min(neg_k, len(neg_idx)))

        for i in keep_idx:
            rows.append({
                "impression_id": iid,
                "history_titles": history_titles,
                "history_categories": history_categories,
                "candidate_title": title_lookup.get(cand_ids[i], ""),
                "candidate_category": cand_cats[i],
                "label": int(labels[i]),
            })
    return rows


def build_vocab(train_rows, max_vocab_size=20000):
    texts, cats = [], []
    for r in train_rows:
        texts.extend(r["history_titles"]); texts.append(r["candidate_title"])
        cats.extend(r["history_categories"]); cats.append(r["candidate_category"])
    return Vocab(texts, max_size=max_vocab_size), CategoryVocab(cats)


def train_model(rows, vocab, cat_vocab, use_category: bool, epochs: int, batch_size: int,
                 lr: float, embed_dim: int, num_heads: int, max_title_len: int, max_hist_len: int,
                 seed: int = 42):
    torch.manual_seed(seed)
    model = NRMS(len(vocab), len(cat_vocab), embed_dim=embed_dim, num_heads=num_heads,
                 use_category=use_category).to(DEVICE)
    dataset = NRMSExampleDataset(rows, vocab, cat_vocab, max_title_len, max_hist_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch["hist_ids"].to(DEVICE), batch["hist_cats"].to(DEVICE),
                            batch["hist_mask"].to(DEVICE), batch["cand_ids"].to(DEVICE),
                            batch["cand_cat"].to(DEVICE))
            loss = loss_fn(logits, batch["label"].to(DEVICE))
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1
        tag = "improved" if use_category else "baseline"
        print(f"    [{tag}] epoch {epoch + 1}/{epochs}: loss={total_loss / max(n_batches, 1):.4f}")
    return model


@torch.no_grad()
def score_rows(model, rows, vocab, cat_vocab, max_title_len: int, max_hist_len: int) -> np.ndarray:
    """Scores every row, encoding each impression's history only once and
    reusing that user vector across all of its candidates."""
    model.eval()
    ds = NRMSExampleDataset(rows, vocab, cat_vocab, max_title_len, max_hist_len)
    by_impression = defaultdict(list)
    for i, r in enumerate(rows):
        by_impression[r["impression_id"]].append(i)

    scores = np.zeros(len(rows), dtype=np.float64)
    for idxs in by_impression.values():
        encoded = [ds._encode_row(rows[j]) for j in idxs]
        hist = encoded[0]
        hist_ids = hist["hist_ids"].unsqueeze(0).to(DEVICE)
        hist_cats = hist["hist_cats"].unsqueeze(0).to(DEVICE)
        hist_mask = hist["hist_mask"].unsqueeze(0).to(DEVICE)
        user_vec = model.encode_user(hist_ids, hist_cats, hist_mask)

        cand_ids = torch.stack([e["cand_ids"] for e in encoded]).to(DEVICE)
        cand_cats = torch.stack([e["cand_cat"] for e in encoded]).to(DEVICE)
        cand_vecs = model.news_encoder(cand_ids, cand_cats)
        s = (cand_vecs * user_vec).sum(dim=-1).cpu().numpy()
        for k, j in enumerate(idxs):
            scores[j] = s[k]
    return scores


def report_metric(name, series):
    vals = series.dropna().values
    mean, lo, hi = bootstrap_ci(vals)
    if mean is None:
        print(f"    {name:<8} = n/a")
        return None
    print(f"    {name:<8} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}


def paired_significance(base_metrics: pd.DataFrame, improved_metrics: pd.DataFrame, metric: str):
    """PAIRED bootstrap over per-impression (improved - baseline) deltas --
    only impressions where BOTH models produced a defined metric value.
    CI excluding zero = statistically significant at 95%."""
    merged = base_metrics[["impression_id", metric]].merge(
        improved_metrics[["impression_id", metric]], on="impression_id", suffixes=("_base", "_improved"))
    merged = merged.dropna()
    diff = (merged[f"{metric}_improved"] - merged[f"{metric}_base"]).values
    mean, lo, hi = bootstrap_ci(diff)
    if mean is None:
        print(f"    {metric:<8} delta = n/a")
        return None
    significant = (lo > 0) or (hi < 0)
    tag = "SIGNIFICANT" if significant else "not significant"
    print(f"    {metric:<8} delta = {mean:+.4f}  (95% CI: [{lo:+.4f}, {hi:+.4f}], n={len(diff)})  [{tag}]")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diff), "significant": significant}


def run(dataset: str, args):
    print(f"\n=== Q3 NRMS: {dataset} ===")
    behavioral, click_hist, articles = load_sampled(
        dataset, args.max_train_impressions, args.max_eval_impressions, args.seed)

    print("  building examples...")
    train_rows = build_rows(behavioral, click_hist, articles, "train", args.neg_k, args.seed)
    val_rows = build_rows(behavioral, click_hist, articles, "val", None, args.seed)
    test_rows = build_rows(behavioral, click_hist, articles, "test", None, args.seed)
    print(f"  train: {len(train_rows)} examples, val: {len(val_rows)} candidates, test: {len(test_rows)} candidates")

    vocab, cat_vocab = build_vocab(train_rows, args.max_vocab_size)
    print(f"  vocab: {len(vocab)} words, {len(cat_vocab)} categories (built from train split only)")

    kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, embed_dim=args.embed_dim,
                  num_heads=args.num_heads, max_title_len=args.max_title_len, max_hist_len=args.max_hist_len,
                  seed=args.seed)
    print("  training baseline (title-only NRMS)...")
    baseline_model = train_model(train_rows, vocab, cat_vocab, use_category=False, **kwargs)
    print("  training improved (category-aware NRMS)...")
    improved_model = train_model(train_rows, vocab, cat_vocab, use_category=True, **kwargs)

    report = {}
    for split_name, rows in [("val", val_rows), ("test", test_rows)]:
        if not rows:
            continue
        eval_df = pd.DataFrame([{"impression_id": r["impression_id"], "clicked": r["label"]} for r in rows])
        eval_df["baseline_score"] = score_rows(baseline_model, rows, vocab, cat_vocab,
                                                args.max_title_len, args.max_hist_len)
        eval_df["improved_score"] = score_rows(improved_model, rows, vocab, cat_vocab,
                                                args.max_title_len, args.max_hist_len)

        print(f"\n  -- {split_name} ({eval_df['impression_id'].nunique()} impressions) --")
        print("  BASELINE (title-only NRMS):")
        base_metrics = evaluate_scores(eval_df, "baseline_score")
        base_report = {m: report_metric(m, base_metrics[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        print("  IMPROVED (+ category-aware encoding):")
        improved_metrics = evaluate_scores(eval_df, "improved_score")
        improved_report = {m: report_metric(m, improved_metrics[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        print("  ABLATION -- paired bootstrap 95% CI on (improved - baseline):")
        deltas = {m: paired_significance(base_metrics, improved_metrics, m)
                  for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        report[split_name] = {"baseline": base_report, "improved": improved_report, "delta": deltas}

    with open(RESULTS_DIR / f"nrms_{dataset}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/nrms_{dataset}_summary.json")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--max-train-impressions", type=int, default=3000)
    ap.add_argument("--max-eval-impressions", type=int, default=1000)
    ap.add_argument("--neg-k", type=int, default=4, help="sampled negatives per positive during training")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--max-title-len", type=int, default=20)
    ap.add_argument("--max-hist-len", type=int, default=20)
    ap.add_argument("--max-vocab-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(ds, args)


if __name__ == "__main__":
    main()
