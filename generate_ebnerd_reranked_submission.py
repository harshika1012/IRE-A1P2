#!/usr/bin/env python
"""Assignment 2, Q5: Codabench submission using the FULL two-stage pipeline
(Q2's BM25/semantic retrieval + LightGBM re-rank), not just bare retrieval.

Builds on generate_ebnerd_submission.py's proven approach -- same article
corpus loading, same batched streaming of ebnerd_testset/behaviors.parquet,
same output format ("{impression_id} [rank_of_candidate_1,...]" in the
candidates' ORIGINAL order). See that file for the retrieval-only baseline
this extends; this one adds a feature-assembly + rerank step per batch on
top of the same retrieval scoring.

CAVEATS (documented, not hidden -- see conversation for the full reasoning):
  - The reranker (results/reranker_ebnerd_model.txt) was trained on
    EB-NeRD-DEMO-derived features (Assignment 2 Q1/Q2). Applying it to
    ebnerd_large's differently-scaled feature distributions is a genuine
    train/serving shift -- the feature *semantics* match, the *scale*
    doesn't exactly.
  - Unlike MIND, EB-NeRD's history.parquet DOES carry per-item timestamps
    (impression_time_fixed) and read_time_fixed, so recency_weighted_click_
    count and avg_dwell_time_before ARE computed properly here (a real
    advantage over the MIND version's approximation).
  - No live session-scoped counters are reconstructed here (would need a
    global sort-by-session pass over the whole test set, not a per-batch
    computation) -- session_click_count_before/session_impressions_before
    default to 0, same simplification Q4's serving benchmark used.
  - `position` is the candidate's rank under the PRIMARY retrieval method
    (bm25 or semantic) BEFORE re-ranking -- the same legitimate,
    available-before-ranking feature Q2 trained on.
  - popularity_ctr and position_ctr_prior are computed from ebnerd_large's
    LABELED train+validation behaviors (ebnerd_large ships no train/val/
    test split of its own the way our Q1 pipeline built one).

Usage:
    python generate_ebnerd_reranked_submission.py --primary-method semantic \
        --articles data/raw/ebnerd/large/articles.parquet \
        --train-behaviors data/raw/ebnerd/large/train/behaviors.parquet \
        --val-behaviors data/raw/ebnerd/large/validation/behaviors.parquet \
        --test-dir data/raw/ebnerd/testset
"""
import argparse
import io
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import compute_embeddings, save_embeddings, load_embeddings, embeddings_exist
from src.query_builder import build_query, build_user_embedding
from src.reranker import FEATURE_COLS
from src.config import COLD_START_MAX_CLICKS, RECENCY_HALF_LIFE_DAYS, ROOT

N_RECENT = 10  # matches Q1's N_RECENT_CLICKS
BATCH_SIZE = 10_000
RESULTS_DIR = ROOT / "results"


def build_articles(articles_path: Path) -> pd.DataFrame:
    articles = pd.read_parquet(articles_path)
    articles["text"] = (articles["title"].fillna("") + " " + articles["subtitle"].fillna("")).str.strip()
    print(f"  corpus: {len(articles)} articles")
    return articles


def build_bm25(articles: pd.DataFrame) -> BM25Index:
    index = BM25Index()
    index.fit(articles["article_id"].tolist(), articles["text"].tolist())
    return index


def build_semantic(articles: pd.DataFrame) -> ANNIndex:
    if embeddings_exist("ebnerd_large"):
        article_ids, embeddings = load_embeddings("ebnerd_large")
    else:
        print("  computing embeddings for full large corpus (one-time, cached after)...")
        article_ids, embeddings = compute_embeddings(articles["article_id"].tolist(), articles["text"].tolist())
        save_embeddings("ebnerd_large", article_ids, embeddings)
    return ANNIndex().fit(article_ids, embeddings)


def build_user_history_lookup(history_path: Path) -> dict:
    """user_id -> (article_ids, timestamps, read_times) -- the test set's
    own provided history, with real per-item timestamps (unlike MIND)."""
    print(f"  loading user history from {history_path}...")
    hist = pd.read_parquet(history_path,
                            columns=["user_id", "article_id_fixed", "impression_time_fixed", "read_time_fixed"])
    lookup = {
        row.user_id: (row.article_id_fixed, row.impression_time_fixed, row.read_time_fixed)
        for row in hist.itertuples(index=False)
    }
    print(f"  {len(lookup):,} users with history")
    return lookup


def build_popularity_and_position_prior(behaviors_paths):
    """One pass over ebnerd_large's LABELED train+validation behaviors for
    a global article popularity CTR and a position->CTR prior -- the
    large-scale equivalent of Q1's train-only popularity_prior_ctr /
    position_ctr_prior. Per-row loop matches src/parse_ebnerd.py's own
    established pattern for this same array-column format."""
    clicks, impr_count, pos_clicks, pos_impr = {}, {}, {}, {}
    for path in behaviors_paths:
        print(f"  scanning {path} for popularity + position priors...")
        pf = pq.ParquetFile(path)
        for record_batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                             columns=["article_ids_inview", "article_ids_clicked"]):
            batch = record_batch.to_pandas()
            for row in batch.itertuples(index=False):
                inview = row.article_ids_inview
                if inview is None:
                    continue
                clicked_set = set(row.article_ids_clicked) if row.article_ids_clicked is not None else set()
                for pos, aid in enumerate(inview):
                    label = int(aid in clicked_set)
                    clicks[aid] = clicks.get(aid, 0) + label
                    impr_count[aid] = impr_count.get(aid, 0) + 1
                    pos_clicks[pos] = pos_clicks.get(pos, 0) + label
                    pos_impr[pos] = pos_impr.get(pos, 0) + 1
    popularity_ctr = {aid: clicks[aid] / impr_count[aid] for aid in impr_count}
    position_ctr_prior = {p: pos_clicks[p] / pos_impr[p] for p in pos_impr}
    print(f"  popularity: {len(popularity_ctr)} articles, position prior: {len(position_ctr_prior)} positions")
    return popularity_ctr, position_ctr_prior


class UserQueryCache:
    def __init__(self, article_lookup: dict, id_to_embedding: dict, user_history: dict):
        self.article_lookup = article_lookup
        self.id_to_embedding = id_to_embedding
        self.user_history = user_history
        self._bm25_cache = {}
        self._semantic_cache = {}

    def _recent_ids(self, user_id):
        entry = self.user_history.get(user_id)
        if entry is None:
            return []
        article_ids, _, _ = entry
        return list(article_ids)[-N_RECENT:]

    def get_bm25_query(self, user_id):
        if user_id not in self._bm25_cache:
            self._bm25_cache[user_id] = build_query(self._recent_ids(user_id), self.article_lookup, n_recent=N_RECENT)
        return self._bm25_cache[user_id]

    def get_semantic_vec(self, user_id):
        if user_id not in self._semantic_cache:
            self._semantic_cache[user_id] = build_user_embedding(
                self._recent_ids(user_id), self.id_to_embedding, n_recent=N_RECENT)
        return self._semantic_cache[user_id]


def process_batch(batch: pd.DataFrame, primary_method: str, bm25_index, semantic_index,
                   query_cache: UserQueryCache, category_lookup: dict, published_lookup: dict,
                   popularity_ctr: dict, position_ctr_prior: dict, user_history: dict,
                   booster, feature_col_idx: dict, out_f):
    batch_rows_meta = []
    batch_feature_rows = []

    for row in batch.itertuples(index=False):
        candidates = list(row.article_ids_inview)
        n = len(candidates)
        now_ts = row.impression_time

        query_text = query_cache.get_bm25_query(row.user_id)
        bm25_scores = (np.array(list(bm25_index.score_docs(candidates, query_text).values()))
                       if query_text else np.zeros(n))
        user_vec = query_cache.get_semantic_vec(row.user_id)
        sem_scores = (semantic_index.score_candidates(candidates, user_vec)
                      if user_vec is not None else np.zeros(n))

        primary_scores = bm25_scores if primary_method == "bm25" else sem_scores
        position_order = np.argsort(-primary_scores, kind="stable")
        position_of = {candidates[i]: rank for rank, i in enumerate(position_order)}

        hist_entry = user_history.get(row.user_id)
        if hist_entry is not None:
            hist_ids, hist_times, hist_read_times = hist_entry
            hist_ids = list(hist_ids)[-N_RECENT:]
            hist_times = list(hist_times)[-N_RECENT:]
            hist_read_times = list(hist_read_times)[-N_RECENT:] if hist_read_times is not None else []
        else:
            hist_ids, hist_times, hist_read_times = [], [], []

        n_clicks_before = len(hist_ids)
        is_cold_start = int(n_clicks_before <= COLD_START_MAX_CLICKS)
        if hist_times:
            ages_days = [(now_ts - pd.Timestamp(t)) / np.timedelta64(1, "D") for t in hist_times]
            recency_weighted_click_count = float(sum(0.5 ** (max(a, 0) / RECENCY_HALF_LIFE_DAYS) for a in ages_days))
        else:
            recency_weighted_click_count = 0.0
        avg_dwell_time_before = float(np.mean(hist_read_times)) if len(hist_read_times) else np.nan
        recent_categories = [category_lookup.get(a) for a in hist_ids if a in category_lookup]

        for i, cand in enumerate(candidates):
            cat = category_lookup.get(cand)
            published = published_lookup.get(cand)
            freshness = ((now_ts - pd.Timestamp(published)) / np.timedelta64(1, "h")
                         if published is not None and pd.notna(published) else np.nan)
            if freshness is not None and not (isinstance(freshness, float) and np.isnan(freshness)) and freshness < 0:
                freshness = np.nan
            pos = position_of[cand]

            feat = np.zeros(len(feature_col_idx))
            feat[feature_col_idx["bm25_score"]] = bm25_scores[i]
            feat[feature_col_idx["semantic_score"]] = sem_scores[i]
            feat[feature_col_idx["n_clicks_before"]] = n_clicks_before
            feat[feature_col_idx["recency_weighted_click_count"]] = recency_weighted_click_count
            feat[feature_col_idx["is_cold_start"]] = is_cold_start
            feat[feature_col_idx["popularity_prior_ctr"]] = popularity_ctr.get(cand, 0.0)
            feat[feature_col_idx["freshness_hours"]] = freshness
            feat[feature_col_idx["category_match"]] = int(cat in recent_categories) if recent_categories else 0
            feat[feature_col_idx["category_match_frac"]] = (
                (sum(c == cat for c in recent_categories) / len(recent_categories))
                if recent_categories else np.nan)
            feat[feature_col_idx["session_click_count_before"]] = 0
            feat[feature_col_idx["session_impressions_before"]] = 0
            feat[feature_col_idx["avg_dwell_time_before"]] = avg_dwell_time_before
            feat[feature_col_idx["position"]] = pos
            feat[feature_col_idx["position_ctr_prior"]] = position_ctr_prior.get(pos, np.nan)
            batch_feature_rows.append(feat)

        batch_rows_meta.append((row.impression_id, candidates, n))

    X = np.vstack(batch_feature_rows)
    scores = booster.predict(X)  # ONE batched call for the whole batch, not one per impression

    offset = 0
    for impression_id, candidates, n in batch_rows_meta:
        s = scores[offset:offset + n]
        offset += n
        order = np.argsort(-s, kind="stable")
        ranked_order = [candidates[i] for i in order]
        rank_of = {cid: i + 1 for i, cid in enumerate(ranked_order)}
        ranks_in_original_order = [rank_of[c] for c in candidates]
        rank_str = "[" + ",".join(str(r) for r in ranks_in_original_order) + "]"
        out_f.write(f"{impression_id} {rank_str}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-method", choices=["bm25", "semantic"], default="semantic")
    ap.add_argument("--articles", required=True)
    ap.add_argument("--train-behaviors", required=True)
    ap.add_argument("--val-behaviors", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model", default=None, help="path to the trained reranker (.txt); "
                     "defaults to results/reranker_ebnerd_model.txt")
    ap.add_argument("--out", default="predictions.txt")  # PLURAL, per EB-NeRD's spec
    ap.add_argument("--zip", default="submission_reranked.zip")
    args = ap.parse_args()

    model_path = args.model or str(RESULTS_DIR / "reranker_ebnerd_model.txt")

    print("Building index over EB-NeRD article corpus...")
    articles = build_articles(Path(args.articles))
    article_lookup = dict(zip(articles["article_id"], articles["text"]))
    category_lookup = dict(zip(articles["article_id"], articles["category_str"]))
    published_lookup = dict(zip(articles["article_id"], articles["published_time"]))

    bm25_index = build_bm25(articles)
    semantic_index = build_semantic(articles)
    id_to_embedding = dict(zip(*load_embeddings("ebnerd_large")))

    print("Building popularity + position priors from ebnerd_large train+validation...")
    popularity_ctr, position_ctr_prior = build_popularity_and_position_prior(
        [Path(args.train_behaviors), Path(args.val_behaviors)])

    print(f"Loading reranker from {model_path}...")
    booster = lgb.Booster(model_file=model_path)
    feature_col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    test_dir = Path(args.test_dir)
    user_history = build_user_history_lookup(test_dir / "history.parquet")
    query_cache = UserQueryCache(article_lookup, id_to_embedding, user_history)

    test_behaviors_path = test_dir / "behaviors.parquet"
    pf = pq.ParquetFile(test_behaviors_path)
    total_rows = pf.metadata.num_rows
    print(f"Streaming {test_behaviors_path} in batches of {BATCH_SIZE:,} "
          f"({total_rows:,} impressions, primary_method={args.primary_method})...")

    # write directly into the zip's compressed stream -- never materialize the
    # full plaintext prediction file on disk (at 13.5M lines that's easily a
    # gigabyte or more, and briefly having BOTH the .txt and the .zip on disk
    # at once right at the end is exactly what can tip over a tight disk quota)
    out_path = Path(args.out)
    n_written = 0
    t_start = time.time()
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf, \
            zf.open(out_path.name, "w") as raw, \
            io.TextIOWrapper(raw, encoding="utf-8") as out_f:
        for record_batch in pf.iter_batches(
                batch_size=BATCH_SIZE, columns=["impression_id", "user_id", "article_ids_inview", "impression_time"]):
            batch = record_batch.to_pandas()
            process_batch(batch, args.primary_method, bm25_index, semantic_index, query_cache,
                          category_lookup, published_lookup, popularity_ctr, position_ctr_prior,
                          user_history, booster, feature_col_idx, out_f)
            n_written += len(batch)
            elapsed = time.time() - t_start
            rate = n_written / elapsed
            eta_min = (total_rows - n_written) / rate / 60 if rate > 0 else float("nan")
            print(f"  {n_written:,}/{total_rows:,} processed... ({rate:.0f}/sec, "
                  f"{elapsed/60:.1f} min elapsed, ~{eta_min:.1f} min remaining)")

    print(f"wrote {n_written:,} lines directly into {args.zip} (arcname={out_path.name}) "
          f"in {(time.time()-t_start)/60:.1f} min -- ready to upload")


if __name__ == "__main__":
    main()
