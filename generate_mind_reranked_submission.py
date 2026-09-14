#!/usr/bin/env python
"""Assignment 2, Q5: Codabench submission using the FULL two-stage pipeline
(Q2's BM25/semantic retrieval + LightGBM re-rank), not just bare retrieval.

Builds on generate_mind_submission.py's proven approach -- same news-corpus
loading, same chunked streaming of MINDlarge_test/behaviors.tsv, same output
format ("{impression_id} [rank_of_candidate_1,rank_of_candidate_2,...]" in
the candidates' ORIGINAL order). See that file for the retrieval-only
baseline this extends; this one adds a feature-assembly + rerank step per
chunk on top of the same retrieval scoring.

CAVEATS (documented, not hidden -- see conversation for the full reasoning):
  - The reranker (results/reranker_mind_model.txt) was trained on
    MIND-SMALL-derived features (Assignment 2 Q1/Q2). Applying it to
    MIND-large's differently-scaled feature distributions (e.g. popularity
    counts over a much bigger corpus) is a genuine train/serving shift --
    the feature *semantics* match, the *scale* doesn't exactly.
  - MIND's raw behaviors.tsv history column has NO per-item timestamps,
    only article IDs -- true point-in-time recency-weighting (as Q1
    computed from full timestamped click logs) isn't reconstructable here.
    recency_weighted_click_count is approximated as click COUNT instead.
  - No session data exists in this format -- session features use the same
    neutral defaults (0 / NaN) as Q4's serving benchmark.
  - `position` is the candidate's rank under the PRIMARY retrieval method
    (bm25 or semantic) BEFORE re-ranking -- the same legitimate,
    available-before-ranking interpretation Q2 trained on and Q4's
    benchmark used (see either file's docstring for why this avoids
    leakage from the historical dataset's own collected position).
  - popularity_ctr and position_ctr_prior are computed from train+dev's
    LABELED behaviors (MINDlarge ships no train/val/test split of its own
    the way our Q1 pipeline built one, so this is the large-scale
    equivalent, built once as a preprocessing pass).

Usage:
    python generate_mind_reranked_submission.py --primary-method semantic \
        --train-dir data/raw/mind_large/train \
        --dev-dir data/raw/mind_large/dev \
        --test-dir data/raw/mind_large/test
"""
import argparse
import io
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import compute_embeddings, save_embeddings, load_embeddings, embeddings_exist
from src.query_builder import build_query, build_user_embedding
from src.reranker import FEATURE_COLS
from src.config import COLD_START_MAX_CLICKS, ROOT

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
N_RECENT = 10  # matches Q1's N_RECENT_CLICKS
CHUNK_SIZE = 10_000
RESULTS_DIR = ROOT / "results"


def load_news(directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / "news.tsv", sep="\t", header=None, names=NEWS_COLS)


def build_corpus(train_dir: Path, dev_dir: Path, test_dir: Path) -> pd.DataFrame:
    print("  loading news.tsv from train/dev/test and deduplicating...")
    news = pd.concat([load_news(train_dir), load_news(dev_dir), load_news(test_dir)],
                      ignore_index=True).drop_duplicates("news_id")
    news["text"] = (news["title"].fillna("") + " " + news["abstract"].fillna("")).str.strip()
    print(f"  full corpus: {len(news)} unique articles")
    return news


def build_bm25(news: pd.DataFrame) -> BM25Index:
    index = BM25Index()
    index.fit(news["news_id"].tolist(), news["text"].tolist())
    return index


def build_semantic(news: pd.DataFrame) -> ANNIndex:
    if embeddings_exist("mind_large"):
        article_ids, embeddings = load_embeddings("mind_large")
    else:
        print("  computing embeddings for full large corpus (one-time, cached after)...")
        article_ids, embeddings = compute_embeddings(news["news_id"].tolist(), news["text"].tolist())
        save_embeddings("mind_large", article_ids, embeddings)
    return ANNIndex().fit(article_ids, embeddings)


def build_popularity_and_position_prior(train_dir: Path, dev_dir: Path):
    """One vectorized pass over train+dev's LABELED behaviors.tsv ('-0/1'
    suffixes, unlike test) for a global article popularity CTR and a
    position->CTR prior -- the large-scale equivalent of Q1's train-only
    popularity_prior_ctr / position_ctr_prior (MINDlarge ships no train/
    val/test split of its own the way our Q1 pipeline built one)."""
    clicks, impr_count, pos_clicks, pos_impr = {}, {}, {}, {}
    for beh_path in (train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv"):
        print(f"  scanning {beh_path} for popularity + position priors...")
        for chunk in pd.read_csv(beh_path, sep="\t", header=None, names=BEH_COLS,
                                  usecols=[4], chunksize=CHUNK_SIZE):
            tokens = chunk["impressions"].str.split().explode()
            # a handful of rows in MINDlarge's raw file have a missing/blank
            # impressions field -- .str.split() turns those into NaN, which
            # would otherwise poison the later int() cast; drop them before
            # computing position so valid rows' positions stay unaffected
            # (a missing field is all-or-nothing per row, never partial)
            tokens = tokens[tokens.notna()]
            if tokens.empty:
                continue
            pos = tokens.groupby(level=0).cumcount()
            nid_label = tokens.str.rsplit("-", n=1, expand=True)
            valid = nid_label[1].notna()  # guard any token still missing its "-label" suffix
            nid, label, pos = nid_label[0][valid], nid_label[1][valid].astype(int), pos[valid]

            g = label.groupby(nid)
            for k, v in g.sum().items():
                clicks[k] = clicks.get(k, 0) + v
            for k, v in g.count().items():
                impr_count[k] = impr_count.get(k, 0) + v

            gp = label.groupby(pos)
            for k, v in gp.sum().items():
                pos_clicks[k] = pos_clicks.get(k, 0) + v
            for k, v in gp.count().items():
                pos_impr[k] = pos_impr.get(k, 0) + v

    popularity_ctr = {nid: clicks[nid] / impr_count[nid] for nid in impr_count}
    position_ctr_prior = {p: pos_clicks[p] / pos_impr[p] for p in pos_impr}
    print(f"  popularity: {len(popularity_ctr)} articles, position prior: {len(position_ctr_prior)} positions")
    return popularity_ctr, position_ctr_prior


class UserQueryCache:
    """Caches BOTH the BM25 text query and the semantic mean-pooled
    embedding per user, built once, reused across all their impressions."""

    def __init__(self, article_lookup: dict, id_to_embedding: dict):
        self.article_lookup = article_lookup
        self.id_to_embedding = id_to_embedding
        self._bm25_cache = {}
        self._semantic_cache = {}

    def get_bm25_query(self, user_id, history):
        if user_id not in self._bm25_cache:
            self._bm25_cache[user_id] = build_query(history, self.article_lookup, n_recent=N_RECENT)
        return self._bm25_cache[user_id]

    def get_semantic_vec(self, user_id, history):
        if user_id not in self._semantic_cache:
            self._semantic_cache[user_id] = build_user_embedding(history, self.id_to_embedding, n_recent=N_RECENT)
        return self._semantic_cache[user_id]


def process_chunk(chunk: pd.DataFrame, primary_method: str, bm25_index, semantic_index,
                   query_cache: UserQueryCache, category_lookup: dict, popularity_ctr: dict,
                   position_ctr_prior: dict, booster, feature_col_idx: dict, out_f):
    chunk_rows_meta = []   # (impression_id, candidates, n_candidates) per row
    chunk_feature_rows = []  # flat list of feature vectors across the whole chunk

    for row in chunk.itertuples(index=False):
        candidates = str(row.impressions).split()
        history = str(row.history).split()[-N_RECENT:] if pd.notna(row.history) and row.history else []
        n = len(candidates)

        bm25_scores_dict = {}
        query_text = query_cache.get_bm25_query(row.user_id, history)
        if query_text:
            bm25_scores_dict = bm25_index.score_docs(candidates, query_text)
        bm25_scores = np.array([bm25_scores_dict.get(c, 0.0) for c in candidates])

        user_vec = query_cache.get_semantic_vec(row.user_id, history)
        sem_scores = (semantic_index.score_candidates(candidates, user_vec)
                      if user_vec is not None else np.zeros(n))

        primary_scores = bm25_scores if primary_method == "bm25" else sem_scores
        position_order = np.argsort(-primary_scores, kind="stable")
        position_of = {candidates[i]: rank for rank, i in enumerate(position_order)}

        n_clicks_before = len(history)
        is_cold_start = int(n_clicks_before <= COLD_START_MAX_CLICKS)
        recency_weighted_click_count = float(n_clicks_before)  # see module docstring: no per-item timestamps
        recent_categories = [category_lookup.get(a) for a in history if a in category_lookup]

        for i, cand in enumerate(candidates):
            cat = category_lookup.get(cand)
            pos = position_of[cand]
            feat = np.zeros(len(feature_col_idx))
            feat[feature_col_idx["bm25_score"]] = bm25_scores[i]
            feat[feature_col_idx["semantic_score"]] = sem_scores[i]
            feat[feature_col_idx["n_clicks_before"]] = n_clicks_before
            feat[feature_col_idx["recency_weighted_click_count"]] = recency_weighted_click_count
            feat[feature_col_idx["is_cold_start"]] = is_cold_start
            feat[feature_col_idx["popularity_prior_ctr"]] = popularity_ctr.get(cand, 0.0)
            feat[feature_col_idx["freshness_hours"]] = np.nan  # MIND ships no published_time at all
            feat[feature_col_idx["category_match"]] = int(cat in recent_categories) if recent_categories else 0
            feat[feature_col_idx["category_match_frac"]] = (
                (sum(c == cat for c in recent_categories) / len(recent_categories))
                if recent_categories else np.nan)
            feat[feature_col_idx["session_click_count_before"]] = 0
            feat[feature_col_idx["session_impressions_before"]] = 0
            feat[feature_col_idx["avg_dwell_time_before"]] = np.nan
            feat[feature_col_idx["position"]] = pos
            feat[feature_col_idx["position_ctr_prior"]] = position_ctr_prior.get(pos, np.nan)
            chunk_feature_rows.append(feat)

        chunk_rows_meta.append((row.impression_id, candidates, n))

    X = np.vstack(chunk_feature_rows)
    scores = booster.predict(X)  # ONE batched call for the whole chunk, not one per impression

    offset = 0
    for impression_id, candidates, n in chunk_rows_meta:
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
    ap.add_argument("--primary-method", choices=["bm25", "semantic"], default="semantic",
                     help="Stage-1 candidate scoring method (also defines `position`)")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model", default=None, help="path to the trained reranker (.txt); "
                     "defaults to results/reranker_mind_model.txt")
    ap.add_argument("--out", default="prediction.txt")
    ap.add_argument("--zip", default="submission_reranked.zip")
    args = ap.parse_args()

    train_dir, dev_dir, test_dir = Path(args.train_dir), Path(args.dev_dir), Path(args.test_dir)
    model_path = args.model or str(RESULTS_DIR / "reranker_mind_model.txt")

    news = build_corpus(train_dir, dev_dir, test_dir)
    article_lookup = dict(zip(news["news_id"], news["text"]))
    category_lookup = dict(zip(news["news_id"], news["category"]))

    print("Building BM25 + semantic indexes...")
    bm25_index = build_bm25(news)
    semantic_index = build_semantic(news)
    id_to_embedding = dict(zip(*load_embeddings("mind_large")))

    print("Building popularity + position priors from train+dev...")
    popularity_ctr, position_ctr_prior = build_popularity_and_position_prior(train_dir, dev_dir)

    print(f"Loading reranker from {model_path}...")
    booster = lgb.Booster(model_file=model_path)
    feature_col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    query_cache = UserQueryCache(article_lookup, id_to_embedding)

    test_behaviors_path = test_dir / "behaviors.tsv"
    print(f"Streaming {test_behaviors_path} in chunks of {CHUNK_SIZE:,} (primary_method={args.primary_method})...")

    # write directly into the zip's compressed stream -- never materialize the
    # full plaintext prediction file on disk (at 2.37M lines that's easily a
    # few hundred MB, and briefly having BOTH the .txt and the .zip on disk
    # at once right at the end is exactly what can tip over a tight disk quota)
    out_path = Path(args.out)
    n_written = 0
    t_start = time.time()
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf, \
            zf.open(out_path.name, "w") as raw, \
            io.TextIOWrapper(raw, encoding="utf-8") as out_f:
        for chunk in pd.read_csv(test_behaviors_path, sep="\t", header=None,
                                  names=BEH_COLS, chunksize=CHUNK_SIZE):
            process_chunk(chunk, args.primary_method, bm25_index, semantic_index, query_cache,
                          category_lookup, popularity_ctr, position_ctr_prior, booster, feature_col_idx, out_f)
            n_written += len(chunk)
            elapsed = time.time() - t_start
            rate = n_written / elapsed
            print(f"  {n_written:,} impressions processed... ({rate:.0f}/sec, {elapsed/60:.1f} min elapsed)")

    print(f"wrote {n_written:,} lines directly into {args.zip} (arcname={out_path.name}) "
          f"in {(time.time()-t_start)/60:.1f} min -- ready to upload")


if __name__ == "__main__":
    main()
