"""Assignment 2, Part I Q2: two-stage retrieve-then-rank pipeline.

Stage 1 (retrieval, reusing Assignment 1's candidate generators): score
each impression's own candidate set with BM25 and semantic similarity --
the same per-impression scoring run_eval_harness.py already does for its
"before" baseline -- using Part I.1's point-in-time recent_article_ids as
the query/user-vector source (not A1's coarser global click history), so
Stage 1 respects the same behavioural-window boundary as Stage 2's
features. Keep the top-K by the chosen PRIMARY method's score (K~100-200
per spec; a no-op for most impressions here since MIND/EB-NeRD's official
candidate lists are usually much smaller, but real for the long tail --
some MIND impressions carry 400+ candidates).

Stage 2 (re-rank): a LightGBM LambdaMART ranker trained on Part I.1's
engineered behavioural features (data/features/behavioral_features.parquet)
plus the two retrieval scores as extra features, one group per impression.
"""
import numpy as np
import pandas as pd

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import load_embeddings, embeddings_exist, compute_embeddings, save_embeddings
from src.query_builder import build_query, build_user_embedding

FEATURE_COLS = [
    "bm25_score", "semantic_score",
    "n_clicks_before", "recency_weighted_click_count", "is_cold_start",
    "popularity_prior_ctr", "freshness_hours",
    "category_match", "category_match_frac",
    "session_click_count_before", "session_impressions_before", "avg_dwell_time_before",
    "position_ctr_prior", "position",
]


def build_bm25_index(articles: pd.DataFrame, dataset: str):
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    index = BM25Index().fit(sub["article_id"].tolist(), sub["text"].tolist())
    return index, dict(zip(sub["article_id"], sub["text"]))


def build_semantic_index(articles: pd.DataFrame, dataset: str):
    if embeddings_exist(dataset):
        article_ids, embeddings = load_embeddings(dataset)
    else:
        sub = articles[articles["dataset"] == dataset].copy()
        sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
        article_ids, embeddings = compute_embeddings(sub["article_id"].tolist(), sub["text"].tolist())
        save_embeddings(dataset, article_ids, embeddings)
    index = ANNIndex().fit(article_ids, embeddings)
    return index, dict(zip(article_ids, embeddings))


def add_retrieval_scores(candidates_df: pd.DataFrame, click_hist: pd.DataFrame,
                          bm25_index: BM25Index, bm25_lookup: dict,
                          semantic_index: ANNIndex, id_to_embedding: dict) -> pd.DataFrame:
    """candidates_df: one row per (impression_id, article_id) candidate
    (a slice of behavioral_features.parquet). Adds bm25_score/semantic_score,
    computed once per impression (all its candidates share one query) via
    score_docs() -- vectorized over that impression's candidate list rather
    than one dot product per row.
    """
    hist_lookup = click_hist.set_index("impression_id")["recent_article_ids"]

    out = candidates_df.copy()
    out["bm25_score"] = 0.0
    out["semantic_score"] = 0.0

    for iid, grp in out.groupby("impression_id", observed=True):
        recent_ids = hist_lookup.get(iid, [])
        candidates = grp["article_id"].tolist()

        query = build_query(recent_ids, bm25_lookup)
        bm25_scores = bm25_index.score_docs(candidates, query) if query else {c: 0.0 for c in candidates}

        user_vec = build_user_embedding(recent_ids, id_to_embedding)
        sem_scores = (semantic_index.score_docs(candidates, user_vec) if user_vec is not None
                      else {c: 0.0 for c in candidates})

        out.loc[grp.index, "bm25_score"] = [bm25_scores.get(c, 0.0) for c in candidates]
        out.loc[grp.index, "semantic_score"] = [sem_scores.get(c, 0.0) for c in candidates]
    return out


def cap_top_k(scored_df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Stage 1's 'retrieve top-K': keep each impression's K highest-scoring
    candidates by score_col. A no-op for impressions with <= K candidates
    already (the common case here); for larger ones, an impression whose
    true click falls outside the top-K becomes an honest retrieval miss --
    evaluate_scores() skips impressions with no positive left, exactly like
    run_eval_harness.py already does for degenerate cases.
    """
    ranked = scored_df.sort_values(["impression_id", score_col], ascending=[True, False])
    rank_in_impression = ranked.groupby("impression_id", observed=True).cumcount()
    return ranked[rank_in_impression < k]


def _prepare_X(df: pd.DataFrame, feature_cols=FEATURE_COLS) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in ("is_cold_start", "category_match"):
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X  # LightGBM handles remaining NaNs (e.g. MIND's freshness_hours) natively


def train_ranker(train_df: pd.DataFrame, feature_cols=FEATURE_COLS, **lgbm_kwargs):
    import lightgbm as lgb
    train_df = train_df.sort_values("impression_id")  # LightGBM needs each group's rows contiguous
    groups = train_df.groupby("impression_id", observed=True, sort=False).size().to_numpy()
    X = _prepare_X(train_df, feature_cols)
    y = train_df["clicked"].to_numpy()

    params = dict(objective="lambdarank", metric="ndcg", n_estimators=200,
                  learning_rate=0.05, num_leaves=31, min_child_samples=20, verbosity=-1)
    params.update(lgbm_kwargs)
    ranker = lgb.LGBMRanker(**params)
    ranker.fit(X, y, group=groups)
    return ranker


def score_ranker(ranker, df: pd.DataFrame, feature_cols=FEATURE_COLS) -> np.ndarray:
    return ranker.predict(_prepare_X(df, feature_cols))


def evaluate_scores(df: pd.DataFrame, score_col: str, label_col: str = "clicked") -> pd.DataFrame:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 for one score column -- mirrors
    run_eval_harness.py's per-impression loop. Impressions with <2
    candidates or no positive label are skipped (metric undefined)."""
    from src.ranking_metrics import auc, mrr, ndcg_at_k
    rows = []
    for iid, grp in df.groupby("impression_id", observed=True):
        scores = grp[score_col].to_numpy()
        labels = grp[label_col].to_numpy()
        if len(scores) < 2 or labels.sum() == 0:
            continue
        rows.append({
            "impression_id": iid,
            "auc": auc(scores, labels), "mrr": mrr(scores, labels),
            "ndcg5": ndcg_at_k(scores, labels, 5), "ndcg10": ndcg_at_k(scores, labels, 10),
        })
    return pd.DataFrame(rows)
