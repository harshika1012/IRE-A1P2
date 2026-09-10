"""Assignment 2, Q2 -- Stage 1 (retrieve): wraps Assignment 1's tuned hybrid
BM25 + semantic retriever as the "candidate generator" the spec asks for.

Important: this does NOT search the full article corpus with
BM25Index.top_k()/ANNIndex.top_k(). MIND and EB-NeRD both hand you a fixed
candidate list per impression (behaviors.tsv's Impressions field / EB-NeRD's
article_ids_inview) with ground-truth labels ONLY for those items -- there's
no point generating extra candidates you have no label for. So "retrieve
top-K" here means: score & rank the impression's OWN candidates with the
tuned hybrid blend (exactly what tune_alpha.py already does per-impression),
then truncate to K. In practice K~100-200 rarely binds since most impressions
have well under 100 candidates on both datasets -- worth a one-line note in
your report rather than something to "fix".

Output of this stage feeds INTO stage 2 (the GBDT/neural re-ranker) as
features -- `retrieval_score` and `retrieval_rank` are not the final ranking,
just two more numbers the re-ranker learns to combine with the Q1 click-
history/session/article features.
"""
import numpy as np
import pandas as pd

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import load_embeddings, embeddings_exist, compute_embeddings, save_embeddings
from src.query_builder import build_query, build_user_embedding


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


def _bm25_score_array(index: BM25Index, doc_ids, query_text: str) -> np.ndarray:
    """Positionally-aligned BM25 scores for doc_ids, duplicates and order
    preserved -- the array-returning counterpart to ANNIndex.score_candidates.

    NOT the same as BM25Index.score_docs(), which returns a dict keyed by
    doc_id: if `doc_ids` contains a repeated article_id (an impression can
    legitimately list the same candidate twice), the dict silently collapses
    it to one entry, so len(score_docs(...).values()) can come back SHORTER
    than len(doc_ids) -- that's what caused the earlier
    "operands could not be broadcast together" crash when it collided with
    ANNIndex.score_candidates' duplicate-preserving array of the true length.
    """
    n = len(doc_ids)
    if not query_text:
        return np.zeros(n)
    q = index.vectorize_query(query_text)
    if q is None:
        return np.zeros(n)
    rows = np.array([index._id_to_row.get(d, -1) for d in doc_ids])
    valid = rows >= 0
    scores = np.zeros(n, dtype=np.float64)
    if valid.any():
        sub = index.W[rows[valid]] @ q.T
        scores[valid] = np.asarray(sub.todense()).ravel()
    return scores


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)


def add_retrieval_features(impressions: pd.DataFrame, articles: pd.DataFrame,
                            dataset: str, alpha: float, top_k: int = 150) -> pd.DataFrame:
    """Adds, per candidate row:
        retrieval_score - blended hybrid score: alpha*bm25_norm + (1-alpha)*sem_norm,
                           normalized independently PER IMPRESSION (matches how
                           tune_alpha.py evaluated alpha, so the chosen alpha
                           is meaningful here)
        retrieval_rank  - 1-indexed rank of this candidate within its impression
                           by retrieval_score (1 = top-retrieved)
        in_top_k        - True if retrieval_rank <= top_k (kept for transparency;
                           since most impressions have far fewer than `top_k`
                           candidates, this is usually all True -- see module
                           docstring)

    `alpha` should come straight out of tune_alpha.py's printed
    "Best alpha by AUC" for this dataset -- do NOT re-tune inside this
    function, or you'd be picking alpha using val-split signal that then also
    trains the stage-2 re-ranker, which is a second, subtler form of leakage.

    IMPORTANT: `impressions` must already have the `recent_article_ids` column
    from click_features.add_recent_article_ids(). We build the BM25 query /
    user embedding from THAT point-in-time history, not from
    user_features.parquet's static click_history -- the latter is frozen at
    the train cutoff (Assignment 1's snapshot), so using it here would make
    every val/test impression's retrieval query stale by however long ago
    the train cutoff was, even though nothing here technically leaks a
    future click. Using recent_article_ids keeps retrieval features on the
    same point-in-time footing as every other Q1 feature.

    Only uses each impression's OWN candidate list (see module docstring) --
    this only re-ranks existing candidates, it never invents new ones.
    """
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    sem_index, sem_lookup = build_semantic_index(articles, dataset)

    sub = impressions[impressions["dataset"] == dataset]
    out_score = pd.Series(0.0, index=sub.index)
    out_rank = pd.Series(np.nan, index=sub.index)

    for iid, grp in sub.groupby("impression_id"):
        candidates = grp["article_id"].tolist()
        # every row of this impression shares the same point-in-time history
        # (it's the same user, same impression timestamp) -- just read it
        # off the first row rather than recomputing per candidate.
        history = grp["recent_article_ids"].iloc[0]

        bm25_query = build_query(history, bm25_lookup)
        sem_vec = build_user_embedding(history, sem_lookup)

        bm25_scores = _bm25_score_array(bm25_index, candidates, bm25_query)
        sem_scores = (sem_index.score_candidates(candidates, sem_vec)
                      if sem_vec is not None else np.zeros(len(candidates)))

        blended = alpha * _minmax(bm25_scores) + (1 - alpha) * _minmax(sem_scores)
        order = np.argsort(-blended)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        out_score.loc[grp.index] = blended
        out_rank.loc[grp.index] = ranks

    result = impressions.copy()
    result.loc[sub.index, "retrieval_score"] = out_score
    result.loc[sub.index, "retrieval_rank"] = out_rank
    result["in_top_k"] = result["retrieval_rank"] <= top_k
    return result