"""Assignment 2, Q2, step 1 -- STRICT reading: "Use Assignment 1's candidate
generator to retrieve top-K candidates (K ~ 100-200)" means actually
searching the full article corpus, not just re-ranking each impression's
already-given candidate list (that lighter-weight alternative lives in
candidate_retrieval.add_retrieval_features and is documented there as a
deviation -- this module is the literal version).

Labeling a candidate that didn't come from MIND/EB-NeRD's own given list:
we label it 0 (not clicked). This is factually correct, not fabricated --
"clicked" is defined per-impression, and an article outside the platform's
own shown set was, by definition, not clicked in this specific impression
event. The real consequence is retrieval recall loss: if the true clicked
article isn't found in this corpus-wide top-K, no downstream re-ranker can
ever recover it for that impression. That's an expected, measurable property
of a genuine retrieve-then-rank system (and feeds Q4's scale story), not a
bug -- see recall_at_k() below to quantify it.

Output shape matches the original `impressions` dataframe exactly
(impression_id, user_id, dataset, timestamp, article_id, clicked, plus
retrieval_score/retrieval_rank standing in for `position`), so it can be fed
straight through the EXISTING Q1 pipeline
(click_features.add_click_history_features / add_recent_article_ids /
add_session_features / add_article_dynamic_features /
add_category_match_features) with zero changes to that code.

Compute cost: a brute-force top-K search of ~65-80K articles PER IMPRESSION,
across potentially millions of impressions, is genuinely expensive -- this is
the real thing Q4 asks you to characterize later. `max_impressions` lets you
run this over a bounded, reproducible sample for development and for a
report-ready result, without requiring a full-dataset run on a laptop.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.candidate_retrieval import build_bm25_index, build_semantic_index, _minmax
from src.query_builder import build_query, build_user_embedding


def build_full_corpus_candidates(impressions: pd.DataFrame, articles: pd.DataFrame,
                                  dataset: str, alpha: float, k: int = 150,
                                  max_impressions: int = None, seed: int = 42) -> pd.DataFrame:
    """Returns a new candidate-level dataframe, one row per (impression,
    retrieved article) pair, with columns:
        impression_id, user_id, dataset, timestamp, article_id, clicked,
        position (== retrieval_rank, aliased so it flows into
        click_features.add_session_features unchanged),
        retrieval_score, retrieval_rank, found_in_original_candidates
    (`found_in_original_candidates` is diagnostic only -- True if this
    retrieved article was actually one of MIND/EB-NeRD's own shown
    candidates for this impression; useful for sanity-checking retrieval
    quality, not used by the downstream re-ranker.)

    Requires `impressions` to already have `recent_article_ids`
    (click_features.add_recent_article_ids) so the query reflects
    point-in-time history, same as candidate_retrieval.py's lighter version.
    """
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    sem_index, sem_lookup = build_semantic_index(articles, dataset)

    sub = impressions[impressions["dataset"] == dataset]
    grouped = sub.groupby("impression_id")
    impression_ids = list(grouped.groups.keys())
    if max_impressions and len(impression_ids) > max_impressions:
        rng = np.random.default_rng(seed)
        impression_ids = list(rng.choice(impression_ids, size=max_impressions, replace=False))

    records = []
    n_no_query, n_recall_hit, n_scored = 0, 0, 0
    for iid in tqdm(impression_ids, desc=f"full-corpus retrieval[{dataset}]"):
        grp = grouped.get_group(iid)
        uid = grp["user_id"].iloc[0]
        ts = grp["timestamp"].iloc[0]
        history = grp["recent_article_ids"].iloc[0]
        true_labels = dict(zip(grp["article_id"], grp["clicked"]))
        true_clicked = {a for a, c in true_labels.items() if c == 1}

        bm25_query = build_query(history, bm25_lookup)
        sem_vec = build_user_embedding(history, sem_lookup)
        if not bm25_query and sem_vec is None:
            n_no_query += 1
            continue

        bm25_hits = dict(bm25_index.top_k(bm25_query, k=k)) if bm25_query else {}
        sem_hits = dict(sem_index.top_k(sem_vec, k=k)) if sem_vec is not None else {}
        all_ids = list(set(bm25_hits) | set(sem_hits))
        if not all_ids:
            continue

        bm25_arr = np.array([bm25_hits.get(a, 0.0) for a in all_ids])
        sem_arr = np.array([sem_hits.get(a, 0.0) for a in all_ids])
        blended = alpha * _minmax(bm25_arr) + (1 - alpha) * _minmax(sem_arr)

        order = np.argsort(-blended)[:k]
        n_scored += 1
        if true_clicked & {all_ids[i] for i in order}:
            n_recall_hit += 1

        for rank, idx in enumerate(order, start=1):
            aid = all_ids[idx]
            records.append({
                "impression_id": iid, "user_id": uid, "dataset": dataset, "timestamp": ts,
                "article_id": aid, "clicked": int(true_labels.get(aid, 0)),
                "position": rank, "retrieval_score": float(blended[idx]), "retrieval_rank": rank,
                "found_in_original_candidates": aid in true_labels,
            })

    print(f"  {dataset}: {n_scored} impressions scored, {n_no_query} skipped (no query), "
          f"recall@{k} = {n_recall_hit / n_scored:.4f}" if n_scored else "  no impressions scored")
    return pd.DataFrame(records)