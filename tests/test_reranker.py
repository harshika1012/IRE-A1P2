"""Q2: unit tests for the retrieve-then-rank pipeline's non-IO pieces
(cap_top_k, evaluate_scores, and the LightGBM train/score wiring). Uses
small synthetic tables -- no downloaded dataset or precomputed features
needed.

Run with: pytest tests/test_reranker.py -q
"""
import numpy as np
import pandas as pd

from src.reranker import cap_top_k, evaluate_scores, train_ranker, score_ranker, FEATURE_COLS


def test_cap_top_k_keeps_only_highest_scoring_candidates_per_impression():
    df = pd.DataFrame({
        "impression_id": ["i1"] * 4 + ["i2"] * 2,
        "article_id": ["a", "b", "c", "d", "e", "f"],
        "score": [0.9, 0.1, 0.5, 0.3, 0.2, 0.8],
    })
    capped = cap_top_k(df, "score", k=2)
    kept = capped.groupby("impression_id")["article_id"].apply(set).to_dict()
    assert kept["i1"] == {"a", "c"}  # top-2 of i1 by score: 0.9, 0.5
    assert kept["i2"] == {"f", "e"}  # i2 only has 2 candidates -> both kept
    assert (capped.groupby("impression_id").size() <= 2).all()


def test_evaluate_scores_skips_degenerate_impressions():
    df = pd.DataFrame({
        "impression_id": ["i1", "i1", "i2", "i2", "i3"],
        "score": [0.9, 0.1, 0.5, 0.5, 0.7],
        "clicked": [1, 0, 0, 0, 1],  # i2: no positive; i3: single candidate
    })
    out = evaluate_scores(df, "score")
    assert set(out["impression_id"]) == {"i1"}
    assert out.iloc[0]["auc"] == 1.0  # the only positive (0.9) outranks the only negative (0.1)


def test_train_ranker_recovers_a_strongly_predictive_feature():
    # a synthetic feature that's simply equal to the label should let the
    # ranker perfectly separate positives from negatives after training --
    # a sanity check that group boundaries / LightGBM wiring are correct,
    # not a performance benchmark.
    rng = np.random.default_rng(0)
    rows = []
    for iid in range(200):
        n_cand = rng.integers(3, 8)
        pos = rng.integers(0, n_cand)
        for c in range(n_cand):
            clicked = int(c == pos)
            rows.append({"impression_id": f"i{iid}", "clicked": clicked,
                         "bm25_score": clicked + rng.normal(0, 0.01),
                         "semantic_score": 0.0, "n_clicks_before": 0,
                         "recency_weighted_click_count": 0.0, "is_cold_start": True,
                         "popularity_prior_ctr": 0.0, "freshness_hours": np.nan,
                         "category_match": False, "category_match_frac": np.nan,
                         "session_click_count_before": 0, "session_impressions_before": 0,
                         "avg_dwell_time_before": np.nan, "position_ctr_prior": 0.0, "position": c})
    df = pd.DataFrame(rows)
    ranker = train_ranker(df, n_estimators=20)
    scores = score_ranker(ranker, df)
    df = df.assign(pred=scores)
    hit_rate = df.loc[df.groupby("impression_id")["pred"].idxmax(), "clicked"].mean()
    assert hit_rate > 0.9  # top-predicted candidate is the true click in almost every impression
