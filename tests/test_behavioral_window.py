"""Q1 / Q9 anti-gaming requirement: 'no future-click leakage' must be
provable, not just asserted. These tests build tiny synthetic click-logs
by hand (no downloaded dataset required) and check that every behavioural
feature only ever reflects events strictly before its own impression.

Run with: pytest tests/test_behavioral_window.py -q
"""
import numpy as np
import pandas as pd
import pytest

from src.behavioral_features import (
    compute_click_history_features, compute_article_dynamic_features,
    compute_session_features, attach_category_match, attach_click_history_embeddings,
    assert_no_leakage,
)

T = pd.Timestamp


def _articles():
    return pd.DataFrame([
        {"dataset": "toy", "article_id": "a1", "category": "sports", "title": "A1",
         "published_time": T("2024-01-01")},
        {"dataset": "toy", "article_id": "a2", "category": "sports", "title": "A2",
         "published_time": T("2024-01-02")},
        {"dataset": "toy", "article_id": "a3", "category": "news", "title": "A3",
         "published_time": T("2024-01-10")},
    ])


def test_click_history_excludes_same_and_future_clicks():
    history = pd.DataFrame([
        {"dataset": "toy", "user_id": "u1", "article_id": "a1", "timestamp": T("2024-01-03 10:00")},
        {"dataset": "toy", "user_id": "u1", "article_id": "a2", "timestamp": T("2024-01-04 10:00")},
    ])
    impressions = pd.DataFrame([
        # impression sits strictly between the two clicks
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1",
         "timestamp": T("2024-01-03 12:00"), "article_id": "a3", "clicked": 0, "position": 0},
    ])
    feats = compute_click_history_features(impressions, history, _articles())
    row = feats.iloc[0]
    assert row["n_clicks_before"] == 1
    assert row["recent_titles"] == ["A1"]
    assert "A2" not in row["recent_titles"]  # a2's click happens AFTER this impression
    assert row["days_since_last_click"] == pytest.approx(2 / 24)  # ~2 hours


def test_click_history_cold_start_user():
    history = pd.DataFrame([
        {"dataset": "toy", "user_id": "u2", "article_id": "a1", "timestamp": T("2024-01-05")},
    ])
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u_never_clicked",
         "timestamp": T("2024-01-06"), "article_id": "a3", "clicked": 0, "position": 0},
    ])
    feats = compute_click_history_features(impressions, history, _articles())
    row = feats.iloc[0]
    assert row["n_clicks_before"] == 0
    assert row["recent_titles"] == []
    assert row["is_cold_start"]
    assert pd.isna(row["days_since_last_click"])
    assert row["recency_weighted_click_count"] == 0.0


def test_article_popularity_excludes_same_timestamp_and_future():
    # two candidates shown at the SAME instant must not see each other's
    # click, and a click that happens later must not count as "prior"
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1", "article_id": "a1",
         "timestamp": T("2024-01-05 10:00"), "clicked": 1, "position": 0, "split": "train"},
        {"impression_id": "i2", "dataset": "toy", "user_id": "u2", "article_id": "a1",
         "timestamp": T("2024-01-05 10:00"), "clicked": 0, "position": 0, "split": "train"},  # same instant as i1
        {"impression_id": "i3", "dataset": "toy", "user_id": "u3", "article_id": "a1",
         "timestamp": T("2024-01-06 10:00"), "clicked": 0, "position": 0, "split": "val"},  # strictly after
    ])
    out = compute_article_dynamic_features(impressions, _articles())

    same_instant = out[out["timestamp"] == T("2024-01-05 10:00")]
    assert (same_instant["cum_clicks_prior"] == 0).all(), \
        "same-timestamp candidates must not see each other's click"

    later = out[out["impression_id"] == "i3"].iloc[0]
    assert later["cum_clicks_prior"] == 1     # the earlier bucket's 1 click IS visible now
    assert later["cum_impressions_prior"] == 2


def test_freshness_never_negative():
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1", "article_id": "a3",
         "timestamp": T("2024-01-01"), "clicked": 0, "position": 0, "split": "val"},  # BEFORE a3 was published
    ])
    out = compute_article_dynamic_features(impressions, _articles())
    assert pd.isna(out["freshness_hours"].iloc[0])  # clock-skew case -> NaN, never negative


def test_session_click_count_excludes_current_and_future():
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1", "article_id": "a1",
         "timestamp": T("2024-01-01 09:00"), "clicked": 1, "position": 0,
         "session_id": None, "read_time": None},
        {"impression_id": "i2", "dataset": "toy", "user_id": "u1", "article_id": "a2",
         "timestamp": T("2024-01-01 09:05"), "clicked": 0, "position": 0,
         "session_id": None, "read_time": None},
        {"impression_id": "i3", "dataset": "toy", "user_id": "u1", "article_id": "a3",
         "timestamp": T("2024-01-01 09:10"), "clicked": 1, "position": 0,
         "session_id": None, "read_time": None},
    ])
    out = compute_session_features(impressions).set_index("impression_id")
    # i1, i2, i3 are within the 30-min gap window -> one derived session
    assert out.loc["i1", "session_click_count_before"] == 0
    assert out.loc["i2", "session_click_count_before"] == 1   # i1 clicked, before i2
    assert out.loc["i3", "session_click_count_before"] == 1   # i1 clicked; i2's own label excluded


def test_click_history_embedding_uses_only_point_in_time_recent_clicks(monkeypatch):
    # user clicked a1 (before the impression) and a2 (after) -- the embedding
    # must be pooled from a1 alone, exactly like recent_article_ids already is
    history = pd.DataFrame([
        {"dataset": "toy", "user_id": "u1", "article_id": "a1", "timestamp": T("2024-01-03 10:00")},
        {"dataset": "toy", "user_id": "u1", "article_id": "a2", "timestamp": T("2024-01-04 10:00")},
    ])
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1",
         "timestamp": T("2024-01-03 12:00"), "article_id": "a3", "clicked": 0, "position": 0},
    ])
    click_hist = compute_click_history_features(impressions, history, _articles())

    fake_vecs = {"a1": np.array([1.0, 0.0]), "a2": np.array([0.0, 1.0])}
    monkeypatch.setattr("src.embeddings.embeddings_exist", lambda ds: True)
    monkeypatch.setattr("src.embeddings.load_embeddings",
                         lambda ds: (np.array(list(fake_vecs)), np.array(list(fake_vecs.values()))))

    out = attach_click_history_embeddings(click_hist)
    emb = out.iloc[0]["click_history_embedding"]
    assert emb is not None
    np.testing.assert_allclose(emb, [1.0, 0.0])  # only a1 -- a2's click is in the future


def test_assert_no_leakage_passes_on_clean_synthetic_pipeline():
    history = pd.DataFrame([
        {"dataset": "toy", "user_id": "u1", "article_id": "a1", "timestamp": T("2024-01-03 10:00")},
    ])
    impressions = pd.DataFrame([
        {"impression_id": "i1", "dataset": "toy", "user_id": "u1", "article_id": "a3",
         "timestamp": T("2024-01-11"), "clicked": 0, "position": 0, "split": "val",
         "session_id": None, "read_time": None},
    ])
    articles = _articles()
    click_hist = compute_click_history_features(impressions, history, articles)
    session = compute_session_features(impressions)
    article_dyn = compute_article_dynamic_features(impressions, articles)
    article_dyn = attach_category_match(article_dyn, click_hist)

    behavioral = article_dyn.merge(
        click_hist.drop(columns=["dataset", "user_id"]), on="impression_id")
    behavioral = behavioral.merge(
        session.drop(columns=["dataset", "user_id", "timestamp"]), on="impression_id")

    assert_no_leakage(behavioral)  # should not raise
