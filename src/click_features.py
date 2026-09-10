"""Assignment 2, Q1: click-history, session, and article-level behavioural
features.

Key design decision vs. Assignment 1's `feature_store.py`
----------------------------------------------------------
Assignment 1 builds ONE static row per user, frozen at the train cutoff
(`build_user_features`). That's correct for a train-only snapshot, but wrong
for val/test: an impression there happens *after* the train cutoff, and the
user may have clicked things since then that are perfectly legitimate to use
at serving time. Freezing everyone at the train cutoff isn't conservative,
it's just the wrong reference point.

So every feature here is computed PER IMPRESSION via pd.merge_asof with
direction="backward", allow_exact_matches=False -- for each impression we ask
"as of the instant strictly before this timestamp, what did the world look
like", so nothing at or after the impression's own time can leak in.

A MIND-specific gotcha
-----------------------
MIND's behaviors.tsv `History` column is frozen BEFORE the eval window starts
and does not actually grow between a user's successive behavior rows the way
EB-NeRD's does. Your parse_mind.py stamps every article in that history with
the *impression row's own timestamp*, so a user with N behavior rows produces
N duplicate copies of the same history entries, each with a different fake
timestamp. Harmless for a single static snapshot (Assignment 1), but it will
inflate a point-in-time recency-weighted score and popularity counts. We
de-duplicate per (user, article) down to the earliest timestamp before using
this table for anything point-in-time -- see `_dedupe_history`. This is a
no-op for EB-NeRD, whose history rows are already one-per-genuine-click.

Usage
-----
    imp = impressions.copy()
    imp = add_click_history_features(imp, history)
    imp = add_recent_article_ids(imp, history, k=20)
    imp = add_session_features(imp)
    imp = add_article_dynamic_features(imp, articles)
    imp = add_category_match_features(imp, history, articles)
    assert_no_future_leakage(imp, history)
"""
import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dedupe_history(history: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated (user, dataset, article) rows down to their earliest
    timestamp. Required for MIND (see module docstring); a no-op for EB-NeRD."""
    return (
        history.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "dataset", "article_id"], keep="first")
    )


def _epoch_seconds(ts: pd.Series) -> pd.Series:
    return ts.astype("int64") / 1e9


# ---------------------------------------------------------------------------
# 1. click-history features
# ---------------------------------------------------------------------------

def add_click_history_features(impressions: pd.DataFrame, history: pd.DataFrame,
                                half_life_days: float = 7.0) -> pd.DataFrame:
    """Adds, per impression, using only clicks strictly BEFORE its timestamp:
        n_clicks_before        - lifetime click count so far
        recency_weighted_hist  - exponentially decayed click count
                                  (half_life_days controls how fast old clicks fade)
        days_since_last_click  - gap to most recent click; NaN if no history yet

    The decay trick: sum_i exp(-lam*(t - click_i)) = exp(-lam*t) * sum_i exp(lam*click_i).
    The right-hand sum is a per-user running total that doesn't depend on t,
    so we can precompute it once with a cumsum and just rescale it per impression
    -- no per-row python loop needed.
    """
    lam = np.log(2) / (half_life_days * SECONDS_PER_DAY)

    hist = _dedupe_history(history).sort_values("timestamp").copy()
    hist["t_epoch"] = _epoch_seconds(hist["timestamp"])
    # Anchor to the earliest history timestamp so the exponent stays small.
    # exp(-lam*(t-ref)) * sum(exp(lam*(c-ref))) == exp(-lam*t) * sum(exp(lam*c))
    # -- the reference cancels out algebraically, but working in absolute
    # Unix-epoch seconds (~1.7e9) overflows exp() instantly, so we must
    # subtract a reference before exponentiating.
    ref_epoch = hist["t_epoch"].min()
    hist["clicks_so_far"] = hist.groupby(["user_id", "dataset"]).cumcount() + 1
    hist["weighted_sum_so_far"] = np.exp(lam * (hist["t_epoch"] - ref_epoch)).groupby(
        [hist["user_id"], hist["dataset"]]).cumsum()

    imp = impressions.sort_values("timestamp").copy()
    merged = pd.merge_asof(
        imp,
        hist[["user_id", "dataset", "timestamp", "clicks_so_far", "weighted_sum_so_far"]]
        .rename(columns={"timestamp": "_hist_ts"}),
        left_on="timestamp", right_on="_hist_ts",
        by=["user_id", "dataset"],
        direction="backward", allow_exact_matches=False,
    )

    merged["n_clicks_before"] = merged["clicks_so_far"].fillna(0).astype(int)
    t_rel = _epoch_seconds(merged["timestamp"]) - ref_epoch
    merged["recency_weighted_hist"] = (
        np.exp(-lam * t_rel) * merged["weighted_sum_so_far"]
    ).fillna(0.0)
    merged["days_since_last_click"] = (
        (merged["timestamp"] - merged["_hist_ts"]).dt.total_seconds() / SECONDS_PER_DAY
    )
    return merged.drop(columns=["clicks_so_far", "weighted_sum_so_far", "_hist_ts"])


def add_recent_article_ids(impressions: pd.DataFrame, history: pd.DataFrame,
                            articles: pd.DataFrame = None, k: int = 20) -> pd.DataFrame:
    """Adds, per impression, using only clicks strictly BEFORE its timestamp:
        recent_article_ids            - last <=k clicked article ids, most recent
                                          first. Join against article_features.parquet
                                          downstream for titles, e.g. for an
                                          attention-based re-ranker (NRMS-style, Q3).
        history_category_distribution - dict {category: count} over that same
                                          <=k window. Only added if `articles` is
                                          passed (needs the article -> category
                                          lookup).
        history_embedding              - elementwise mean of the available article
                                          embeddings in that same window. Only added
                                          if `articles` is passed AND its `embedding`
                                          column has real values -- stays None/NaN
                                          until Assignment 1's embeddings are actually
                                          filled into article_features.parquet (see
                                          that file's own placeholder note).

    Note: this loops per user group (one window slice per impression), which is
    fine at MIND-small/EB-NeRD-demo scale but will need porting to Polars or
    numba if you run it over the large bundles for Q4's scale analysis.
    """
    hist = _dedupe_history(history).sort_values("timestamp")

    has_cat = has_emb = False
    if articles is not None:
        join_cols = ["article_id", "dataset", "category"]
        has_cat = True
        has_emb = "embedding" in articles.columns and articles["embedding"].notna().any()
        if has_emb:
            join_cols.append("embedding")
        hist = hist.merge(articles[join_cols], on=["article_id", "dataset"], how="left")

    keep_cols = ["timestamp", "article_id"] + (["category"] if has_cat else []) + (["embedding"] if has_emb else [])
    hist_by_user = {}
    for key, g in hist.groupby(["user_id", "dataset"], sort=False):
        entry = {
            "times": g["timestamp"].to_numpy(dtype="datetime64[ns]"),
            "ids": g["article_id"].to_numpy(),
        }
        if has_cat:
            entry["categories"] = g["category"].to_numpy()
        if has_emb:
            entry["embeddings"] = g["embedding"].to_numpy()
        hist_by_user[key] = entry

    imp = impressions.reset_index(drop=True)
    imp_times = imp["timestamp"].to_numpy(dtype="datetime64[ns]")
    recent_ids = np.empty(len(imp), dtype=object)
    cat_dist = np.empty(len(imp), dtype=object) if has_cat else None
    hist_emb = np.empty(len(imp), dtype=object) if has_emb else None

    empty_times = np.array([], dtype="datetime64[ns]")
    for key, idx in imp.groupby(["user_id", "dataset"], sort=False).indices.items():
        entry = hist_by_user.get(key)
        times = entry["times"] if entry is not None else empty_times
        for i in idx:
            if entry is None or len(times) == 0:
                recent_ids[i] = []
                if has_cat: cat_dist[i] = None
                if has_emb: hist_emb[i] = None
                continue
            pos = np.searchsorted(times, imp_times[i], side="left")  # strictly-before index
            lo = max(0, pos - k)
            recent_ids[i] = list(entry["ids"][lo:pos][::-1])
            if has_cat:
                cats = entry["categories"][lo:pos]
                cats = cats[~pd.isna(cats)]
                if len(cats):
                    vals, counts = np.unique(cats, return_counts=True)
                    cat_dist[i] = dict(zip(vals.tolist(), counts.tolist()))
                else:
                    cat_dist[i] = None
            if has_emb:
                embs = [e for e in entry["embeddings"][lo:pos]
                        if e is not None and not (isinstance(e, float) and pd.isna(e))]
                hist_emb[i] = list(np.mean(np.stack(embs), axis=0)) if embs else None

    imp = imp.copy()
    imp["recent_article_ids"] = recent_ids
    if has_cat:
        imp["history_category_distribution"] = cat_dist
    if has_emb:
        imp["history_embedding"] = hist_emb
    return imp


# ---------------------------------------------------------------------------
# 2. session features
# ---------------------------------------------------------------------------

def add_session_features(impressions: pd.DataFrame, session_gap_minutes: float = 30.0) -> pd.DataFrame:
    """Adds:
        session_id                       - MIND has no native session id, so we derive
                                            one with the standard web-analytics
                                            convention: a new session starts whenever
                                            the gap since the user's previous impression
                                            exceeds session_gap_minutes. If your raw
                                            EB-NeRD behaviors.parquet has its own
                                            `session_id` column, extend parse_ebnerd.py
                                            to carry it through and this function will
                                            leave it untouched.
        candidate_position                - candidate's rank position within ITS OWN
                                            impression's candidate list. Named
                                            `candidate_position` (not `session_position`)
                                            because a session can span several
                                            impressions -- this has never measured
                                            position within a session, only within one
                                            impression event.
        session_length                    - number of candidates shown in this
                                            impression -- a direct position-bias
                                            control feature
        is_first_in_session               - True if this is the user's first
                                            impression in the derived session
        session_impression_count_before   - # of earlier impression EVENTS by this
                                            user in the same session (genuine
                                            within-session behavioural signal)
        session_click_count_before        - # of clicks made earlier in this same
                                            session (genuine within-session signal)
        time_since_previous_click         - minutes since the last click THIS
                                            session; NaN if no click yet this
                                            session. Distinct from the global,
                                            cross-session `days_since_last_click`
                                            computed by add_click_history_features.
    All three "genuine" session signals are computed strictly BEFORE the current
    impression event, so they carry the same leak-proof guarantee as the other
    Q1 features.
    """
    imp = impressions.copy()

    if "session_id" not in imp.columns:
        order = (imp.drop_duplicates("impression_id")
                    .sort_values(["user_id", "dataset", "timestamp"]))
        gap = order.groupby(["user_id", "dataset"])["timestamp"].diff()
        new_session = gap.isna() | (gap > pd.Timedelta(minutes=session_gap_minutes))
        session_num = new_session.groupby([order["user_id"], order["dataset"]]).cumsum()
        order["session_id"] = (
            order["user_id"].astype(str) + "_" + order["dataset"] + "_" + session_num.astype(str)
        )
        imp = imp.merge(order[["impression_id", "session_id"]], on="impression_id", how="left")

    imp["candidate_position"] = imp["position"]
    imp["session_length"] = imp.groupby("impression_id")["article_id"].transform("count")
    first_ts_per_session = imp.groupby("session_id")["timestamp"].transform("min")
    imp["is_first_in_session"] = imp["timestamp"] == first_ts_per_session

    # ---- genuine within-session behavioural counts ----
    # collapse to one row per impression EVENT (not per candidate) to compute
    # session-level running stats, then broadcast the result back onto every
    # candidate row of that impression.
    imp_level = (
        imp.groupby(["session_id", "impression_id"], as_index=False)
           .agg(timestamp=("timestamp", "first"), n_clicks=("clicked", "sum"))
           .sort_values(["session_id", "timestamp"])
    )
    g = imp_level.groupby("session_id")
    imp_level["session_impression_count_before"] = g.cumcount()
    imp_level["session_click_count_before"] = g["n_clicks"].cumsum() - imp_level["n_clicks"]

    imp_level["_click_time_if_clicked"] = imp_level["timestamp"].where(imp_level["n_clicks"] > 0)
    imp_level["_prior_click_time"] = imp_level.groupby("session_id")["_click_time_if_clicked"].shift(1)
    imp_level["_last_click_time_in_session"] = imp_level.groupby("session_id")["_prior_click_time"].ffill()
    imp_level["time_since_previous_click"] = (
        (imp_level["timestamp"] - imp_level["_last_click_time_in_session"]).dt.total_seconds() / 60.0
    )

    imp = imp.merge(
        imp_level[["impression_id", "session_impression_count_before",
                   "session_click_count_before", "time_since_previous_click"]],
        on="impression_id", how="left",
    )
    return imp


# ---------------------------------------------------------------------------
# 3. article-level dynamic features
# ---------------------------------------------------------------------------

def add_article_dynamic_features(impressions: pd.DataFrame, articles: pd.DataFrame,
                                  half_life_days: float = 3.0) -> pd.DataFrame:
    """Adds, per candidate article shown in an impression:
        popularity_before  - exponentially-decayed count of how many times
                              this article was clicked by ANYONE strictly
                              before this impression's timestamp (fast decay
                              by default -- news popularity is short-lived).
                              Built from impressions' own clicked==1 rows
                              rather than the `history` table, since impression
                              timestamps are genuine per-event and don't have
                              the MIND duplication issue described above.
        freshness_hours    - hours since the article was published (NaN for
                              MIND, which doesn't ship publish timestamps)
    Category match is handled separately in add_category_match_features,
    since it needs the user's history table too.
    """
    lam = np.log(2) / (half_life_days * SECONDS_PER_DAY)

    clicks = impressions.loc[impressions["clicked"] == 1, ["dataset", "article_id", "timestamp"]].copy()
    clicks = clicks.sort_values("timestamp")
    clicks["t_epoch"] = _epoch_seconds(clicks["timestamp"])
    # Same anchoring as add_click_history_features: subtract a reference
    # epoch before exponentiating or this overflows (absolute Unix epoch
    # seconds are ~1.7e9, and exp() overflows well before that).
    ref_epoch = clicks["t_epoch"].min() if len(clicks) else 0.0
    clicks["weighted_sum_so_far"] = np.exp(lam * (clicks["t_epoch"] - ref_epoch)).groupby(
        [clicks["dataset"], clicks["article_id"]]).cumsum()

    imp = impressions.sort_values("timestamp").copy()
    merged = pd.merge_asof(
        imp,
        clicks[["dataset", "article_id", "timestamp", "weighted_sum_so_far"]]
        .rename(columns={"timestamp": "_click_ts"}),
        left_on="timestamp", right_on="_click_ts",
        by=["dataset", "article_id"],
        direction="backward", allow_exact_matches=False,
    )
    t_rel = _epoch_seconds(merged["timestamp"]) - ref_epoch
    merged["popularity_before"] = (np.exp(-lam * t_rel) * merged["weighted_sum_so_far"]).fillna(0.0)
    merged = merged.drop(columns=["weighted_sum_so_far", "_click_ts"])

    merged = merged.merge(
        articles[["article_id", "dataset", "category", "published_time"]],
        on=["article_id", "dataset"], how="left",
    )
    merged["freshness_hours"] = (
        (merged["timestamp"] - merged["published_time"]).dt.total_seconds() / 3600.0
    )
    return merged


def add_category_match_features(impressions_with_features: pd.DataFrame, history: pd.DataFrame,
                                 articles: pd.DataFrame) -> pd.DataFrame:
    """Adds `category_match`: 1.0 if the candidate's category equals the
    user's single most-clicked category strictly BEFORE this impression,
    0.0 if not, NaN if the user has no qualifying history yet.
    `impressions_with_features` must already have a `category` column
    (added by add_article_dynamic_features)."""
    hist = (
        _dedupe_history(history)
        .merge(articles[["article_id", "dataset", "category"]], on=["article_id", "dataset"], how="left")
        .sort_values("timestamp")
    )

    def _top_category_so_far(g: pd.DataFrame) -> pd.Series:
        top, counts = [], {}
        for c in g["category"]:
            top.append(max(counts, key=counts.get) if counts else None)
            if c is not None and not pd.isna(c):
                counts[c] = counts.get(c, 0) + 1
        return pd.Series(top, index=g.index)

    hist["top_category_before"] = (
        hist.groupby(["user_id", "dataset"], group_keys=False).apply(_top_category_so_far, include_groups=False)
    )

    imp = impressions_with_features.sort_values("timestamp").copy()
    merged = pd.merge_asof(
        imp,
        hist[["user_id", "dataset", "timestamp", "top_category_before"]]
        .rename(columns={"timestamp": "_hist_ts"}),
        left_on="timestamp", right_on="_hist_ts", by=["user_id", "dataset"],
        direction="backward", allow_exact_matches=False,
    )
    merged["category_match"] = np.where(
        merged["top_category_before"].isna(), np.nan,
        (merged["category"] == merged["top_category_before"]).astype(float),
    )
    return merged.drop(columns=["top_category_before", "_hist_ts"])


# ---------------------------------------------------------------------------
# anti-leakage check -- required by Q1's boundary rule and Q9's anti-gaming test
# ---------------------------------------------------------------------------

def assert_no_future_leakage(features: pd.DataFrame, history: pd.DataFrame, n_sample: int = 200):
    """Fails loudly if any future click leaked into a feature:
      1. no impression should have a negative days_since_last_click
      2. for a random sample, brute-force recompute n_clicks_before and check
         it matches the vectorised merge_asof result exactly
    Run this in CI / a pytest test, not just manually -- Q9 explicitly asks
    for a test asserting the behaviour-window boundary."""
    assert (features["days_since_last_click"].dropna() >= 0).all(), \
        "found a negative days_since_last_click -- a future click leaked in"

    hist = _dedupe_history(history)
    sample = features.sample(min(n_sample, len(features)), random_state=0)
    for _, row in sample.iterrows():
        brute = hist[
            (hist["user_id"] == row["user_id"]) & (hist["dataset"] == row["dataset"]) &
            (hist["timestamp"] < row["timestamp"])
        ]
        assert len(brute) == row["n_clicks_before"], (
            f"leakage/undercount for user {row['user_id']}: "
            f"brute-force={len(brute)} vs feature={row['n_clicks_before']}"
        )
    print(f"  no-leakage check passed on {len(sample)} sampled impressions")