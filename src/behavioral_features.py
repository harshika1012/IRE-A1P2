"""Assignment 2, Part I Q1: behavioural features from click-logs.

Everything here is point-in-time correct: a feature attached to an
impression only ever aggregates events with a timestamp strictly before
that impression's own timestamp (candidates sharing the exact same
timestamp are bucketed together and excluded from each other -- see
`compute_article_dynamic_features`). This is the "behavioural-window
boundary" the assignment requires enforced at both training and serving
time; `tests/test_behavioral_window.py` checks it holds.

Four groups of features, each a function below:
  1. compute_click_history_features -- per impression: recent clicked
     titles/categories, click count, exponential recency decay.
  2. compute_session_features       -- per impression: within-session
     click count so far, dwell time (EB-NeRD only; MIND has no signal).
  3. compute_article_dynamic_features / attach_category_match -- per
     candidate: time-aware popularity, freshness, category match with
     the user's history.
  4. compute_position_ctr_prior     -- corpus-level position bias, fit
     on the train split only.

build_behavioral_features() wires all four into one table.
"""
import numpy as np
import pandas as pd

from src.config import (N_RECENT_CLICKS, RECENCY_HALF_LIFE_DAYS,
                         COLD_START_MAX_CLICKS, SESSION_GAP_MINUTES)


def _decay_weight(age_days, half_life_days=RECENCY_HALF_LIFE_DAYS):
    return 0.5 ** (np.asarray(age_days, dtype=float) / half_life_days)


def compute_click_history_features(impressions: pd.DataFrame, history: pd.DataFrame,
                                    articles: pd.DataFrame,
                                    n_recent: int = N_RECENT_CLICKS,
                                    half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> pd.DataFrame:
    """One row per impression_id: point-in-time click-history features.

    Columns: impression_id, dataset, user_id, n_clicks_before,
    days_since_last_click, recency_weighted_click_count, is_cold_start,
    recent_article_ids (list[str]), recent_titles (list[str]),
    recent_categories (list[str]). Embeddings are attached separately by
    attach_click_history_embeddings (needs data/features/embeddings_*.npz
    from Part I.3, which not every caller has built yet).
    """
    hist = history.sort_values(["dataset", "user_id", "timestamp"])
    imp_unique = (impressions.drop_duplicates("impression_id")
                  [["impression_id", "dataset", "user_id", "timestamp"]])

    # pre-group history once (O(n)) instead of re-filtering per user inside the loop
    hist_groups = {
        key: (g["timestamp"].values, g["article_id"].values)
        for key, g in hist.groupby(["dataset", "user_id"], sort=False, observed=True)
    }
    empty = (np.array([], dtype="datetime64[ns]"), np.array([], dtype=object))

    rows = []
    for (ds, uid), g_imp in imp_unique.groupby(["dataset", "user_id"], sort=False, observed=True):
        times, aids = hist_groups.get((ds, uid), empty)
        q_times = g_imp["timestamp"].values
        cuts = np.searchsorted(times, q_times, side="left")  # strict "<" boundary
        for iid, t, cut in zip(g_imp["impression_id"], q_times, cuts):
            prior_times = times[:cut]
            recent_ids = list(aids[:cut][-n_recent:])
            if cut > 0:
                days_since = (t - prior_times[-1]) / np.timedelta64(1, "D")
                age_days = (t - prior_times) / np.timedelta64(1, "D")
                recency_w = float(_decay_weight(age_days, half_life_days).sum())
            else:
                days_since = np.nan
                recency_w = 0.0
            rows.append({
                "impression_id": iid, "dataset": ds, "user_id": uid,
                "n_clicks_before": int(cut),
                "days_since_last_click": days_since,
                "recency_weighted_click_count": recency_w,
                "is_cold_start": cut <= COLD_START_MAX_CLICKS,
                "recent_article_ids": recent_ids,
            })
    out = pd.DataFrame(rows)

    # map recent article_ids -> titles/categories via a vectorized explode+merge
    # (avoids a python-level dict lookup per row across millions of impressions)
    recent = out[["impression_id", "dataset", "recent_article_ids"]].explode("recent_article_ids")
    recent = recent.dropna(subset=["recent_article_ids"])
    recent = recent.merge(
        articles[["dataset", "article_id", "title", "category"]],
        left_on=["dataset", "recent_article_ids"], right_on=["dataset", "article_id"], how="left")
    titles = recent.groupby("impression_id", observed=True)["title"].apply(list).rename("recent_titles")
    cats = recent.groupby("impression_id", observed=True)["category"].apply(list).rename("recent_categories")

    out = out.merge(titles, on="impression_id", how="left").merge(cats, on="impression_id", how="left")
    out["recent_titles"] = out["recent_titles"].apply(lambda x: x if isinstance(x, list) else [])
    out["recent_categories"] = out["recent_categories"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def attach_click_history_embeddings(click_hist_features: pd.DataFrame) -> pd.DataFrame:
    """Adds click_history_embedding: a mean-pooled, re-normalized embedding
    of the user's point-in-time recent_article_ids (Q1's 'embeddings'
    click-history feature), reusing Part I.3's precomputed article
    embeddings and the same pooling as src/query_builder.py's BM25/semantic
    retrieval so results stay comparable. Requires
    data/features/embeddings_<dataset>.npz to already exist (build it via
    run_semantic.py first); datasets without one get None embeddings.

    Only ever call this on the small per-impression click-history table
    (one row per impression_id), never on the exploded per-candidate table
    -- broadcasting a dense vector across every candidate in an impression
    would multiply its memory footprint by the average candidate-set size.
    """
    from src.embeddings import embeddings_exist, load_embeddings
    from src.query_builder import build_user_embedding

    out = click_hist_features.copy()
    out["click_history_embedding"] = None
    for ds in out["dataset"].unique():
        if not embeddings_exist(ds):
            print(f"  [skip] no embeddings_{ds}.npz found -- run run_semantic.py first; "
                  f"click_history_embedding will be None for {ds}")
            continue
        article_ids, vecs = load_embeddings(ds)
        lookup = dict(zip(article_ids, vecs))
        mask = (out["dataset"] == ds).values
        pooled = [build_user_embedding(ids, lookup, n_recent=N_RECENT_CLICKS)
                  for ids in out.loc[mask, "recent_article_ids"]]
        # assigning a plain list of same-length arrays lets pandas collapse it
        # into a 2D block and raise "equal len keys and value" -- an explicit
        # object ndarray keeps each row's vector (or None) as one opaque cell
        vals = np.empty(len(pooled), dtype=object)
        vals[:] = pooled
        out.loc[mask, "click_history_embedding"] = vals
    return out


def compute_session_features(impressions: pd.DataFrame,
                              gap_minutes: float = SESSION_GAP_MINUTES) -> pd.DataFrame:
    """One row per impression_id: within-session click pattern + dwell time.

    EB-NeRD ships an explicit `session_id` and `read_time` (seconds spent on
    the article read just before this impression); MIND ships neither, so
    we derive session boundaries from a `gap_minutes` inactivity threshold
    per user and leave dwell time as NaN (no equivalent signal in MIND).
    """
    imp_meta = impressions.drop_duplicates("impression_id")[
        ["impression_id", "dataset", "user_id", "timestamp", "session_id", "read_time"]].copy()
    had_click = impressions.groupby("impression_id", observed=True)["clicked"].max().rename("had_click")
    imp = imp_meta.merge(had_click, on="impression_id").sort_values(["dataset", "user_id", "timestamp"])
    imp["dataset"] = imp["dataset"].astype(str)
    imp["user_id"] = imp["user_id"].astype(str)

    # derive a session id per user from inactivity gaps, for datasets without one
    derived = np.empty(len(imp), dtype=float)
    for _, g in imp.groupby(["dataset", "user_id"], sort=False, observed=True):
        idx = g.index.values
        t = g["timestamp"].values
        gap_min = np.diff(t) / np.timedelta64(1, "m")
        new_session = np.concatenate([[True], gap_min > gap_minutes])
        derived[imp.index.get_indexer(idx)] = np.cumsum(new_session)
    imp["derived_session_id"] = derived

    has_native = imp["session_id"].notna()
    imp["session_key"] = np.where(
        has_native,
        imp["dataset"] + "_u" + imp["user_id"] + "_sid" + imp["session_id"].astype(str),
        imp["dataset"] + "_u" + imp["user_id"] + "_dsid" + imp["derived_session_id"].astype(str),
    )
    imp["session_key"] = imp["session_key"].astype("category")

    # within-session click pattern: strictly-prior impressions in the same session
    imp = imp.sort_values(["session_key", "timestamp"])
    g = imp.groupby("session_key", observed=True)
    imp["session_click_count_before"] = g["had_click"].cumsum() - imp["had_click"]
    imp["session_impressions_before"] = g.cumcount()

    # dwell time (EB-NeRD only): mean read_time over this user's PRIOR impressions
    imp = imp.sort_values(["dataset", "user_id", "timestamp"])
    imp["read_time"] = pd.to_numeric(imp["read_time"], errors="coerce")
    ug = imp.groupby(["dataset", "user_id"], observed=True)
    read_filled = imp["read_time"].fillna(0.0)
    read_notna = imp["read_time"].notna().astype(float)
    cum_sum = ug["read_time"].transform(lambda s: s.fillna(0.0).cumsum()) - read_filled
    cum_n = ug["read_time"].transform(lambda s: s.notna().astype(float).cumsum()) - read_notna
    imp["avg_dwell_time_before"] = cum_sum / cum_n.replace(0, np.nan)

    return imp[["impression_id", "dataset", "user_id", "timestamp", "session_key",
                "session_click_count_before", "session_impressions_before",
                "avg_dwell_time_before"]]


def compute_article_dynamic_features(impressions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """One row per (impression_id, article_id) candidate: time-aware
    popularity-so-far and freshness. Call on the exploded impressions
    table (one row per candidate).

    Popularity uses a strictly-prior cumulative click/impression count per
    article: same-timestamp candidates are bucketed together first so they
    never see each other's label, then only strictly-earlier buckets count.
    """
    # dataset/article_id as category (not object strings) keeps every merge/groupby
    # below cheap even at millions of rows -- this table is one row per candidate,
    # the largest one this module touches.
    imp = impressions[["impression_id", "dataset", "user_id", "article_id",
                        "timestamp", "clicked", "position", "split"]].copy()
    for col in ("impression_id", "dataset", "user_id", "article_id"):
        imp[col] = imp[col].astype("category")

    # groupby on the MultiIndex (no reset_index) so cumsum stays aligned to it --
    # then look candidates up via Series.reindex() rather than DataFrame.merge().
    # merge() always rebuilds the ENTIRE left table (all columns, via a hash
    # join); reindex() only allocates the one new column being looked up. At
    # 9M+ rows and ~8 merges in this module, that difference is the whole
    # reason this pipeline used to peak near 9GB RAM and get OOM-killed.
    bucket = (imp.groupby(["dataset", "article_id", "timestamp"], observed=True)["clicked"]
                 .agg(n_clicks="sum", n_impressions="count"))
    bg = bucket.groupby(level=["dataset", "article_id"], observed=True)
    cum_clicks_prior = bg["n_clicks"].cumsum() - bucket["n_clicks"]
    cum_impressions_prior = bg["n_impressions"].cumsum() - bucket["n_impressions"]
    popularity_prior_ctr = cum_clicks_prior / cum_impressions_prior.replace(0, np.nan)

    bucket_key = pd.MultiIndex.from_arrays([imp["dataset"], imp["article_id"], imp["timestamp"]])
    imp["cum_clicks_prior"] = cum_clicks_prior.reindex(bucket_key).to_numpy()
    imp["cum_impressions_prior"] = cum_impressions_prior.reindex(bucket_key).to_numpy()
    imp["popularity_prior_ctr"] = popularity_prior_ctr.reindex(bucket_key).to_numpy()

    # cast onto imp's exact categories: articles rows for article_ids that never
    # appear as a candidate become NaN here, which is fine -- a left-lookup from
    # imp only ever looks up imp's own keys, so those rows are simply unused.
    art = articles[["dataset", "article_id", "category", "published_time"]].copy()
    art["dataset"] = art["dataset"].astype(imp["dataset"].dtype)
    art["article_id"] = art["article_id"].astype(imp["article_id"].dtype)
    art = art.dropna(subset=["dataset", "article_id"]).drop_duplicates(["dataset", "article_id"])
    art_idx = art.set_index(["dataset", "article_id"])
    art_key = pd.MultiIndex.from_arrays([imp["dataset"], imp["article_id"]])
    imp["category"] = art_idx["category"].reindex(art_key).astype("category").to_numpy()
    published_time = art_idx["published_time"].reindex(art_key).to_numpy()

    imp["freshness_hours"] = (imp["timestamp"].to_numpy() - published_time) / np.timedelta64(1, "h")
    # negative = article "published" after it was shown (bad metadata / clock skew,
    # e.g. MIND ships no published_time at all) -- never report a negative age
    imp.loc[imp["freshness_hours"] < 0, "freshness_hours"] = np.nan
    return imp


def attach_category_match(article_dynamic: pd.DataFrame, click_hist_features: pd.DataFrame) -> pd.DataFrame:
    """Adds category_match (bool) and category_match_frac to article_dynamic
    (mutated in place -- see compute_article_dynamic_features's docstring on
    why this module avoids merge() on the big per-candidate table), computed
    against the point-in-time recent_categories from
    compute_click_history_features.
    """
    hist = click_hist_features[["impression_id", "recent_categories"]].copy()
    hist["impression_id"] = hist["impression_id"].astype(article_dynamic["impression_id"].dtype)
    exploded = hist.explode("recent_categories").rename(columns={"recent_categories": "category"}).dropna()

    match_index = pd.MultiIndex.from_frame(exploded[["impression_id", "category"]].drop_duplicates())
    counts = exploded.groupby(["impression_id", "category"], observed=True).size()
    hist_len = hist.assign(hist_len=hist["recent_categories"].apply(len)).set_index("impression_id")["hist_len"]

    out = article_dynamic
    key = pd.MultiIndex.from_arrays([out["impression_id"], out["category"]])
    out["category_match"] = key.isin(match_index)
    hist_len_per_row = hist_len.reindex(out["impression_id"]).to_numpy()
    count_per_row = counts.reindex(key).fillna(0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 for cold-start rows, replaced by NaN below
        out["category_match_frac"] = np.where(hist_len_per_row > 0, count_per_row / hist_len_per_row, np.nan)
    return out


def compute_position_ctr_prior(impressions: pd.DataFrame) -> pd.DataFrame:
    """Empirical CTR-by-position, fit on the train split only -- used as a
    position-bias prior/feature at every split. Never fit on val/test."""
    train = impressions[impressions["split"] == "train"]
    return (train.groupby(["dataset", "position"], observed=True)["clicked"].mean()
                 .rename("position_ctr_prior").reset_index())


def build_behavioral_features(impressions: pd.DataFrame, history: pd.DataFrame,
                               articles: pd.DataFrame, with_embeddings: bool = True):
    """Wire the four feature groups into one (impression_id, article_id)
    table. Returns (behavioral, click_hist, session) -- the last two are
    also returned since they're independently useful (e.g. click_hist for
    slicing cold-start vs. warm users in Q4).

    click_hist keeps the per-user list/vector columns (recent_titles,
    recent_categories, click_history_embedding); `behavioral` (one row per
    CANDIDATE) deliberately only gets the scalar click-history features --
    broadcasting a list or a 384-dim embedding across every candidate in an
    impression would multiply its memory footprint by the average
    candidate-set size, for no benefit (category_match/_frac already
    distill recent_categories down to a per-candidate scalar).
    """
    click_hist = compute_click_history_features(impressions, history, articles)
    if with_embeddings:
        click_hist = attach_click_history_embeddings(click_hist)
    session = compute_session_features(impressions)
    article_dyn = compute_article_dynamic_features(impressions, articles)
    article_dyn = attach_category_match(article_dyn, click_hist)
    pos_prior = compute_position_ctr_prior(impressions)

    # article_dyn's impression_id/dataset are categorical (see compute_article_dynamic_features).
    # Assign each small table's columns onto article_dyn via Series.reindex(),
    # never merge() -- merge rebuilds the entire (multi-million row) left
    # table for every call; reindex only allocates the one new column.
    behavioral = article_dyn  # mutated in place from here on, not copied
    imp_key = behavioral["impression_id"]

    click_hist_small = click_hist.set_index(
        click_hist["impression_id"].astype(imp_key.dtype))[
        ["n_clicks_before", "days_since_last_click", "recency_weighted_click_count", "is_cold_start"]]
    for col in click_hist_small.columns:
        behavioral[col] = click_hist_small[col].reindex(imp_key).to_numpy()

    session_small = session.set_index(session["impression_id"].astype(imp_key.dtype))[
        ["session_click_count_before", "session_impressions_before", "avg_dwell_time_before"]]
    for col in session_small.columns:
        behavioral[col] = session_small[col].reindex(imp_key).to_numpy()

    pos_prior_idx = pos_prior.set_index(
        [pos_prior["dataset"].astype(behavioral["dataset"].dtype), pos_prior["position"]])["position_ctr_prior"]
    pos_key = pd.MultiIndex.from_arrays([behavioral["dataset"], behavioral["position"]])
    behavioral["position_ctr_prior"] = pos_prior_idx.reindex(pos_key).to_numpy()

    return behavioral, click_hist, session


def build_and_save_behavioral_features(impressions: pd.DataFrame, history: pd.DataFrame,
                                        articles: pd.DataFrame, features_dir):
    """build_behavioral_features() + the leakage check + writing the three
    parquet files -- shared by build_pipeline.py's optional Step 5 and
    build_behavioral_features.py so the two don't drift apart."""
    behavioral, click_hist, session = build_behavioral_features(impressions, history, articles)
    assert_no_leakage(behavioral)
    print("  OK: no future clicks / future-published articles leak into any feature")

    behavioral.to_parquet(features_dir / "behavioral_features.parquet", index=False)
    click_hist.to_parquet(features_dir / "click_history_features.parquet", index=False)
    session.to_parquet(features_dir / "session_features.parquet", index=False)
    print(f"  saved behavioral_features.parquet ({len(behavioral)} rows, {behavioral.shape[1]} cols), "
          f"click_history_features.parquet ({len(click_hist)} rows), "
          f"session_features.parquet ({len(session)} rows) to {features_dir}")
    return behavioral, click_hist, session


def assert_no_leakage(behavioral: pd.DataFrame) -> None:
    """Cheap structural checks for the anti-gaming requirement: raises
    AssertionError if any feature could see its own impression's future.
    Not a substitute for tests/test_behavioral_window.py's unit tests, but
    cheap enough to run every time the pipeline builds features."""
    assert (behavioral["days_since_last_click"].dropna() >= 0).all(), \
        "found a 'last click' at or after its own impression's timestamp"
    assert (behavioral["freshness_hours"].dropna() >= 0).all(), \
        "found an article 'published' after the impression that showed it"
    assert (behavioral["n_clicks_before"] >= 0).all()
    assert (behavioral["cum_clicks_prior"] >= 0).all()
    assert (behavioral["cum_clicks_prior"] <= behavioral["cum_impressions_prior"]).all()
