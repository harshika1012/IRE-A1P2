#!/usr/bin/env python
"""Q4: serving & scale analysis for the Q2 retrieve-then-rank pipeline
(BM25/semantic candidate generation -> LightGBM re-ranking). Reuses the
model already trained in Q2 (results/reranker_<dataset>_model.txt) --
this script measures serving characteristics, it does not train anything.

1. Index memory: measured size of the BM25 index, the semantic (ANN)
   index, and an in-memory "serving feature store" -- NOT Q1's full
   per-impression training table (that's an offline artifact; a live
   feature store holds one row per article/user, refreshed periodically).
2. Latency: wall-clock p50/p95/p99 for ONE simulated user request, done
   end-to-end (candidate generation over the FULL catalog + feature
   lookup + rerank), repeated --n-requests times.
3. Cost/QPS: back-of-envelope, from the measured single-core latency and
   a stated cloud price/vCPU-hour (an assumption, printed explicitly).
4. Scaling argument: printed, referencing the measured latency breakdown
   and index sizes -- not a separate simulation, per the assignment's own
   "a measured local benchmark plus a scaling argument suffices."

A note on `position` / `position_ctr_prior` (Q9's "features unavailable at
serving time" concern): Q2 trained on the candidate's position WITHIN THE
HISTORICAL DATASET IMPRESSION, which a fresh live request could never have
(that's an artifact of how MIND/EB-NeRD were collected, decided after
ranking). This benchmark instead uses the candidate's rank from THIS
request's own retrieval pass as `position` -- a value that genuinely does
exist before re-ranking runs, so no leakage, but it does mean the model is
now fed a subtly different distribution for that one feature than it saw
in training; that gap is exactly the kind of serving/training mismatch Q9
asks you to be honest about.

Usage:
    python run_serving_benchmark.py --dataset mind
    python run_serving_benchmark.py --dataset all --n-requests 500 --k 150
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.reranker import build_bm25_index, build_semantic_index, FEATURE_COLS
from src.behavioral_features import compute_position_ctr_prior
from src.query_builder import build_query, build_user_embedding

RESULTS_DIR = ROOT / "results"
COLD_START_MAX_CLICKS = 5  # matches src/config.py's COLD_START_MAX_CLICKS


def mb(nbytes) -> float:
    return nbytes / (1024 ** 2)


def build_article_store(articles: pd.DataFrame, impressions: pd.DataFrame) -> pd.DataFrame:
    """One row per article: category, published_time, and a fresh global
    CTR snapshot -- stands in for a feature store's batch-refreshed
    article table (recomputed periodically offline, read in O(1) online)."""
    ctr = impressions.groupby("article_id")["clicked"].mean().rename("popularity_ctr")
    store = articles.set_index("article_id")[["category", "published_time"]].join(ctr)
    store["popularity_ctr"] = store["popularity_ctr"].fillna(0.0)
    return store


def build_user_store(user_features: pd.DataFrame, click_hist: pd.DataFrame) -> pd.DataFrame:
    """One row per user: click history (for query-building) + the most
    complete point-in-time snapshot of Q1's engagement features seen for
    that user -- stands in for a feature store's user profile table."""
    snapshot = click_hist.sort_values("n_clicks_before").drop_duplicates("user_id", keep="last")
    snapshot = snapshot.set_index("user_id")[
        ["n_clicks_before", "recency_weighted_click_count", "is_cold_start", "recent_categories"]]
    store = user_features.set_index("user_id")[["click_history"]].join(snapshot, how="left")
    store["n_clicks_before"] = store["n_clicks_before"].fillna(0).astype(int)
    store["recency_weighted_click_count"] = store["recency_weighted_click_count"].fillna(0.0)
    store["is_cold_start"] = (store["is_cold_start"] != False)  # noqa: E712 -- NaN (no snapshot) -> True
    store["recent_categories"] = store["recent_categories"].apply(lambda x: x if isinstance(x, list) else [])
    return store


def measure_index_memory(bm25_index, semantic_index, article_store, user_store) -> dict:
    W = bm25_index.W
    bm25_bytes = W.data.nbytes + W.indices.nbytes + W.indptr.nbytes
    bm25_vocab_bytes = len(bm25_index.vectorizer.vocabulary_) * 100  # rough per-entry dict overhead

    semantic_bytes = semantic_index.embeddings.nbytes

    report = {
        "bm25_index_mb": mb(bm25_bytes + bm25_vocab_bytes),
        "semantic_index_mb": mb(semantic_bytes),
        "article_store_mb": mb(article_store.memory_usage(deep=True).sum()),
        "user_store_mb": mb(user_store.memory_usage(deep=True).sum()),
    }
    report["total_mb"] = sum(report.values())
    return report


def serve_one_request(user_id, bm25_index, bm25_lookup, semantic_index, id_to_embedding,
                       article_store, user_store, pos_ctr_prior, booster, k, now_ts):
    """One simulated request: candidate generation over the FULL catalog,
    then feature assembly + re-rank for the top-K. Returns
    (ranked_ids, elapsed_seconds, stage_seconds) -- the stage breakdown
    (retrieval / feature_assembly / rerank) is measured, not assumed, so
    the scaling argument can point at the actual bottleneck."""
    t0 = time.perf_counter()

    t_retrieval_start = time.perf_counter()
    history = user_store.at[user_id, "click_history"]
    user_vec = build_user_embedding(history, id_to_embedding)
    if user_vec is None:
        results = bm25_index.top_k(build_query(history, bm25_lookup), k=k)
        cand_ids = [aid for aid, _ in results]
        semantic_scores = np.zeros(len(cand_ids))
        bm25_scores = np.array([s for _, s in results])
    else:
        results = semantic_index.top_k(user_vec, k=k)
        cand_ids = [aid for aid, _ in results]
        semantic_scores = np.array([s for _, s in results])
        bm25_query = build_query(history, bm25_lookup)
        bm25_scores = (np.array(list(bm25_index.score_docs(cand_ids, bm25_query).values()))
                       if bm25_query else np.zeros(len(cand_ids)))
    t_retrieval = time.perf_counter() - t_retrieval_start

    n = len(cand_ids)
    if n == 0:
        elapsed = time.perf_counter() - t0
        return [], elapsed, {"retrieval": t_retrieval, "feature_assembly": 0.0, "rerank": 0.0}

    t_features_start = time.perf_counter()

    recent_categories = user_store.at[user_id, "recent_categories"]
    n_clicks_before = user_store.at[user_id, "n_clicks_before"]
    recency_weighted = user_store.at[user_id, "recency_weighted_click_count"]
    is_cold_start = user_store.at[user_id, "is_cold_start"]

    rows = np.zeros((n, len(FEATURE_COLS)), dtype=np.float64)
    col = {c: i for i, c in enumerate(FEATURE_COLS)}
    for i, aid in enumerate(cand_ids):
        cat = article_store.at[aid, "category"] if aid in article_store.index else None
        published = article_store.at[aid, "published_time"] if aid in article_store.index else pd.NaT
        pop = article_store.at[aid, "popularity_ctr"] if aid in article_store.index else 0.0
        freshness = (now_ts - published) / np.timedelta64(1, "h") if pd.notna(published) else np.nan
        if freshness is not None and not pd.isna(freshness) and freshness < 0:
            freshness = np.nan

        rows[i, col["bm25_score"]] = bm25_scores[i]
        rows[i, col["semantic_score"]] = semantic_scores[i]
        rows[i, col["n_clicks_before"]] = n_clicks_before
        rows[i, col["recency_weighted_click_count"]] = recency_weighted
        rows[i, col["is_cold_start"]] = int(is_cold_start)
        rows[i, col["popularity_prior_ctr"]] = pop
        rows[i, col["freshness_hours"]] = freshness if freshness is not None else np.nan
        rows[i, col["category_match"]] = int(cat in recent_categories) if recent_categories else 0
        rows[i, col["category_match_frac"]] = (
            (sum(c == cat for c in recent_categories) / len(recent_categories)) if recent_categories else np.nan)
        # no live session store in this benchmark -- treated as a fresh session (see module docstring)
        rows[i, col["session_click_count_before"]] = 0
        rows[i, col["session_impressions_before"]] = 0
        rows[i, col["avg_dwell_time_before"]] = np.nan
        rows[i, col["position"]] = i  # THIS request's own retrieval rank -- see module docstring
        rows[i, col["position_ctr_prior"]] = pos_ctr_prior.get(i, np.nan)
    t_features = time.perf_counter() - t_features_start

    t_rerank_start = time.perf_counter()
    scores = booster.predict(rows)
    order = np.argsort(-scores)
    ranked = [cand_ids[j] for j in order]
    t_rerank = time.perf_counter() - t_rerank_start

    elapsed = time.perf_counter() - t0
    return ranked, elapsed, {"retrieval": t_retrieval, "feature_assembly": t_features, "rerank": t_rerank}


def run(dataset: str, n_requests: int, k: int, cost_per_vcpu_hour: float, sla_ms: float):
    print(f"\n=== Q4 serving benchmark: {dataset} ===")
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=[("dataset", "==", dataset)])
    impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet", filters=[("dataset", "==", dataset)])
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet", filters=[("dataset", "==", dataset)])
    click_hist = pd.read_parquet(FEATURES_DIR / "click_history_features.parquet",
                                  filters=[("dataset", "==", dataset)])

    print("  building indexes + feature stores...")
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    semantic_index, id_to_embedding = build_semantic_index(articles, dataset)
    article_store = build_article_store(articles, impressions)
    user_store = build_user_store(user_features, click_hist)
    pos_ctr_prior_df = compute_position_ctr_prior(impressions.assign(split="train"))
    pos_ctr_prior = dict(zip(pos_ctr_prior_df["position"], pos_ctr_prior_df["position_ctr_prior"]))

    booster = lgb.Booster(model_file=str(RESULTS_DIR / f"reranker_{dataset}_model.txt"))

    mem_report = measure_index_memory(bm25_index, semantic_index, article_store, user_store)
    print("\n  -- index / feature-store memory --")
    for k_, v in mem_report.items():
        print(f"    {k_:<20} = {v:8.2f} MB")

    now_ts = impressions["timestamp"].max()
    rng = np.random.default_rng(42)
    sample_users = rng.choice(user_store.index.to_numpy(), size=min(n_requests, len(user_store)), replace=False)

    print(f"\n  -- latency ({len(sample_users)} simulated requests, K={k}) --")
    latencies_ms = []
    stage_totals = {"retrieval": 0.0, "feature_assembly": 0.0, "rerank": 0.0}
    for uid in sample_users:
        _, elapsed, stages = serve_one_request(uid, bm25_index, bm25_lookup, semantic_index, id_to_embedding,
                                                 article_store, user_store, pos_ctr_prior, booster, k, now_ts)
        latencies_ms.append(elapsed * 1000)
        for s, v in stages.items():
            stage_totals[s] += v
    latencies_ms = np.array(latencies_ms)

    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    mean_ms = latencies_ms.mean()
    print(f"    mean = {mean_ms:.2f} ms")
    print(f"    p50  = {p50:.2f} ms")
    print(f"    p95  = {p95:.2f} ms")
    print(f"    p99  = {p99:.2f} ms")

    stage_share = {s: v / sum(stage_totals.values()) for s, v in stage_totals.items()}
    print(f"    stage breakdown (share of total time, measured): "
          + ", ".join(f"{s}={share:.0%}" for s, share in stage_share.items()))

    # back-of-envelope cost/QPS: single-core throughput from mean latency,
    # one vCPU dedicated to serving, target SLA on p99
    single_core_qps = 1000.0 / mean_ms
    cores_for_sla = 1 if p99 <= sla_ms else int(np.ceil(p99 / sla_ms))
    cost_per_1k = (1000.0 / single_core_qps) / 3600.0 * cost_per_vcpu_hour

    print(f"\n  -- cost / QPS (back-of-envelope) --")
    print(f"    single-core throughput      ~ {single_core_qps:.1f} QPS (from mean latency)")
    print(f"    p99 vs SLA ({sla_ms:.0f}ms)          -> {'OK on 1 core' if cores_for_sla == 1 else f'need ~{cores_for_sla}x cores to bring p99 under SLA'}")
    print(f"    cost per 1000 queries        ~ ${cost_per_1k:.4f}  (@ ${cost_per_vcpu_hour:.2f}/vCPU-hr, 1 core, no batching/caching)")

    report = {
        "dataset": dataset, "n_requests": len(sample_users), "k": k,
        "memory_mb": mem_report,
        "latency_ms": {"mean": float(mean_ms), "p50": float(p50), "p95": float(p95), "p99": float(p99)},
        "stage_share": stage_share,
        "cost": {"assumed_usd_per_vcpu_hour": cost_per_vcpu_hour, "sla_ms": sla_ms,
                 "single_core_qps": single_core_qps, "cost_per_1000_queries_usd": cost_per_1k},
    }
    with open(RESULTS_DIR / f"serving_benchmark_{dataset}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/serving_benchmark_{dataset}.json")
    return report


def print_scaling_argument(reports: dict):
    print("\n=== Q4.4 scaling argument (10x load) ===")
    for dataset, r in reports.items():
        mem = r["memory_mb"]
        lat = r["latency_ms"]
        stages = r["stage_share"]
        bottleneck = max(stages, key=stages.get)
        print(f"\n  {dataset}: index+store = {mem['total_mb']:.1f}MB now, p99 = {lat['p99']:.1f}ms now")
        print(f"    measured stage split: retrieval={stages['retrieval']:.0%}, "
              f"feature_assembly={stages['feature_assembly']:.0%}, rerank={stages['rerank']:.0%} "
              f"-> '{bottleneck}' is today's single biggest stage, but that's not the same question as "
              f"which stage's COST GROWS with catalog size (only retrieval's does -- see below).")

        # retrieval (ANNIndex's brute-force cosine search) is the ONLY stage
        # whose cost is O(catalog_size); feature_assembly and rerank are both
        # O(K), the number of retrieved candidates, which doesn't change with
        # catalog growth. So retrieval's SHARE rises with catalog size
        # regardless of which stage happens to be biggest at today's scale --
        # picking whichever stage tops today's benchmark and assuming it's
        # also the one catalog growth will hurt is the wrong inference.
        print(f"    10x CATALOG SIZE -> retrieval is currently {stages['retrieval']:.0%} of wall-clock and "
              f"is the ONLY stage whose cost scales with catalog size (brute-force cosine search over "
              f"every article, O(catalog_size)); feature_assembly and rerank both cost O(K) -- the "
              f"number of RETRIEVED candidates, unrelated to catalog size -- so their absolute cost stays "
              f"flat as the catalog grows. Retrieval's share will keep rising with catalog size even "
              f"though it isn't necessarily the top stage today; the fix, ahead of that, is an "
              f"approximate index (FAISS IVF/HNSW) in place of brute-force IndexFlatIP, trading a little "
              f"recall for sublinear search time.")
        if bottleneck != "retrieval":
            other = "feature_assembly" if bottleneck == "rerank" else "rerank"
            fix = ("vectorizing the per-candidate Python loop (numpy fancy-indexing into the article "
                   "store instead of a per-row .at[] loop)" if bottleneck == "feature_assembly" else
                   "batching LightGBM's predict() call across multiple concurrent requests, or a "
                   "smaller/shallower tree ensemble")
            print(f"    today's actual top stage ('{bottleneck}', {stages[bottleneck]:.0%}) is driven by K "
                  f"(candidates retrieved), not catalog size -- it won't get worse from a bigger catalog, "
                  f"only from a bigger K or more traffic. If it needs to be cheaper independent of scale, "
                  f"the fix is {fix}, not an ANN index (that only helps retrieval's {stages['retrieval']:.0%}).")

        print(f"    10x QUERY VOLUME (QPS) -> the user/article feature stores here are in-process "
              f"Python dicts/DataFrames ({mem['article_store_mb'] + mem['user_store_mb']:.1f}MB); "
              f"that's fine for one process, but a single Python process serializes requests (no "
              f"free parallelism from the GIL for this CPU-bound path), so 10x QPS needs ~10x more "
              f"processes/replicas, not just a faster machine -- this is the first thing that breaks, "
              f"well before the index or model themselves. Beyond a handful of replicas the feature "
              f"stores also need to move out-of-process (Redis/DynamoDB-style) so every replica "
              f"reads a consistent, centrally-updated copy instead of holding a stale local snapshot.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--n-requests", type=int, default=500)
    ap.add_argument("--k", type=int, default=150, help="candidates retrieved per request (spec: 100-200)")
    ap.add_argument("--cost-per-vcpu-hour", type=float, default=0.05,
                     help="assumed cloud cost per vCPU-hour (default: a generic on-demand-instance ballpark)")
    ap.add_argument("--sla-ms", type=float, default=100.0, help="target p99 SLA in ms")
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    reports = {}
    for ds in datasets:
        reports[ds] = run(ds, args.n_requests, args.k, args.cost_per_vcpu_hour, args.sla_ms)
    print_scaling_argument(reports)


if __name__ == "__main__":
    main()
