"""Step 4: build a small, reusable feature store.

article_features.parquet : one row per article -> text fields + category + entities
                            (embeddings are left as a placeholder column here; Part I.3
                            fills these in once you compute/load embeddings)
user_features.parquet    : one row per user -> click history (article_id list) and
                            recency (days since last click), computed ONLY from the
                            train split so val/test never leak into features.
"""
import pandas as pd
from src.config import FEATURES_DIR


def build_article_features(articles: pd.DataFrame) -> pd.DataFrame:
    feats = articles.copy()
    feats["text"] = (
        feats["title"].fillna("") + " " +
        feats["abstract"].fillna("") + " " +
        feats["body"].fillna("")
    ).str.strip()
    feats["embedding"] = None  # filled in later (Part I.3, semantic candidate generation)
    return feats[["article_id", "dataset", "text", "title", "abstract", "category",
                  "subcategory", "entities", "embedding"]]


def build_user_features(history: pd.DataFrame, impressions_train: pd.DataFrame) -> pd.DataFrame:
    # restrict to clicks that happened before/at train cutoff to avoid leakage
    train_cutoff = impressions_train["timestamp"].max()
    hist = history[history["timestamp"] <= train_cutoff]

    grouped = hist.sort_values("timestamp").groupby(["user_id", "dataset"])
    rows = []
    for (uid, ds), g in grouped:
        rows.append({
            "user_id": uid,
            "dataset": ds,
            "click_history": list(g["article_id"]),
            "n_clicks": len(g),
            "last_click_time": g["timestamp"].max(),
            "recency_days": (train_cutoff - g["timestamp"].max()).days,
        })
    return pd.DataFrame(rows)


def save(article_features, user_features):
    article_features.to_parquet(FEATURES_DIR / "article_features.parquet", index=False)
    user_features.to_parquet(FEATURES_DIR / "user_features.parquet", index=False)
    print(f"  saved article_features.parquet ({len(article_features)} rows)")
    print(f"  saved user_features.parquet ({len(user_features)} rows)")