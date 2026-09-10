"""Step 2 (EB-NeRD side): articles/behaviors/history parquet -> unified schema.

EB-NeRD official columns (RecSys'24 Challenge release):
  articles.parquet  : article_id, title, subtitle, body, category_str,
                       ner_clusters, published_time, url, ...
  behaviors.parquet : impression_id, user_id, impression_time,
                       article_ids_inview, article_ids_clicked, ...
  history.parquet   : user_id, article_id_fixed, impression_time_fixed, ...

NOTE: EB-NeRD is Danish; column names above match the public schema as of
the challenge release. If your downloaded parquet has slightly different
column names, adjust the `.rename()` maps below to match — run
`pd.read_parquet(path).columns` once to check.
"""
import pandas as pd
from src.config import EBNERD_DIR


def _load_articles():
    # articles.parquet is shared; try top-level first, fall back to the train/ folder
    path = EBNERD_DIR / "articles.parquet"
    if not path.exists():
        path = EBNERD_DIR / "train" / "articles.parquet"
    df = pd.read_parquet(path)
    out = pd.DataFrame({
        "article_id": df["article_id"].astype(str),
        "dataset": "ebnerd",
        "title": df.get("title"),
        "abstract": df.get("subtitle"),
        "body": df.get("body"),
        "category": df.get("category_str", df.get("category")),
        "subcategory": None,
        "entities": df.get("ner_clusters", pd.Series([None] * len(df))).astype(str),
        "published_time": pd.to_datetime(df.get("published_time"), errors="coerce"),
        "url": df.get("url"),
    })
    return out


def _load_behaviors():
    # EB-NeRD ships pre-split train/ and validation/ folders; load both and
    # let our own temporal_split.py decide train/val/test from timestamps.
    dfs = []
    for sub in ("train", "validation"):
        p = EBNERD_DIR / sub / "behaviors.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    df["impression_time"] = pd.to_datetime(df["impression_time"])

    imp_rows = []
    for _, row in df.iterrows():
        inview = row["article_ids_inview"]
        clicked = set(row["article_ids_clicked"]) if row["article_ids_clicked"] is not None else set()
        if inview is None:
            continue
        for pos, aid in enumerate(inview):
            imp_rows.append((str(row["impression_id"]), "ebnerd", str(row["user_id"]), row["impression_time"],
                              str(aid), int(aid in clicked), pos))
    impressions_df = pd.DataFrame(imp_rows, columns=[
        "impression_id", "dataset", "user_id", "timestamp", "article_id", "clicked", "position"])
    return impressions_df


def _load_history():
    dfs = []
    for sub in ("train", "validation"):
        p = EBNERD_DIR / sub / "history.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    hist_rows = []
    for _, row in df.iterrows():
        aids = row["article_id_fixed"]
        times = row["impression_time_fixed"]
        if aids is None:
            continue
        for aid, t in zip(aids, times):
            hist_rows.append((str(row["user_id"]), "ebnerd", str(aid), pd.to_datetime(t)))
    return pd.DataFrame(hist_rows, columns=["user_id", "dataset", "article_id", "timestamp"])


def load_ebnerd():
    articles = _load_articles()
    impressions = _load_behaviors()
    history = _load_history()
    return articles, impressions, history