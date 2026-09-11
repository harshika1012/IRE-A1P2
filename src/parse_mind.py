"""Step 2 (MIND side): news.tsv + behaviors.tsv -> unified schema.

MIND raw formats:
  news.tsv       : News ID, Category, SubCategory, Title, Abstract, URL,
                    Title Entities (json), Abstract Entities (json)
  behaviors.tsv  : Impression ID, User ID, Time, History (space-sep News IDs),
                    Impressions (space-sep "NewsID-0/1")
"""
import pandas as pd
from src.config import MIND_TRAIN_DIR, MIND_DEV_DIR

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _load_news(split_dir):
    df = pd.read_csv(split_dir / "news.tsv", sep="\t", header=None, names=NEWS_COLS)
    df["entities"] = df["title_entities"].fillna("") + "|" + df["abstract_entities"].fillna("")
    out = pd.DataFrame({
        "article_id": df["news_id"].astype(str),
        "dataset": "mind",
        "title": df["title"],
        "abstract": df["abstract"],
        "body": None,  # MIND does not ship full body text
        "category": df["category"],
        "subcategory": df["subcategory"],
        "entities": df["entities"],
        "published_time": pd.NaT,
        "url": df["url"],
    })
    return out


def _load_behaviors(split_dir):
    df = pd.read_csv(split_dir / "behaviors.tsv", sep="\t", header=None, names=BEH_COLS)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")

    # long-format click history
    hist_rows = []
    for _, row in df.iterrows():
        if pd.isna(row["history"]):
            continue
        for aid in str(row["history"]).split():
            hist_rows.append((str(row["user_id"]), "mind", str(aid), row["time"]))
    history_df = pd.DataFrame(hist_rows, columns=["user_id", "dataset", "article_id", "timestamp"])

    # exploded impressions (one row per candidate article). MIND has no
    # session_id or dwell-time signal (unlike EB-NeRD) -- keep the columns
    # for schema parity but leave them null; src/behavioral_features.py
    # derives session boundaries from a time-gap heuristic instead.
    imp_rows = []
    for _, row in df.iterrows():
        for pos, tok in enumerate(str(row["impressions"]).split()):
            aid, label = tok.rsplit("-", 1)
            imp_rows.append((str(row["impression_id"]), "mind", str(row["user_id"]), row["time"],
                              str(aid), int(label), pos, None, None))
    impressions_df = pd.DataFrame(imp_rows, columns=[
        "impression_id", "dataset", "user_id", "timestamp", "article_id", "clicked", "position",
        "session_id", "read_time"])

    return impressions_df, history_df


def load_mind():
    articles = pd.concat([_load_news(MIND_TRAIN_DIR), _load_news(MIND_DEV_DIR)],
                          ignore_index=True).drop_duplicates("article_id")
    imp_train, hist_train = _load_behaviors(MIND_TRAIN_DIR)
    imp_dev, hist_dev = _load_behaviors(MIND_DEV_DIR)
    impressions = pd.concat([imp_train, imp_dev], ignore_index=True)
    history = pd.concat([hist_train, hist_dev], ignore_index=True)
    return articles, impressions, history