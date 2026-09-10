"""Quick diagnostic: for a sample of impressions, does the query share ANY
token with the ground-truth clicked article? If not, BM25 can never recall
it regardless of K -- this separates 'model can't retrieve it' from 'model
just isn't ranking it high enough'."""
import sys, random
sys.path.insert(0, ".")
import pandas as pd
from src.config import PROCESSED_DIR, FEATURES_DIR
from src.query_builder import build_query
from src.tokenizer import tokenize

DATASET = "ebnerd"   # change to "ebnerd" to check the other
SPLIT = "val"
SAMPLE = 2000

articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet")
impressions = pd.read_parquet(PROCESSED_DIR / "impressions.parquet")
user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet")

articles_ds = articles[articles["dataset"] == DATASET]
article_lookup = dict(zip(articles_ds["article_id"],
    (articles_ds["title"].fillna("") + " " + articles_ds["abstract"].fillna(""))))
uf = user_features[user_features["dataset"] == DATASET].set_index("user_id")["click_history"].to_dict()

imp = impressions[(impressions["dataset"] == DATASET) & (impressions["split"] == SPLIT)]
truth = imp[imp["clicked"] == 1].groupby("impression_id")["article_id"].apply(set)
imp_users = imp.drop_duplicates("impression_id").set_index("impression_id")["user_id"]

ids = list(truth.index)
random.seed(0)
ids = random.sample(ids, min(SAMPLE, len(ids)))

no_overlap = 0
checked = 0
for iid in ids:
    uid = imp_users.loc[iid]
    hist = uf.get(uid, [])
    q = build_query(hist, article_lookup)
    if not q:
        continue
    q_tokens = set(tokenize(q))
    gt_id = next(iter(truth.loc[iid]))
    gt_text = article_lookup.get(gt_id, "")
    gt_tokens = set(tokenize(gt_text))
    checked += 1
    if not (q_tokens & gt_tokens):
        no_overlap += 1

print(f"{DATASET}: {no_overlap}/{checked} ({no_overlap/checked:.1%}) queries share ZERO tokens with the ground-truth article -> hard ceiling for BM25 regardless of K")