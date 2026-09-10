"""Run this from your Assignment 1 root, after unzipping, to see each
dataset's date range so you can pick sensible TEST_DAYS / VAL_DAYS."""
import pandas as pd

# MIND
mind_train = pd.read_csv("data/raw/mind/train/behaviors.tsv", sep="\t", header=None,
                          names=["impression_id","user_id","time","history","impressions"])
mind_dev = pd.read_csv("data/raw/mind/dev/behaviors.tsv", sep="\t", header=None,
                        names=["impression_id","user_id","time","history","impressions"])
mind_times = pd.to_datetime(pd.concat([mind_train["time"], mind_dev["time"]]),
                             format="%m/%d/%Y %I:%M:%S %p")
print("MIND   :", mind_times.min(), "->", mind_times.max(),
      f"({(mind_times.max()-mind_times.min()).days} days)")

# EB-NeRD
eb_train = pd.read_parquet("data/raw/ebnerd/demo/train/behaviors.parquet")
eb_val = pd.read_parquet("data/raw/ebnerd/demo/validation/behaviors.parquet")
eb_times = pd.to_datetime(pd.concat([eb_train["impression_time"], eb_val["impression_time"]]))
print("EB-NeRD:", eb_times.min(), "->", eb_times.max(),
      f"({(eb_times.max()-eb_times.min()).days} days)")