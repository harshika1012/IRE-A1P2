"""Step 3: temporal split. Last TEST_DAYS = test, preceding VAL_DAYS = val, rest = train.
Applied independently per dataset (MIND and EB-NeRD cover different calendar ranges),
then concatenated back together with a `split` column.
"""
import pandas as pd
from src.config import SPLIT_DAYS


def _split_one(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days = SPLIT_DAYS[dataset_name]
    max_t = df["timestamp"].max()
    test_start = max_t - pd.Timedelta(days=days["test_days"])
    val_start = test_start - pd.Timedelta(days=days["val_days"])

    def label(ts):
        if ts >= test_start:
            return "test"
        elif ts >= val_start:
            return "val"
        return "train"

    df["split"] = df["timestamp"].apply(label)
    return df


def temporal_split(impressions: pd.DataFrame) -> pd.DataFrame:
    parts = [_split_one(name, g) for name, g in impressions.groupby("dataset")]
    out = pd.concat(parts, ignore_index=True)
    print(out.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    return out