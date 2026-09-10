"""Run on the cluster to find a sensible cold-start threshold per dataset."""
import pandas as pd
uf = pd.read_parquet("data/features/user_features.parquet")
for ds in ["mind", "ebnerd"]:
    sub = uf[uf["dataset"] == ds]["n_clicks"]
    print(f"\n{ds}: n={len(sub)}")
    print(sub.describe())
    print("quantiles:", sub.quantile([0.1, 0.25, 0.5]).to_dict())
