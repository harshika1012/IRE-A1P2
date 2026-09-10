"""Bootstrap confidence intervals for any per-impression metric list."""
import numpy as np


def bootstrap_ci(values, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    """values: list/array of per-impression metric values (already filtered
    of Nones). Returns (mean, lower, upper) for the given CI level."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)
    return float(values.mean()), float(lower), float(upper)