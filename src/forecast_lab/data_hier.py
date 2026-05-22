"""Synthetic grouped retail-style dataset for hierarchical experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .hierarchy import Hierarchy


def synthetic_retail(n_days: int = 365 * 2,
                     regions=("NA", "EU"),
                     stores_per_region: int = 3,
                     skus_per_store: int = 4,
                     seed: int = 0) -> Hierarchy:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    bottom = {}
    for r in regions:
        region_factor = 1.0 + 0.4 * rng.normal()
        for s in range(stores_per_region):
            store_factor = 1.0 + 0.2 * rng.normal()
            for k in range(skus_per_store):
                level = 20 + 8 * rng.normal()
                weekly = 6 * np.sin(2 * np.pi * np.arange(n_days) / 7
                                    + rng.uniform(0, 2 * np.pi))
                trend = 0.02 * np.arange(n_days) * rng.normal(scale=0.3)
                noise = rng.normal(0, 2.0, n_days)
                y = np.clip(level * region_factor * store_factor
                            + weekly + trend + noise, 0, None)
                bottom[(r, f"store{s}", f"sku{k}")] = pd.Series(y, index=idx)
    return Hierarchy(bottom=bottom,
                     levels=["region", "store", "sku"]).build()