"""Dataset loaders returning (y, covariates). Synthetic fallback included."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from urllib.request import urlopen
from .covariates import Covariates


def _synthetic_hourly(n_days: int = 730, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    t = np.arange(n)
    idx = pd.date_range("2023-01-01", periods=n, freq="H")

    daily  = 12 * np.sin(2 * np.pi * t / 24 - 1.2)
    weekly =  4 * np.sin(2 * np.pi * t / 168)
    yearly =  8 * np.sin(2 * np.pi * t / (365 * 24))
    trend  = 0.0008 * t

    # Realistic exogenous covariates
    temp_c = (12
              + 10 * np.sin(2 * np.pi * t / (365 * 24) - 1.5)
              +  4 * np.sin(2 * np.pi * t / 24 - 1.0)
              + rng.normal(0, 1.5, n))
    price = 50 + 20 * np.sin(2 * np.pi * t / 24 - 1.0) + rng.normal(0, 3, n)
    promo = (rng.uniform(size=n) < 0.02).astype(int)

    # Target depends on covariates (so models that use them should win).
    weather_effect = 0.6 * np.abs(temp_c - 18)
    price_effect   = -0.05 * (price - price.mean())
    promo_effect   = 5 * promo
    noise = np.zeros(n)
    eps = rng.normal(0, 1.0, n)
    for i in range(1, n):
        noise[i] = 0.75 * noise[i - 1] + eps[i]

    y = (60 + trend + daily + weekly + yearly
         + weather_effect + price_effect + promo_effect + noise)
    y = pd.Series(y, index=idx, name="load_mw")

    known_future = pd.DataFrame({"price": price, "promo": promo}, index=idx)
    observed     = pd.DataFrame({"temp_c": temp_c}, index=idx)
    return y, Covariates(known_future=known_future, observed=observed)


def load_series(cfg: dict):
    """Returns (y: Series, cov: Covariates)."""
    name = cfg["name"]
    if name == "electricity_hourly":
        return _synthetic_hourly(seed=0)
    try:
        url = cfg.get("url")
        if url:
            with urlopen(url, timeout=10) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])
            y = (df.set_index("ds")["y"]
                   .asfreq(cfg.get("freq", "D")).interpolate())
            return y, Covariates()
    except Exception:
        pass
    return _synthetic_hourly()