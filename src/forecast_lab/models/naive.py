"""SeasonalNaive — the baseline that most forecasting papers conveniently forget.

Forecast at step h = value observed `season_length` steps ago, tiled across
the horizon. Prediction intervals use the empirical std of in-sample seasonal
differences with random-walk-style √h scaling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseModel, Forecast


class SeasonalNaive(BaseModel):
    name = "seasonal_naive"
    produces_intervals = True
    accepts_covariates = False

    def __init__(self, season_length: int):
        self.m = int(season_length)

    def fit(self, y: pd.Series, cov=None) -> "SeasonalNaive":
        self.y_ = y.values.astype(float)
        if len(self.y_) <= self.m:
            raise ValueError(
                f"SeasonalNaive needs more than {self.m} observations to fit; "
                f"got {len(self.y_)}.")
        diffs = self.y_[self.m:] - self.y_[:-self.m]
        self.sigma_ = float(np.std(diffs))
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        from scipy.stats import norm
        last_season = self.y_[-self.m:]
        reps = int(np.ceil(horizon / self.m))
        mean = np.tile(last_season, reps)[:horizon]

        z = norm.ppf(1 - alpha / 2)
        # PI radius grows with √(ceil(h/m)) — analogous to random-walk scaling
        h_steps = np.arange(1, horizon + 1)
        band = z * self.sigma_ * np.sqrt(np.ceil(h_steps / self.m))

        return Forecast(mean=mean, lo=mean - band, hi=mean + band)