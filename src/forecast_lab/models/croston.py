"""Intermittent-demand forecasters: Croston, SBA, TSB, ADIDA.

Used when demand is sparse / lumpy (many zeros), e.g. spare-parts, slow-moving
SKUs. Standard MLE-style models (ARIMA/Prophet/DL) over-smooth and fail to
predict the *zero rate*.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseModel, Forecast


def _croston_smooth(y: np.ndarray, alpha: float, variant: str):
    """Returns (level, interval) Croston-family smoothing series.

    variant ∈ {'classic', 'sba'} — SBA (Syntetos-Boylan) applies the bias
    correction factor 1 - α/2 to the level.
    """
    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return 0.0, np.inf
    # level: smoothed non-zero demand size
    Z = y[nz[0]]
    P = nz[0] + 1.0          # initial inter-arrival
    last = nz[0]
    for k, t in enumerate(nz[1:], start=1):
        Z = alpha * y[t] + (1 - alpha) * Z
        P = alpha * (t - last) + (1 - alpha) * P
        last = t
    if variant == "sba":
        Z *= (1 - alpha / 2)
    return Z, max(P, 1.0)


class CrostonModel(BaseModel):
    """Classic Croston / SBA constant forecast for intermittent demand."""
    name = "croston"
    produces_intervals = True

    def __init__(self, alpha: float = 0.1, variant: str = "sba"):
        self.alpha = alpha
        self.variant = variant

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()
        self.Z_, self.P_ = _croston_smooth(y.values.astype(float),
                                           self.alpha, self.variant)
        # empirical residuals on a one-step-ahead replay for PIs
        f = self.Z_ / self.P_
        self.sigma_ = float(np.std(y.values - f))
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        f = np.full(horizon, self.Z_ / self.P_)
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        band = z * self.sigma_ * np.sqrt(np.arange(1, horizon + 1))
        return Forecast(mean=f, lo=np.maximum(f - band, 0.0), hi=f + band)


class TSBModel(BaseModel):
    """Teunter-Syntetos-Babai: smooths probability-of-demand and size separately.

    Handles obsolescence (long runs of zeros decrease the demand probability).
    """
    name = "tsb"
    produces_intervals = True

    def __init__(self, alpha_p: float = 0.1, alpha_z: float = 0.1):
        self.alpha_p, self.alpha_z = alpha_p, alpha_z

    def fit(self, y: pd.Series, cov=None):
        y = y.values.astype(float)
        p, z = 0.0, 0.0
        nz_seen = False
        for v in y:
            indicator = float(v > 0)
            p = self.alpha_p * indicator + (1 - self.alpha_p) * (p if nz_seen else indicator)
            if v > 0:
                z = self.alpha_z * v + (1 - self.alpha_z) * (z if nz_seen else v)
                nz_seen = True
        self.f_ = p * z
        self.sigma_ = float(np.std(y - self.f_))
        self.y_ = y
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        f = np.full(horizon, self.f_)
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        band = z * self.sigma_ * np.sqrt(np.arange(1, horizon + 1))
        return Forecast(mean=f, lo=np.maximum(f - band, 0.0), hi=f + band)


class ADIDAModel(BaseModel):
    """Aggregate-Disaggregate Intermittent Demand Approach.

    Temporally aggregate to a less intermittent grain, forecast there with a
    simple smoother, then equal-disaggregate back to the original frequency.
    """
    name = "adida"
    produces_intervals = True

    def __init__(self, agg: int = 7, base: str = "ses", alpha: float = 0.2):
        self.agg, self.base, self.alpha = agg, base, alpha

    def fit(self, y: pd.Series, cov=None):
        v = y.values.astype(float)
        n_pad = (-len(v)) % self.agg
        v = np.concatenate([np.zeros(n_pad), v])
        agg = v.reshape(-1, self.agg).sum(axis=1)
        # simple exponential smoothing on the aggregated series
        s = agg[0]
        for a in agg[1:]:
            s = self.alpha * a + (1 - self.alpha) * s
        self.level_per_step_ = s / self.agg
        self.sigma_ = float(np.std(y.values - self.level_per_step_))
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        f = np.full(horizon, self.level_per_step_)
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        band = z * self.sigma_ * np.sqrt(np.arange(1, horizon + 1))
        return Forecast(mean=f, lo=np.maximum(f - band, 0.0), hi=f + band)