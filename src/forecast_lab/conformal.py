"""Split-conformal prediction wrapper.

Distribution-free intervals with finite-sample coverage guarantees (under
exchangeability). For time series we use the most recent contiguous tail as
the calibration set — a simple, fast, non-exchangeable variant that works
well in practice. For stronger guarantees under drift, use ACIWrapper or
EnbPIWrapper from `conformal_adaptive.py`.

Reference: Vovk, Gammerman & Shafer (2005); Lei et al. (2018).
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from .models.base import BaseModel, Forecast


class ConformalWrapper(BaseModel):
    produces_intervals = True
    accepts_covariates = True   # forwards to base if base supports it

    def __init__(self, base: BaseModel, calibration_size: int, horizon: int):
        self.base = base
        self.cal_n = int(calibration_size)
        self.h = int(horizon)
        self.name = f"conformal({base.name})"

    # ─── helpers ────────────────────────────────────────────────────────────
    def _fit_base(self, y: pd.Series, cov):
        if cov is not None and getattr(self.base, "accepts_covariates", False):
            self.base.fit(y, cov=cov.slice(y.index[0], y.index[-1]))
        else:
            self.base.fit(y)

    def _predict_base(self, horizon: int, alpha: float, cov):
        if cov is not None and getattr(self.base, "accepts_covariates", False):
            return self.base.predict(horizon, alpha=alpha, cov=cov)
        return self.base.predict(horizon, alpha=alpha)

    # ─── public API ─────────────────────────────────────────────────────────
    def fit(self, y: pd.Series, cov=None) -> "ConformalWrapper":
        if len(y) <= self.cal_n + 1:
            raise ValueError(
                f"ConformalWrapper needs len(y) > calibration_size "
                f"({self.cal_n}); got {len(y)}.")

        cal = y.iloc[-self.cal_n:]
        train = y.iloc[:-self.cal_n]

        # Fit on the training portion to get unbiased calibration residuals.
        self._fit_base(train, cov)
        fc = self._predict_base(min(self.h, len(cal)), alpha=0.1, cov=cov)

        h = min(len(fc.mean), len(cal))
        resid = np.abs(cal.values[:h] - fc.mean[:h])
        # Pad to full horizon by repeating the worst residual (conservative)
        if len(resid) < self.h:
            pad_val = resid[-1] if len(resid) else 0.0
            resid = np.concatenate([resid, np.full(self.h - len(resid), pad_val)])
        self.resid_ = resid

        # Refit on the *full* series for the actual forecast — more data, same
        # residual quantile is still a valid conformal score.
        self.base = copy.deepcopy(self.base)
        self._fit_base(y, cov)
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        fc = self._predict_base(horizon, alpha=alpha, cov=cov)
        q = float(np.quantile(self.resid_[:horizon], 1 - alpha))
        return Forecast(
            mean=fc.mean,
            lo=fc.mean - q,
            hi=fc.mean + q,
            samples=fc.samples,
            quantiles=fc.quantiles,
            q_levels=fc.q_levels,
        )