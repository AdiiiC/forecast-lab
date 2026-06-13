"""LightGBM with covariates: lags + calendar + holidays + exogenous regressors,
plus quantile heads for native PIs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from ..features import build_matrix
from ..calendars import calendar_known_future
from ..covariates import Covariates, merge_covariate_matrix
from .base import BaseModel, Forecast


class LightGBMModel(BaseModel):
    name = "lightgbm"
    produces_intervals = True
    accepts_covariates = True

    def __init__(self, lags, n_estimators=800, learning_rate=0.05,
                 num_leaves=64, min_data_in_leaf=20,
                 observed_lags=(1, 24, 168), country: str | None = None):
        self.lags = list(lags)
        self.observed_lags = list(observed_lags)
        self.country = country
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
            verbose=-1,
        )

    def _one(self, X, y, objective, alpha=None):
        p = dict(self.params, objective=objective)
        if alpha is not None:
            p["alpha"] = alpha
        m = lgb.LGBMRegressor(**p)
        m.fit(X, y)
        return m

    def fit(self, y: pd.Series, cov: Covariates | None = None):
        self.y_ = y.copy()
        self.cov_ = cov or Covariates()
        X, yy = build_matrix(y, self.lags, cov=self.cov_,
                             observed_lags=self.observed_lags,
                             country=self.country)
        self.feat_ = list(X.columns)
        self.mean_ = self._one(X, yy, "regression")
        self.qlo_  = self._one(X, yy, "quantile", alpha=0.05)
        self.qhi_  = self._one(X, yy, "quantile", alpha=0.95)
        return self

    def predict(self, horizon: int, alpha: float = 0.1,
                cov: Covariates | None = None) -> Forecast:
        history = self.y_.copy()
        freq = pd.infer_freq(history.index) or "H"
        fut_cov = cov or self.cov_
        means, los, his = [], [], []
        for _ in range(horizon):
            next_ts = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
            idx = pd.DatetimeIndex([next_ts])
            cal = calendar_known_future(idx, country=self.country)
            lag_row = {f"lag_{L}": history.iloc[-L] for L in self.lags}
            covs = (merge_covariate_matrix(idx, fut_cov, self.observed_lags)
                    if fut_cov.has_any else pd.DataFrame(index=idx))
            X_next = pd.concat(
                [pd.DataFrame([lag_row], index=idx), cal, covs], axis=1)
            X_next = (X_next.reindex(columns=self.feat_)
                            .fillna(method="ffill").fillna(0.0))
            mu = float(self.mean_.predict(X_next)[0])
            lo = float(self.qlo_.predict(X_next)[0])
            hi = float(self.qhi_.predict(X_next)[0])
            means.append(mu)
            los.append(lo)
            his.append(hi)
            history = pd.concat([history, pd.Series([mu], index=idx)])
        return Forecast(np.array(means), np.array(los), np.array(his))