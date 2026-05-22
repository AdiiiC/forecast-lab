"""Feature engineering with strict no-leakage and exogenous covariate support."""
from __future__ import annotations
import pandas as pd
from .calendars import calendar_known_future
from .covariates import Covariates, merge_covariate_matrix


def lag_features(y: pd.Series, lags: list[int]) -> pd.DataFrame:
    return pd.concat({f"lag_{L}": y.shift(L) for L in lags}, axis=1)


def build_matrix(y: pd.Series, lags: list[int],
                 cov: Covariates | None = None,
                 observed_lags: list[int] | None = None,
                 country: str | None = None
                 ) -> tuple[pd.DataFrame, pd.Series]:
    cov = cov or Covariates()
    observed_lags = observed_lags or [1, 24, 168]
    parts = [lag_features(y, lags), calendar_known_future(y.index, country=country)]
    if cov.has_any:
        parts.append(merge_covariate_matrix(y.index, cov, observed_lags))
    X = pd.concat(parts, axis=1).dropna()
    return X, y.loc[X.index]