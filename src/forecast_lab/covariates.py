"""First-class exogenous covariates.

Two kinds, with strict semantics:
  * known_future : values are known at forecast time for every step in horizon
                   (calendar features, scheduled prices, planned promos, holidays,
                    weather *forecasts* if you trust them).
  * observed     : values are only known up to t; must be lagged to be safe
                   (realized weather, realized prices, observed competitor data).

The framework guarantees:
  - Observed covariates are lagged before being shown to any model.
  - Known-future covariates are passed unlagged for the forecast horizon.
  - Index alignment & missing-data handling is centralized here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Covariates:
    known_future: pd.DataFrame = field(default_factory=pd.DataFrame)
    observed: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_any(self) -> bool:
        return not (self.known_future.empty and self.observed.empty)

    def align(self, idx: pd.DatetimeIndex) -> "Covariates":
        kf = self.known_future.reindex(idx) if not self.known_future.empty else self.known_future
        ob = self.observed.reindex(idx)     if not self.observed.empty     else self.observed
        return Covariates(known_future=kf, observed=ob)

    def slice(self, start, end) -> "Covariates":
        return Covariates(
            known_future=self.known_future.loc[start:end] if not self.known_future.empty else self.known_future,
            observed=self.observed.loc[start:end]         if not self.observed.empty     else self.observed,
        )

    def to_lagged_observed(self, lags: list[int]) -> pd.DataFrame:
        if self.observed.empty:
            return pd.DataFrame(index=self.observed.index)
        out = {}
        for col in self.observed.columns:
            for L in lags:
                out[f"{col}_lag{L}"] = self.observed[col].shift(L)
        return pd.DataFrame(out)


def merge_covariate_matrix(idx: pd.DatetimeIndex, cov: Covariates,
                           observed_lags: list[int]) -> pd.DataFrame:
    """Return a (n × k) frame safe to feed any tabular model."""
    parts = []
    if not cov.known_future.empty:
        parts.append(cov.known_future.reindex(idx))
    if not cov.observed.empty:
        parts.append(cov.to_lagged_observed(observed_lags).reindex(idx))
    if not parts:
        return pd.DataFrame(index=idx)
    return pd.concat(parts, axis=1)