from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class Forecast:
    mean: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    samples: np.ndarray | None = None       # (H, S) when available
    quantiles: np.ndarray | None = None     # (H, K)
    q_levels: np.ndarray | None = None      # (K,)
    dist: Any = None                        # PredictiveDistribution if any
    meta: dict = field(default_factory=dict)


class BaseModel:
    name: str = "base"
    produces_intervals: bool = False
    produces_distribution: bool = False
    accepts_covariates: bool = False

    def fit(self, y: pd.Series, cov=None) -> "BaseModel":
        raise NotImplementedError

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        raise NotImplementedError