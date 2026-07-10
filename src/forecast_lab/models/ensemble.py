"""Weighted-average ensemble of BaseModel instances.

Weights are either supplied explicitly or learned automatically from
the last holdout fold via one of three strategies:
  * ``mae``       — inverse-MAE weights (default)
  * ``winkler``   — inverse-Winkler weights (accuracy + calibration)
  * ``uniform``   — equal weight to every constituent

The ensemble is itself a `BaseModel` and slots directly into the
existing backtest runner and model registry.
"""
from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import pandas as pd

from .base import BaseModel, Forecast


WeightStrategy = Literal["mae", "winkler", "uniform"]


class EnsembleModel(BaseModel):
    """Weighted-average ensemble.

    Parameters
    ----------
    models          : list of fitted or unfitted BaseModel instances
    weights         : explicit weight vector (len == len(models)); if
                      None, weights are learned from a holdout split
                      using ``weight_strategy``
    weight_strategy : 'mae' | 'winkler' | 'uniform'
    holdout_frac    : fraction of training data used as holdout to
                      learn weights (ignored when weights are given)
    """

    produces_intervals = True

    def __init__(
        self,
        models: list[BaseModel],
        weights: list[float] | None = None,
        weight_strategy: WeightStrategy = "mae",
        holdout_frac: float = 0.1,
    ):
        self.models = models
        self._weights_init = weights
        self.weight_strategy = weight_strategy
        self.holdout_frac = holdout_frac
        names = "+".join(getattr(m, "name", type(m).__name__) for m in models)
        self.name = f"ensemble({names})"

    # ------------------------------------------------------------------
    def _learn_weights(self, y: pd.Series, horizon: int, alpha: float) -> np.ndarray:
        """Fit each model on a training prefix; evaluate on the holdout tail."""
        n = len(y)
        split = max(horizon, int(n * (1 - self.holdout_frac)))
        test_ho = y.iloc[split : split + horizon]
        h = len(test_ho)
        if h == 0:
            return np.ones(len(self.models_)) / len(self.models_)

        scores = []
        for m in self.models_:
            fc = m.predict(h, alpha=alpha)
            if self.weight_strategy == "winkler" and fc.lo is not None:
                from ..metrics import winkler_score
                score = winkler_score(test_ho.values, fc.lo, fc.hi, alpha)
            else:
                from ..metrics import mae as _mae
                score = _mae(test_ho.values, fc.mean)
            scores.append(max(score, 1e-9))

        inv = np.array([1.0 / s for s in scores])
        return inv / inv.sum()

    def fit(self, y: pd.Series, cov=None) -> "EnsembleModel":
        import inspect

        horizon = max(1, int(len(y) * self.holdout_frac))
        alpha_default = 0.1

        self.models_ = []
        for m in self.models:
            mc = copy.deepcopy(m)
            sig = inspect.signature(mc.fit)
            if "cov" in sig.parameters and cov is not None:
                mc.fit(y, cov=cov)
            else:
                mc.fit(y)
            self.models_.append(mc)

        if self._weights_init is not None:
            w = np.asarray(self._weights_init, dtype=float)
        elif self.weight_strategy == "uniform":
            w = np.ones(len(self.models_)) / len(self.models_)
        else:
            w = self._learn_weights(y, horizon, alpha_default)

        self.weights_ = w / w.sum()
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        import inspect

        means, los, his, all_samples = [], [], [], []
        for m, w in zip(self.models_, self.weights_):
            sig = inspect.signature(m.predict)
            fc: Forecast = (
                m.predict(horizon, alpha=alpha, cov=cov)
                if "cov" in sig.parameters and cov is not None
                else m.predict(horizon, alpha=alpha)
            )
            means.append(fc.mean * w)
            if fc.lo is not None:
                los.append(fc.lo * w)
                his.append(fc.hi * w)
            if fc.samples is not None:
                all_samples.append(fc.samples)

        mean = np.sum(means, axis=0)
        lo = np.sum(los, axis=0) if los else None
        hi = np.sum(his, axis=0) if his else None

        samples = None
        if all_samples:
            # stack samples from all constituents (H, S*K)
            min_s = min(s.shape[1] for s in all_samples)
            samples = np.concatenate([s[:, :min_s] for s in all_samples], axis=1)

        return Forecast(mean=mean, lo=lo, hi=hi, samples=samples,
                        meta={"weights": self.weights_.tolist(),
                              "constituents": [getattr(m, "name", "?") for m in self.models_]})
