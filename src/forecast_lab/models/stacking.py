"""Stacking meta-learner for forecast combination.

Stage 1 — each constituent model is fitted on a rolling-CV split and
generates out-of-fold predictions.
Stage 2 — a lightweight meta-learner (Ridge by default, or LightGBM
for non-linear combination) is trained on those OOF predictions to
minimise the final target.

The stacked model implements BaseModel so it works as a drop-in inside
the existing backtest runner.
"""
from __future__ import annotations

import copy
import inspect
from typing import Literal

import numpy as np
import pandas as pd

from .base import BaseModel, Forecast

MetaLearner = Literal["ridge", "lasso", "lgbm", "mlp"]


class StackingModel(BaseModel):
    """Stacking ensemble with a fitted meta-learner.

    Parameters
    ----------
    models       : list of level-0 BaseModel instances
    meta         : meta-learner type — 'ridge' | 'lasso' | 'lgbm' | 'mlp'
    n_folds      : number of rolling CV folds for OOF generation
    alpha_meta   : Ridge / Lasso regularisation strength
    """

    produces_intervals = True

    def __init__(
        self,
        models: list[BaseModel],
        meta: MetaLearner = "ridge",
        n_folds: int = 3,
        alpha_meta: float = 1.0,
    ):
        self.models = models
        self.meta = meta
        self.n_folds = n_folds
        self.alpha_meta = alpha_meta
        names = "+".join(getattr(m, "name", type(m).__name__) for m in models)
        self.name = f"stacking({names})"

    # ------------------------------------------------------------------
    def _build_meta(self):
        if self.meta == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=self.alpha_meta, fit_intercept=True)
        if self.meta == "lasso":
            from sklearn.linear_model import Lasso
            return Lasso(alpha=self.alpha_meta, fit_intercept=True, max_iter=2000)
        if self.meta == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     num_leaves=15, verbose=-1)
        if self.meta == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500,
                                learning_rate_init=1e-3, random_state=0)
        raise ValueError(f"unknown meta learner: {self.meta}")

    def _oof_preds(self, y: pd.Series, horizon: int) -> np.ndarray:
        """Generate out-of-fold point predictions for each constituent."""
        n = len(y)
        fold_size = n // (self.n_folds + 1)
        oof = np.full((n, len(self.models)), np.nan)

        for fold in range(self.n_folds):
            te = (fold + 1) * fold_size
            ee = min(te + horizon, n)
            if ee <= te:
                continue
            train_y = y.iloc[:te]
            for j, m in enumerate(self.models):
                mc = copy.deepcopy(m)
                sig = inspect.signature(mc.fit)
                mc.fit(train_y)
                fc = mc.predict(ee - te)
                oof[te:ee, j] = fc.mean[: (ee - te)]

        return oof

    def fit(self, y: pd.Series, cov=None) -> "StackingModel":
        horizon = max(1, len(y) // (self.n_folds + 2))

        # 1. OOF predictions
        oof = self._oof_preds(y, horizon)
        valid = ~np.isnan(oof).any(axis=1)
        X_meta = oof[valid]
        y_meta = y.values[valid]

        # 2. Fit meta-learner
        self.meta_ = self._build_meta()
        self.meta_.fit(X_meta, y_meta)

        # 3. Refit all constituents on full training data
        self.models_ = []
        for m in self.models:
            mc = copy.deepcopy(m)
            sig = inspect.signature(mc.fit)
            if "cov" in sig.parameters and cov is not None:
                mc.fit(y, cov=cov)
            else:
                mc.fit(y)
            self.models_.append(mc)

        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        base_preds = []
        los, his, all_samples = [], [], []

        for m in self.models_:
            sig = inspect.signature(m.predict)
            fc: Forecast = (
                m.predict(horizon, alpha=alpha, cov=cov)
                if "cov" in sig.parameters and cov is not None
                else m.predict(horizon, alpha=alpha)
            )
            base_preds.append(fc.mean)
            if fc.lo is not None:
                los.append(fc.lo)
                his.append(fc.hi)
            if fc.samples is not None:
                all_samples.append(fc.samples)

        # Apply meta-learner step-wise
        X = np.column_stack(base_preds)   # (H, K)
        mean = self.meta_.predict(X)

        lo, hi = None, None
        if los:
            lo = self.meta_.predict(np.column_stack(los))
            hi = self.meta_.predict(np.column_stack(his))
            # Ensure lo <= mean <= hi ordering
            lo = np.minimum(lo, mean)
            hi = np.maximum(hi, mean)

        samples = None
        if all_samples:
            min_s = min(s.shape[1] for s in all_samples)
            samples = np.stack([s[:, :min_s] for s in all_samples], axis=2).mean(axis=2)

        return Forecast(mean=mean, lo=lo, hi=hi, samples=samples,
                        meta={"meta_learner": self.meta, "n_constituents": len(self.models_)})
