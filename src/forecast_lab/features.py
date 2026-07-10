"""Feature engineering with strict no-leakage and exogenous covariate support.

Enhancements
------------
* ``target_encode``        — smooth target encoding for categorical covariates
* ``interaction_features`` — pairwise product / ratio / diff features
* ``shap_prune``           — drop low-importance features via SHAP values
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
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


# ─── Target encoding ──────────────────────────────────────────────────────────

def target_encode(
    X: pd.DataFrame,
    y: pd.Series,
    cols: Sequence[str],
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """Smooth target (mean) encoding for categorical columns.

    Shrinks small groups toward the global mean:
        encoded = (count * group_mean + smoothing * global_mean) / (count + smoothing)

    Call on the training set only; apply the returned mapping to held-out data.
    """
    X_enc = X.copy()
    global_mean = y.mean()
    for col in cols:
        if col not in X_enc.columns:
            continue
        stats = (
            pd.concat([X_enc[col].rename("cat"), y.rename("target")], axis=1)
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        stats["encoded"] = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        X_enc[col] = X_enc[col].map(stats["encoded"]).fillna(global_mean)
    return X_enc


# ─── Interaction features ─────────────────────────────────────────────────────

def interaction_features(
    X: pd.DataFrame,
    pairs: list[tuple[str, str]],
    ops: Sequence[str] = ("product", "ratio"),
) -> pd.DataFrame:
    """Append pairwise interaction features (product / ratio / diff).

    Parameters
    ----------
    pairs : list of (col_a, col_b) tuples
    ops   : subset of {'product', 'ratio', 'diff'}
    """
    out = X.copy()
    for a, b in pairs:
        if a not in out.columns or b not in out.columns:
            continue
        if "product" in ops:
            out[f"{a}_x_{b}"] = out[a] * out[b]
        if "ratio" in ops:
            denom = out[b].replace(0, np.nan)
            out[f"{a}_div_{b}"] = (out[a] / denom).fillna(0.0)
        if "diff" in ops:
            out[f"{a}_minus_{b}"] = out[a] - out[b]
    return out


# ─── SHAP-based pruning ───────────────────────────────────────────────────────

def shap_prune(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01,
    max_features: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features whose mean |SHAP| / max-mean-|SHAP| < threshold.

    Fits a LightGBM model internally; falls back to returning all features if
    shap or lightgbm is unavailable.
    """
    try:
        import lightgbm as lgb
        import shap  # type: ignore

        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                   num_leaves=31, verbose=-1)
        model.fit(X, y)
        shap_vals = shap.TreeExplainer(model).shap_values(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        norm = mean_abs / max(mean_abs.max(), 1e-12)

        keep_mask = norm >= threshold
        if max_features is not None:
            top_idx = np.argsort(norm)[::-1][:max_features]
            m2 = np.zeros(len(keep_mask), dtype=bool)
            m2[top_idx] = True
            keep_mask = keep_mask & m2

        kept = [c for c, k in zip(X.columns, keep_mask) if k] or list(X.columns)
        return X[kept], kept

    except ImportError:
        return X, list(X.columns)
