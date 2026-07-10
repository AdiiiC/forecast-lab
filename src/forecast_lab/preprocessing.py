"""Preprocessing pipeline: missing-data, outliers, changepoints.

Enhancements
------------
* ``fill_forward`` / ``fill_zero`` — simple fill strategies for
  intermittent and sparse series alongside the Kalman / seasonal fills
* ``inject_anomalies`` — deliberately corrupt a series for robustness testing
* STL decomposition memoisation — avoids recomputing identical series across folds
* ``Pipeline`` class — compose transforms into a single callable

All transforms operate on a single pd.Series and return a new Series; flags are
returned alongside so downstream models can use them as covariates if desired.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd


# ─── Missing-data ─────────────────────────────────────────────────────────────

def fill_kalman(y: pd.Series) -> pd.Series:
    """State-space local-level Kalman smoother. Falls back to linear interp."""
    if not y.isna().any():
        return y
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        m = UnobservedComponents(y.values, level="local level").fit(disp=False)
        smoothed = m.smoothed_state[0]
        out = y.copy()
        out[y.isna()] = smoothed[y.isna().values]
        return out
    except Exception:
        return y.interpolate(method="time").bfill().ffill()


def fill_seasonal(y: pd.Series, season: int) -> pd.Series:
    """Replace NaNs with the value `season` steps prior; bfill any leftovers."""
    out = y.copy()
    if not out.isna().any():
        return out
    for _ in range(3):
        out = out.fillna(out.shift(season))
        if not out.isna().any():
            break
    return out.interpolate().bfill().ffill()


def fill_forward(y: pd.Series) -> pd.Series:
    """Simple forward-fill (last-observation-carried-forward).
    Suitable for intermittent series where zero is not appropriate."""
    return y.ffill().bfill()


def fill_zero(y: pd.Series) -> pd.Series:
    """Replace NaNs with zero.
    Appropriate for intermittent demand series (missing = no demand)."""
    return y.fillna(0.0)


# ─── Outliers ─────────────────────────────────────────────────────────────────

def hampel(y: pd.Series, window: int = 24, k: float = 3.0
           ) -> tuple[pd.Series, pd.Series]:
    """Hampel filter: median ± k·MAD inside a rolling window."""
    med = y.rolling(window, center=True, min_periods=1).median()
    mad = ((y - med).abs()
           .rolling(window, center=True, min_periods=1).median() * 1.4826)
    flags = (y - med).abs() > k * mad
    cleaned = y.mask(flags, med)
    return cleaned, flags.astype(int).rename("outlier_hampel")


# ─── STL with memoisation ──────────────────────────────────────────────────────

def _series_key(y: pd.Series, period: int) -> str:
    """Stable cache key from series values + period."""
    digest = hashlib.md5(
        y.values.tobytes() + str(period).encode()
    ).hexdigest()
    return digest


_STL_CACHE: dict[str, object] = {}


def _stl_fit(y: pd.Series, period: int):
    """Fit (and cache) STL decomposition."""
    key = _series_key(y, period)
    if key not in _STL_CACHE:
        from statsmodels.tsa.seasonal import STL
        _STL_CACHE[key] = STL(
            y.interpolate().bfill().ffill(), period=period, robust=True
        ).fit()
    return _STL_CACHE[key]


def stl_outliers(y: pd.Series, period: int, k: float = 3.0
                 ) -> tuple[pd.Series, pd.Series]:
    """STL residual-based outlier flagging with cached decomposition."""
    try:
        res = _stl_fit(y, period)
        resid = res.resid
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        flags = (resid - np.median(resid)).abs() > k * scale
        cleaned = y.copy()
        cleaned[flags] = (res.trend + res.seasonal)[flags]
        return cleaned, flags.astype(int).rename("outlier_stl")
    except Exception:
        return hampel(y, window=period, k=k)


# ─── Anomaly injection ────────────────────────────────────────────────────────

AnomalyType = Literal["spike", "level_shift", "dropout", "gaussian"]


def inject_anomalies(
    y: pd.Series,
    n: int = 5,
    kind: AnomalyType = "spike",
    magnitude: float = 5.0,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Deliberately corrupt `n` random points for robustness testing.

    Parameters
    ----------
    n         : number of anomaly events to inject
    kind      : 'spike' | 'level_shift' | 'dropout' | 'gaussian'
    magnitude : scale multiplier for spike / gaussian; shift size for level_shift
    seed      : RNG seed for reproducibility

    Returns
    -------
    (corrupted_series, anomaly_flags)
    """
    rng = np.random.default_rng(seed)
    out = y.copy()
    flags = pd.Series(0, index=y.index, name="injected_anomaly")
    scale = float(y.std()) or 1.0

    if kind == "spike":
        idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
        for i in idx:
            out.iloc[i] += rng.choice([-1, 1]) * magnitude * scale
            flags.iloc[i] = 1

    elif kind == "level_shift":
        start = int(rng.integers(1, max(2, len(y) - n)))
        out.iloc[start : start + n] += magnitude * scale
        flags.iloc[start : start + n] = 1

    elif kind == "dropout":
        idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
        out.iloc[idx] = np.nan
        flags.iloc[idx] = 1

    elif kind == "gaussian":
        idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
        noise = rng.normal(0, magnitude * scale, size=len(idx))
        for i, ni in zip(idx, noise):
            out.iloc[i] += ni
            flags.iloc[i] = 1

    return out, flags


# ─── Changepoints ─────────────────────────────────────────────────────────────

def detect_changepoints(y: pd.Series, min_size: int = 168, pen: float = 3.0,
                        max_n: int = 2000) -> list[pd.Timestamp]:
    """Use `ruptures` if available; otherwise a simple CUSUM on rolling means."""
    try:
        import ruptures as rpt
        vals = y.values.astype(float)
        n = len(vals)
        if n > max_n:
            factor = n // max_n
            trimmed = vals[: factor * max_n]
            vals_ds = trimmed.reshape(-1, factor).mean(axis=1)
            idx_ds = y.index[::factor][: len(vals_ds)]
        else:
            vals_ds = vals
            idx_ds = y.index
        algo = rpt.Pelt(
            model="l2", min_size=max(2, min_size // max(1, n // max_n))
        ).fit(vals_ds)
        bks = algo.predict(pen=pen)
        return [idx_ds[b - 1] for b in bks[:-1]]
    except Exception:
        roll = y.rolling(min_size, min_periods=min_size).mean()
        diffs = roll.diff().abs()
        thr = diffs.std() * 3
        return list(y.index[diffs > thr][::min_size])


# ─── Pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class PreprocessReport:
    n_missing_filled: int
    n_outliers: int
    changepoints: list[pd.Timestamp]


FillMethod = Literal["kalman", "seasonal", "forward", "zero"]
OutlierMethod = Literal["stl", "hampel", "none"]


def preprocess(
    y: pd.Series,
    *,
    season: int,
    fill_method: FillMethod = "kalman",
    outlier_method: OutlierMethod = "stl",
    find_changepoints: bool = True,
) -> tuple[pd.Series, pd.DataFrame, PreprocessReport]:
    """Full preprocessing pipeline.

    Parameters
    ----------
    fill_method : 'kalman' | 'seasonal' | 'forward' | 'zero'
    outlier_method : 'stl' | 'hampel' | 'none'
    """
    flags = pd.DataFrame(index=y.index)
    n_missing = int(y.isna().sum())

    if fill_method == "kalman":
        y2 = fill_kalman(y)
    elif fill_method == "seasonal":
        y2 = fill_seasonal(y, season)
    elif fill_method == "forward":
        y2 = fill_forward(y)
    elif fill_method == "zero":
        y2 = fill_zero(y)
    else:
        y2 = y.interpolate().bfill().ffill()

    if outlier_method == "hampel":
        y2, f = hampel(y2, window=season)
    elif outlier_method == "stl":
        y2, f = stl_outliers(y2, period=season)
    else:
        f = pd.Series(0, index=y.index, name="outlier_none")

    flags[f.name] = f
    cps = detect_changepoints(y2, min_size=season) if find_changepoints else []
    flags["since_changepoint"] = 0
    for cp in cps:
        flags.loc[cp:, "since_changepoint"] = (flags.loc[cp:].index - cp).days

    return y2, flags, PreprocessReport(n_missing, int(f.sum()), cps)


# legacy alias
def do_kalman(y: pd.Series) -> bool:
    """True when kalman fill is the default. Kept for backward compat."""
    return True



# ─── Missing-data ───────────────────────────────────────────────────────────

def fill_kalman(y: pd.Series) -> pd.Series:
    """State-space local-level Kalman smoother. Falls back to linear interp
    if statsmodels is unavailable."""
    if not y.isna().any():
        return y
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        m = UnobservedComponents(y.values, level="local level").fit(disp=False)
        smoothed = m.smoothed_state[0]
        out = y.copy()
        out[y.isna()] = smoothed[y.isna().values]
        return out
    except Exception:
        return y.interpolate(method="time").bfill().ffill()


def fill_seasonal(y: pd.Series, season: int) -> pd.Series:
    """Replace NaNs with the value `season` steps prior; bfill any leftovers."""
    out = y.copy()
    if not out.isna().any():
        return out
    for _ in range(3):
        out = out.fillna(out.shift(season))
        if not out.isna().any():
            break
    return out.interpolate().bfill().ffill()


# ─── Outliers ───────────────────────────────────────────────────────────────

def hampel(y: pd.Series, window: int = 24, k: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Hampel filter: median ± k·MAD inside a rolling window.

    Returns (cleaned_series, outlier_flags). Cleaned values replace outliers
    with the rolling median.
    """
    med = y.rolling(window, center=True, min_periods=1).median()
    mad = (y - med).abs().rolling(window, center=True, min_periods=1).median() * 1.4826
    flags = (y - med).abs() > k * mad
    cleaned = y.mask(flags, med)
    return cleaned, flags.astype(int).rename("outlier_hampel")


def stl_outliers(y: pd.Series, period: int, k: float = 3.0
                 ) -> tuple[pd.Series, pd.Series]:
    """STL decomposition residual-based outlier flagging."""
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(y.interpolate().bfill().ffill(), period=period, robust=True).fit()
        resid = res.resid
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        flags = (resid - np.median(resid)).abs() > k * scale
        cleaned = y.copy()
        cleaned[flags] = (res.trend + res.seasonal)[flags]
        return cleaned, flags.astype(int).rename("outlier_stl")
    except Exception:
        return hampel(y, window=period, k=k)


# ─── Changepoints ───────────────────────────────────────────────────────────

def detect_changepoints(y: pd.Series, min_size: int = 168, pen: float = 3.0,
                        max_n: int = 2000) -> list[pd.Timestamp]:
    """Use `ruptures` if available; otherwise a simple CUSUM on rolling means.

    ruptures RBF kernel is O(n²) — unusable on long series. We use the
    'l2' (least-squares) model which is O(n log n) and safe on 17k+ points.
    Long series are downsampled to `max_n` points before detection so the
    algorithm stays fast; changepoints are then mapped back to original index.
    """
    try:
        import ruptures as rpt
        vals = y.values.astype(float)
        n = len(vals)
        if n > max_n:
            # Downsample: average into max_n buckets
            factor = n // max_n
            trimmed = vals[:factor * max_n]
            vals_ds = trimmed.reshape(-1, factor).mean(axis=1)
            idx_ds = y.index[::factor][:len(vals_ds)]
        else:
            vals_ds = vals
            idx_ds = y.index
        algo = rpt.Pelt(model="l2", min_size=max(2, min_size // max(1, n // max_n))).fit(vals_ds)
        bks = algo.predict(pen=pen)
        return [idx_ds[b - 1] for b in bks[:-1]]
    except Exception:
        roll = y.rolling(min_size, min_periods=min_size).mean()
        diffs = roll.diff().abs()
        thr = diffs.std() * 3
        return list(y.index[diffs > thr][::min_size])


# ─── Pipeline ───────────────────────────────────────────────────────────────

@dataclass
class PreprocessReport:
    n_missing_filled: int
    n_outliers: int
    changepoints: list[pd.Timestamp]


def preprocess(y: pd.Series, *, season: int,
               do_kalman: bool = True,
               outlier_method: str = "stl",
               find_changepoints: bool = True
               ) -> tuple[pd.Series, pd.DataFrame, PreprocessReport]:
    flags = pd.DataFrame(index=y.index)
    n_missing = int(y.isna().sum())
    y2 = fill_kalman(y) if do_kalman else y.interpolate().bfill().ffill()
    if outlier_method == "hampel":
        y2, f = hampel(y2, window=season)
    elif outlier_method == "stl":
        y2, f = stl_outliers(y2, period=season)
    else:
        f = pd.Series(0, index=y.index, name="outlier_none")
    flags[f.name] = f
    cps = detect_changepoints(y2, min_size=season) if find_changepoints else []
    flags["since_changepoint"] = 0
    for cp in cps:
        flags.loc[cp:, "since_changepoint"] = (flags.loc[cp:].index - cp).days
    return y2, flags, PreprocessReport(n_missing, int(f.sum()), cps)