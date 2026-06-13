"""Preprocessing pipeline: missing-data, outliers, changepoints.

All transforms operate on a single pd.Series and return a new Series; flags are
returned alongside so downstream models can use them as covariates if desired.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


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

def detect_changepoints(y: pd.Series, min_size: int = 168, pen: float = 10.0
                        ) -> list[pd.Timestamp]:
    """Use `ruptures` if available; otherwise a simple CUSUM on rolling means."""
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(y.values)
        bks = algo.predict(pen=pen)
        return [y.index[b - 1] for b in bks[:-1]]
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