"""Holiday & calendar features. Uses the `holidays` package if available."""
from __future__ import annotations
import numpy as np
import pandas as pd


def holiday_flags(idx: pd.DatetimeIndex, country: str = "US",
                  windows: tuple[int, ...] = (1,)) -> pd.DataFrame:
    """Indicator + leading/trailing windows around holidays."""
    try:
        import holidays as _h
        cal = _h.country_holidays(country, years=range(idx.year.min(), idx.year.max() + 1))
    except Exception:
        return pd.DataFrame(index=idx)
    days = pd.Series(pd.to_datetime(list(cal.keys())))
    is_hol = idx.normalize().isin(days)
    df = pd.DataFrame({f"is_holiday_{country}": is_hol.astype(int)}, index=idx)
    for w in windows:
        df[f"pre_holiday_{country}_{w}"]  = pd.Series(is_hol, index=idx).shift(-w * (idx.freq.n if idx.freq else 1)).fillna(0).astype(int).values
        df[f"post_holiday_{country}_{w}"] = pd.Series(is_hol, index=idx).shift( w * (idx.freq.n if idx.freq else 1)).fillna(0).astype(int).values
    return df


def calendar_known_future(idx: pd.DatetimeIndex, country: str | None = None) -> pd.DataFrame:
    """All deterministic-in-time features — known at forecast time by definition."""
    df = pd.DataFrame(index=idx)
    df["hour"]       = idx.hour
    df["dow"]        = idx.dayofweek
    df["month"]      = idx.month
    df["weekofyear"] = idx.isocalendar().week.astype(int).values
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    for period, K in [(24, 3), (168, 3), (24 * 365, 2)]:
        t = np.arange(len(idx))
        for k in range(1, K + 1):
            df[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    if country:
        df = pd.concat([df, holiday_flags(idx, country=country)], axis=1)
    return df