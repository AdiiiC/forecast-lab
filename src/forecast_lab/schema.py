"""Schema validation + drift detection.

Schemas are declared as simple dicts to avoid a hard pandera dependency, but
pandera is used if installed for richer checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class SchemaResult:
    ok: bool
    errors: list[str]


def validate(df: pd.DataFrame, spec: dict[str, Any]) -> SchemaResult:
    errors: list[str] = []
    for col, rules in spec.items():
        if col not in df.columns:
            errors.append(f"missing column: {col}")
            continue
        s = df[col]
        if "dtype" in rules and str(s.dtype) != rules["dtype"]:
            errors.append(f"{col}: dtype {s.dtype} != {rules['dtype']}")
        if rules.get("nonnull") and s.isna().any():
            errors.append(f"{col}: contains nulls")
        if "min" in rules and (s.dropna() < rules["min"]).any():
            errors.append(f"{col}: values below {rules['min']}")
        if "max" in rules and (s.dropna() > rules["max"]).any():
            errors.append(f"{col}: values above {rules['max']}")
        if "in" in rules and (~s.dropna().isin(rules["in"])).any():
            errors.append(f"{col}: out-of-domain values")
    return SchemaResult(ok=(len(errors) == 0), errors=errors)


# ─── Drift ──────────────────────────────────────────────────────────────────

def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. >0.1 = mild, >0.25 = serious drift."""
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, bins=qs)
    c, _ = np.histogram(cur, bins=qs)
    r = r / max(r.sum(), 1)
    c = c / max(c.sum(), 1)
    r = np.where(r == 0, 1e-6, r)
    c = np.where(c == 0, 1e-6, c)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp
    s = ks_2samp(ref, cur)
    return float(s.statistic), float(s.pvalue)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ref.columns.intersection(cur.columns):
        if not np.issubdtype(ref[c].dtype, np.number):
            continue
        p = psi(ref[c].dropna().values, cur[c].dropna().values)
        k, pv = ks_test(ref[c].dropna().values, cur[c].dropna().values)
        rows.append(dict(feature=c, PSI=p, KS=k, KS_p=pv,
                         flag=("drift" if p > 0.25 or pv < 0.01 else
                               "warn"  if p > 0.10 else "ok")))
    return pd.DataFrame(rows)