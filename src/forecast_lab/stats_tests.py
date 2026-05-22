"""Forecast-comparison statistical tests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class DMResult:
    statistic: float
    p_value: float
    h: int
    loss: str
    better: str   # which model has lower mean loss


def _loss(e: np.ndarray, kind: str):
    if kind == "mse":  return e ** 2
    if kind == "mae":  return np.abs(e)
    raise ValueError(kind)


def diebold_mariano(y, f1, f2, h: int = 1, loss: str = "mae",
                    power: bool = False) -> DMResult:
    """Diebold–Mariano test (Harvey-corrected for small samples).

    H0: equal predictive accuracy between f1 and f2.
    Returns a two-sided test. Negative statistic ⇒ f1 better.
    """
    y, f1, f2 = map(np.asarray, (y, f1, f2))
    e1, e2 = y - f1, y - f2
    d = _loss(e1, loss) - _loss(e2, loss)
    n = len(d)
    mean_d = d.mean()
    # Newey-West long-run variance with bandwidth h-1
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        var_d += 2 * cov
    var_d = max(var_d, 1e-12)
    dm = mean_d / np.sqrt(var_d / n)
    # Harvey-Leybourne-Newbold correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corr = dm * hln
    p = 2 * (1 - stats.t.cdf(np.abs(dm_corr), df=n - 1))
    return DMResult(statistic=float(dm_corr), p_value=float(p), h=h, loss=loss,
                    better=("f1" if mean_d < 0 else "f2"))


def giacomini_white(y, f1, f2, h: int = 1, loss: str = "mae",
                    test_fn: np.ndarray | None = None) -> DMResult:
    """Giacomini–White conditional predictive ability test.

    `test_fn` is the conditioning information (n, k). Defaults to a constant +
    lagged loss differential (unconditional GW reduces to a Wald form).
    """
    y, f1, f2 = map(np.asarray, (y, f1, f2))
    d = _loss(y - f1, loss) - _loss(y - f2, loss)
    n = len(d)
    if test_fn is None:
        test_fn = np.column_stack([np.ones(n), np.r_[0, d[:-1]]])
    z = test_fn * d[:, None]
    z_bar = z.mean(axis=0)
    # HAC covariance with bandwidth h-1
    omega = z.T @ z / n
    for k in range(1, h):
        g = z[k:].T @ z[:-k] / n
        omega += (g + g.T)
    stat = n * z_bar @ np.linalg.pinv(omega) @ z_bar
    p = 1 - stats.chi2.cdf(stat, df=test_fn.shape[1])
    return DMResult(statistic=float(stat), p_value=float(p), h=h, loss=loss,
                    better=("f1" if d.mean() < 0 else "f2"))