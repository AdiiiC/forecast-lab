"""Forecast metrics: point, interval, and loss-aware / business-cost scores."""
from __future__ import annotations
import numpy as np


# ─── Point-forecast metrics ─────────────────────────────────────────────────

def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    denom = (np.abs(y) + np.abs(p)) / 2.0
    return float(np.mean(np.where(denom == 0, 0.0,
                                  np.abs(y - p) / denom))) * 100


def mase(y_true, y_pred, y_train, season: int) -> float:
    """Mean Absolute Scaled Error.

    Scaled by the in-sample seasonal-naive error on `y_train`. MASE < 1
    means the model beats in-sample seasonal-naive on average.
    """
    y_train = np.asarray(y_train)
    d = np.mean(np.abs(y_train[season:] - y_train[:-season])) + 1e-12
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / d)


# ─── Interval / probabilistic basics ────────────────────────────────────────

def pinball(y, q_pred, q: float) -> float:
    """Pinball loss at quantile level q."""
    y, q_pred = np.asarray(y), np.asarray(q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, lo, hi) -> float:
    """Empirical fraction of observations inside [lo, hi]."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo, hi) -> float:
    """Mean width of the prediction interval — a sharpness summary."""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def winkler_score(y, lo, hi, alpha: float = 0.1) -> float:
    """Winkler / interval score — a strictly proper score for prediction intervals.

    Penalises both width and miss magnitude; misses are amplified by 2/α.
    Lower is better.
    """
    y, lo, hi = map(np.asarray, (y, lo, hi))
    width = hi - lo
    under = (lo - y) * (y < lo)
    over  = (y - hi) * (y > hi)
    return float(np.mean(width + (2 / alpha) * (under + over)))


# ─── Loss-aware / business-cost metrics ─────────────────────────────────────

def asymmetric_mae(y, p, over: float = 1.0, under: float = 1.0) -> float:
    """Asymmetric L1: penalize over-forecast vs. under-forecast differently.

    Defaults reduce to standard MAE.
    Typical retail:            under > over   (stockouts cost more)
    Typical capacity planning: over  > under  (under-provision is catastrophic)
    """
    y, p = np.asarray(y), np.asarray(p)
    err = p - y                             # positive ⇒ over-forecast
    return float(np.mean(np.where(err > 0, over * err, -under * err)))


def newsvendor_cost(y, p, c_under: float, c_over: float) -> float:
    """Classic newsvendor expected cost for a point forecast `p`.

    The optimal order quantity for a known distribution is the quantile
    c_under / (c_under + c_over); this metric just scores a chosen order.
    """
    y, p = np.asarray(y), np.asarray(p)
    shortfall = np.maximum(y - p, 0.0)
    surplus   = np.maximum(p - y, 0.0)
    return float(np.mean(c_under * shortfall + c_over * surplus))


def stockout_rate(y, p) -> float:
    """Fraction of periods where the forecast underestimated demand."""
    return float(np.mean(np.asarray(p) < np.asarray(y)))