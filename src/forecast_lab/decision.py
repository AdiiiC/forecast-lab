"""Forecast → decision: convert (mean, quantiles, samples) into policies.

Three policies provided:
  * newsvendor_order       — single-period optimal order under linear costs
  * safety_stock           — multi-period replenishment with a service-level target
  * dispatch_threshold     — call an upstream resource if upper PI exceeds capacity

All operate on a `Forecast` object (mean + lo/hi or samples/quantiles).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models.base import Forecast


def _quantile_from_forecast(fc: Forecast, q: float) -> np.ndarray:
    if fc.samples is not None:
        return np.quantile(fc.samples, q, axis=1)
    if fc.quantiles is not None and fc.q_levels is not None:
        return np.array([np.interp(q, fc.q_levels, fc.quantiles[h])
                         for h in range(fc.quantiles.shape[0])])
    if fc.lo is not None and fc.hi is not None:
        # Assume symmetric — recover sigma from 90% PI as a fallback
        from scipy.stats import norm
        sigma = (fc.hi - fc.lo) / (2 * norm.ppf(0.95))
        return fc.mean + sigma * norm.ppf(q)
    raise ValueError("forecast has no usable distributional info")


def newsvendor_order(fc: Forecast, c_under: float, c_over: float) -> np.ndarray:
    """Per-horizon optimal order = quantile c_under/(c_under+c_over)."""
    q = c_under / (c_under + c_over)
    return _quantile_from_forecast(fc, q)


def safety_stock(fc: Forecast, lead_time: int, service_level: float = 0.95
                 ) -> float:
    """Reorder point = expected lead-time demand + safety stock for service level."""
    h = min(lead_time, len(fc.mean))
    mu_lt = float(np.sum(fc.mean[:h]))
    # variance of lead-time demand from interval width (Gaussian assumption)
    from scipy.stats import norm
    z = norm.ppf(service_level)
    if fc.samples is not None:
        sigma_lt = float(np.std(fc.samples[:h].sum(axis=0)))
    elif fc.lo is not None:
        sigma_h = (fc.hi[:h] - fc.lo[:h]) / (2 * norm.ppf(0.95))
        sigma_lt = float(np.sqrt(np.sum(sigma_h ** 2)))
    else:
        sigma_lt = 0.0
    return mu_lt + z * sigma_lt


@dataclass
class DispatchPlan:
    capacity: float
    risk_alpha: float
    triggered_steps: np.ndarray
    expected_overage: np.ndarray


def dispatch_threshold(fc: Forecast, capacity: float,
                       risk_alpha: float = 0.1) -> DispatchPlan:
    """Trigger an upstream/peaker resource where the (1-α) upper bound exceeds
    capacity. Returns per-step trigger flags and expected overage."""
    upper = _quantile_from_forecast(fc, 1 - risk_alpha)
    triggered = upper > capacity
    overage = np.maximum(fc.mean - capacity, 0.0)
    return DispatchPlan(capacity=capacity, risk_alpha=risk_alpha,
                        triggered_steps=triggered.astype(int),
                        expected_overage=overage)