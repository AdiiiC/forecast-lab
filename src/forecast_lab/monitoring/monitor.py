"""Production monitoring: residual drift + coverage drift + alert thresholds."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..schema import psi


@dataclass
class MonitorReport:
    residual_psi: float
    rolling_mae: float
    rolling_coverage: float
    nominal_coverage: float
    alerts: list[str]


def evaluate(ref_residuals: np.ndarray, cur_residuals: np.ndarray,
             cur_in_interval: np.ndarray, nominal_coverage: float,
             mae_threshold: float | None = None,
             coverage_tol: float = 0.05) -> MonitorReport:
    alerts: list[str] = []
    p = psi(ref_residuals, cur_residuals)
    if p > 0.25:                                  alerts.append(f"residual PSI={p:.2f} (drift)")
    elif p > 0.10:                                alerts.append(f"residual PSI={p:.2f} (warn)")
    cov = float(np.mean(cur_in_interval))
    if abs(cov - nominal_coverage) > coverage_tol:
        alerts.append(f"coverage drift: {cov:.2f} vs nominal {nominal_coverage:.2f}")
    mae = float(np.mean(np.abs(cur_residuals)))
    if mae_threshold and mae > mae_threshold:
        alerts.append(f"MAE {mae:.2f} > threshold {mae_threshold}")
    return MonitorReport(residual_psi=p, rolling_mae=mae,
                         rolling_coverage=cov,
                         nominal_coverage=nominal_coverage, alerts=alerts)