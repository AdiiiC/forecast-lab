"""Production monitoring: residual drift + coverage drift + concept drift + alerts.

New in this version
-------------------
* **Page-Hinkley test** — online concept drift on rolling MAE
* **ADWIN detector** — adaptive windowing for non-stationary streams
* **Covariate shift** — wires schema.ks_test per feature into the report
* **retrain_needed** flag in MonitorReport
* **Webhook delivery** — POST MonitorReport as JSON to a configurable URL
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..schema import psi, ks_test


# ──────────────────────────────────────────────────────────────────────────────
# Page-Hinkley online drift detector
# ──────────────────────────────────────────────────────────────────────────────

class PageHinkley:
    """Page-Hinkley test for detecting upward shifts in a stream.

    Parameters
    ----------
    delta   : allowed magnitude of change (sensitivity)
    lambda_ : detection threshold (higher → fewer false alarms)
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0):
        self.delta = delta
        self.lambda_ = lambda_
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0

    def update(self, value: float) -> bool:
        """Add one observation. Returns True if drift is detected."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        return (self._sum - self._min_sum) > self.lambda_

    def reset(self):
        self.__init__(self.delta, self.lambda_)


# ──────────────────────────────────────────────────────────────────────────────
# Simplified ADWIN
# ──────────────────────────────────────────────────────────────────────────────

class ADWIN:
    """Adaptive Windowing (simplified) drift detector.

    Monitors a stream of values. When the mean of a recent sub-window
    differs significantly from the rest, drift is flagged and the
    window is shrunk.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: list[float] = []
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self._window.append(value)
        self.drift_detected = self._detect()
        return self.drift_detected

    def _detect(self) -> bool:
        n = len(self._window)
        if n < 4:
            return False
        w = np.array(self._window)
        total_mean = w.mean()
        # Check all possible cut-points
        for cut in range(2, n - 1):
            n0, n1 = cut, n - cut
            m0, m1 = w[:cut].mean(), w[cut:].mean()
            # Hoeffding-style bound
            eps = np.sqrt((1 / (2 * n0) + 1 / (2 * n1)) * np.log(4 * n / self.delta))
            if abs(m0 - m1) >= eps:
                # Shrink: keep the more recent window
                self._window = self._window[cut:]
                return True
        return False

    def reset(self):
        self._window.clear()
        self.drift_detected = False


# ──────────────────────────────────────────────────────────────────────────────
# Report dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MonitorReport:
    residual_psi: float
    rolling_mae: float
    rolling_coverage: float
    nominal_coverage: float
    alerts: list[str]
    retrain_needed: bool = False
    covariate_drift: dict[str, Any] = field(default_factory=dict)
    ph_drift: bool = False
    adwin_drift: bool = False

    def to_dict(self) -> dict:
        return {
            "residual_psi": self.residual_psi,
            "rolling_mae": self.rolling_mae,
            "rolling_coverage": self.rolling_coverage,
            "nominal_coverage": self.nominal_coverage,
            "alerts": self.alerts,
            "retrain_needed": self.retrain_needed,
            "covariate_drift": self.covariate_drift,
            "ph_drift": self.ph_drift,
            "adwin_drift": self.adwin_drift,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluate function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    ref_residuals: np.ndarray,
    cur_residuals: np.ndarray,
    cur_in_interval: np.ndarray,
    nominal_coverage: float,
    mae_threshold: float | None = None,
    coverage_tol: float = 0.05,
    ref_features: pd.DataFrame | None = None,
    cur_features: pd.DataFrame | None = None,
    psi_retrain_threshold: float = 0.25,
    ph_delta: float = 0.005,
    ph_lambda: float = 50.0,
    adwin_delta: float = 0.002,
) -> MonitorReport:
    """Evaluate a production window and return a MonitorReport.

    Parameters
    ----------
    ref_residuals          : residuals from the training / reference window
    cur_residuals          : residuals from the current production window
    cur_in_interval        : boolean array — was each prediction inside PI?
    nominal_coverage       : target coverage level (e.g. 0.9)
    mae_threshold          : optional hard MAE alert threshold
    coverage_tol           : allowed deviation from nominal_coverage
    ref_features / cur_features : optional DataFrames for covariate shift
    psi_retrain_threshold  : PSI level that triggers retrain_needed=True
    ph_delta / ph_lambda   : Page-Hinkley sensitivity / threshold
    adwin_delta            : ADWIN significance level
    """
    alerts: list[str] = []
    retrain_needed = False

    # ── Residual PSI ──────────────────────────────────────────────────
    p = psi(ref_residuals, cur_residuals)
    if p > psi_retrain_threshold:
        alerts.append(f"residual PSI={p:.3f} (serious drift)")
        retrain_needed = True
    elif p > 0.10:
        alerts.append(f"residual PSI={p:.3f} (warn)")

    # ── Coverage drift ────────────────────────────────────────────────
    cov = float(np.mean(cur_in_interval))
    if abs(cov - nominal_coverage) > coverage_tol:
        alerts.append(f"coverage drift: {cov:.3f} vs nominal {nominal_coverage:.3f}")
        if abs(cov - nominal_coverage) > 2 * coverage_tol:
            retrain_needed = True

    # ── MAE threshold ─────────────────────────────────────────────────
    rolling_mae = float(np.mean(np.abs(cur_residuals)))
    if mae_threshold is not None and rolling_mae > mae_threshold:
        alerts.append(f"MAE {rolling_mae:.3f} > threshold {mae_threshold}")
        retrain_needed = True

    # ── Page-Hinkley on rolling MAE ───────────────────────────────────
    ph = PageHinkley(delta=ph_delta, lambda_=ph_lambda)
    ph_drift = False
    for err in np.abs(cur_residuals):
        if ph.update(float(err)):
            ph_drift = True
            break
    if ph_drift:
        alerts.append("Page-Hinkley: upward MAE shift detected")
        retrain_needed = True

    # ── ADWIN on residual stream ──────────────────────────────────────
    adwin = ADWIN(delta=adwin_delta)
    adwin_drift = False
    for err in np.abs(cur_residuals):
        if adwin.update(float(err)):
            adwin_drift = True
            break
    if adwin_drift:
        alerts.append("ADWIN: non-stationary residuals detected")
        retrain_needed = True

    # ── Covariate shift ───────────────────────────────────────────────
    covariate_drift: dict[str, Any] = {}
    if ref_features is not None and cur_features is not None:
        common = ref_features.columns.intersection(cur_features.columns)
        for col in common:
            if not np.issubdtype(ref_features[col].dtype, np.number):
                continue
            ks_stat, ks_pval = ks_test(
                ref_features[col].dropna().values,
                cur_features[col].dropna().values,
            )
            flag = "drift" if ks_pval < 0.01 else ("warn" if ks_pval < 0.05 else "ok")
            covariate_drift[col] = {"KS": round(ks_stat, 4), "p": round(ks_pval, 4), "flag": flag}
            if flag == "drift":
                alerts.append(f"covariate shift: {col} (KS p={ks_pval:.3f})")
                retrain_needed = True

    return MonitorReport(
        residual_psi=p,
        rolling_mae=rolling_mae,
        rolling_coverage=cov,
        nominal_coverage=nominal_coverage,
        alerts=alerts,
        retrain_needed=retrain_needed,
        covariate_drift=covariate_drift,
        ph_drift=ph_drift,
        adwin_drift=adwin_drift,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Webhook delivery
# ──────────────────────────────────────────────────────────────────────────────

def deliver_report(report: MonitorReport, webhook_url: str | None = None,
                   timeout: float = 5.0) -> bool:
    """POST the MonitorReport as JSON to ``webhook_url`` (Slack / PagerDuty / custom).

    Returns True if delivery succeeded.
    """
    if webhook_url is None:
        return False
    try:
        import urllib.request
        payload = json.dumps(report.to_dict()).encode()
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return True
    except Exception:
        return False
