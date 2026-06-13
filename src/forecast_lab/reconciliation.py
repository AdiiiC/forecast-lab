"""Forecast reconciliation.

Given base forecasts ŷ_all of shape (H, n_total) and the summation matrix S,
produce reconciled forecasts ỹ_all = S @ G @ ŷ_all that are coherent
(node = sum of its children) and, for MinT, minimise trace(W) where W is the
covariance of the reconciled forecast errors.

Implements: bottom-up, top-down (avg-proportions), OLS, WLS-struct, MinT-shrink.
Reference: Wickramasuriya, Athanasopoulos, Hyndman (2019).
"""
from __future__ import annotations
import numpy as np


def _bottom_up(S: np.ndarray) -> np.ndarray:
    """G picks bottom rows of ŷ; reconciled = S @ G @ ŷ."""
    n_total, n_bottom = S.shape
    G = np.zeros((n_bottom, n_total))
    G[:, -n_bottom:] = np.eye(n_bottom)
    return G


def _top_down(S: np.ndarray, history_bottom: np.ndarray) -> np.ndarray:
    """Average historical proportions; ŷ at the top row drives bottom."""
    n_total, n_bottom = S.shape
    p = history_bottom.mean(axis=0)
    p = p / p.sum() if p.sum() > 0 else np.full(n_bottom, 1.0 / n_bottom)
    G = np.zeros((n_bottom, n_total))
    G[:, 0] = p
    return G


def _ols(S: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(S.T @ S) @ S.T


def _wls_struct(S: np.ndarray) -> np.ndarray:
    """Weights = row sums of S (proxy for series scale)."""
    w = S.sum(axis=1)
    W = np.diag(1.0 / w)
    M = S.T @ W @ S
    return np.linalg.pinv(M) @ S.T @ W


def _shrink_cov(E: np.ndarray) -> np.ndarray:
    """Schäfer-Strimmer shrinkage of covariance toward its diagonal."""
    n, _ = E.shape
    s = np.cov(E, rowvar=False)
    d = np.diag(np.diag(s))
    # shrinkage intensity λ* (clamped to [0,1])
    num = ((E - E.mean(0)).T ** 2) @ ((E - E.mean(0)) ** 2) / n - s ** 2
    den = (s - d) ** 2
    lam = num.sum() / max(den.sum(), 1e-12)
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * d + (1 - lam) * s


def _mint_shrink(S: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """residuals: (T × n_total) in-sample one-step-ahead residuals across all nodes."""
    W = _shrink_cov(residuals)
    Winv = np.linalg.pinv(W)
    M = S.T @ Winv @ S
    return np.linalg.pinv(M) @ S.T @ Winv


def reconcile(method: str, S: np.ndarray, y_hat: np.ndarray,
              history_bottom: np.ndarray | None = None,
              residuals: np.ndarray | None = None) -> np.ndarray:
    """
    Parameters
    ----------
    method         : 'bu' | 'td' | 'ols' | 'wls' | 'mint_shrink'
    S              : (n_total, n_bottom)
    y_hat          : (H, n_total) base forecasts
    history_bottom : (T, n_bottom) — required for 'td'
    residuals      : (T, n_total) — required for 'mint_shrink'

    Returns
    -------
    y_tilde : (H, n_total) reconciled forecasts (coherent)
    """
    if method == "bu":
        G = _bottom_up(S)
    elif method == "td":
        if history_bottom is None:
            raise ValueError("top-down needs history_bottom")
        G = _top_down(S, history_bottom)
    elif method == "ols":
        G = _ols(S)
    elif method == "wls":
        G = _wls_struct(S)
    elif method == "mint_shrink":
        if residuals is None:
            raise ValueError("mint_shrink needs residuals")
        G = _mint_shrink(S, residuals)
    else:
        raise ValueError(f"unknown reconciliation method: {method}")

    P = S @ G                       # n_total × n_total projector
    return y_hat @ P.T