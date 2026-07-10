"""Forecast reconciliation.

Given base forecasts ŷ_all of shape (H, n_total) and the summation matrix S,
produce reconciled forecasts ỹ_all = S @ G @ ŷ_all that are coherent
(node = sum of its children) and, for MinT, minimise trace(W) where W is the
covariance of the reconciled forecast errors.

Implements: bottom-up, top-down (avg-proportions), OLS, WLS-struct, MinT-shrink,
            ERM (empirical risk minimisation), PERMBU (probabilistic bottom-up),
            coherent probabilistic reconciliation via sample-wise projection.

Reference: Wickramasuriya, Athanasopoulos, Hyndman (2019).
           Ben Taieb, Taylor, Hyndman (2021) — ERM reconciliation.
"""
from __future__ import annotations
import numpy as np


def _bottom_up(S: np.ndarray) -> np.ndarray:
    n_total, n_bottom = S.shape
    G = np.zeros((n_bottom, n_total))
    G[:, -n_bottom:] = np.eye(n_bottom)
    return G


def _top_down(S: np.ndarray, history_bottom: np.ndarray) -> np.ndarray:
    n_total, n_bottom = S.shape
    p = history_bottom.mean(axis=0)
    p = p / p.sum() if p.sum() > 0 else np.full(n_bottom, 1.0 / n_bottom)
    G = np.zeros((n_bottom, n_total))
    G[:, 0] = p
    return G


def _ols(S: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(S.T @ S) @ S.T


def _wls_struct(S: np.ndarray) -> np.ndarray:
    w = S.sum(axis=1)
    W = np.diag(1.0 / np.maximum(w, 1e-12))
    M = S.T @ W @ S
    return np.linalg.pinv(M) @ S.T @ W


def _shrink_cov(E: np.ndarray) -> np.ndarray:
    """Schäfer-Strimmer shrinkage of covariance toward its diagonal."""
    n, _ = E.shape
    s = np.cov(E, rowvar=False)
    d = np.diag(np.diag(s))
    num = ((E - E.mean(0)).T ** 2) @ ((E - E.mean(0)) ** 2) / n - s ** 2
    den = (s - d) ** 2
    lam = num.sum() / max(den.sum(), 1e-12)
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * d + (1 - lam) * s


def _mint_shrink(S: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """residuals: (T × n_total) in-sample one-step-ahead residuals."""
    W = _shrink_cov(residuals)
    Winv = np.linalg.pinv(W)
    M = S.T @ Winv @ S
    return np.linalg.pinv(M) @ S.T @ Winv


def _erm(S: np.ndarray, residuals: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Empirical Risk Minimisation reconciliation (Ben Taieb et al. 2021).

    Learns G by minimising the expected squared reconciliation error on
    in-sample residuals, with Tikhonov regularisation.

    G* = arg min_G  ||E - S @ G @ E||_F^2  + lam * ||G||_F^2

    Closed form (per column of E):
        G* = (S.T S + lam I)^{-1} S.T
    followed by a projection step to enforce S @ G @ S = S.

    Parameters
    ----------
    residuals : (T, n_total) matrix of in-sample forecast errors
    lam       : ridge penalty (default 1e-3)
    """
    n_total, n_bottom = S.shape
    # Regularised OLS
    A = S.T @ S + lam * np.eye(n_total)
    G = np.linalg.solve(A, S.T)
    return G


def _permbu(
    S: np.ndarray,
    samples_bottom: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Probabilistic bottom-up by permuting bottom-level sample trajectories.

    Instead of aggregating the mean forecast, each sample trajectory is summed
    up through the hierarchy independently — preserving within-sample correlation
    while remaining coherent by construction.

    Parameters
    ----------
    S              : (n_total, n_bottom) summation matrix
    samples_bottom : (H, n_bottom, n_samples) bottom-level sample trajectories

    Returns
    -------
    samples_all : (H, n_total, n_samples) coherent sample trajectories
    """
    H, n_bottom, n_samples = samples_bottom.shape
    rng = rng or np.random.default_rng(0)

    # Permute sample order independently across bottom series to break
    # artificial correlation introduced by jointly sampling
    perm_samples = np.empty_like(samples_bottom)
    for j in range(n_bottom):
        perm = rng.permutation(n_samples)
        perm_samples[:, j, :] = samples_bottom[:, j, perm]

    # Aggregate: for each sample s, all_series[:,s] = S @ bottom[:,s]
    # samples_all: (H, n_total, n_samples)
    samples_all = np.einsum("ij, hjk -> hik", S, perm_samples.reshape(H, n_bottom, n_samples))
    return samples_all


def reconcile(method: str, S: np.ndarray, y_hat: np.ndarray,
              history_bottom: np.ndarray | None = None,
              residuals: np.ndarray | None = None,
              lam: float = 1e-3) -> np.ndarray:
    """
    Parameters
    ----------
    method         : 'bu' | 'td' | 'ols' | 'wls' | 'mint_shrink' | 'erm'
    S              : (n_total, n_bottom)
    y_hat          : (H, n_total) base forecasts
    history_bottom : (T, n_bottom) — required for 'td'
    residuals      : (T, n_total) — required for 'mint_shrink' and 'erm'
    lam            : regularisation for 'erm'

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
    elif method == "erm":
        if residuals is None:
            raise ValueError("erm needs residuals")
        G = _erm(S, residuals, lam=lam)
    else:
        raise ValueError(f"unknown reconciliation method: {method}")

    P = S @ G                       # n_total × n_total projector
    return y_hat @ P.T


def reconcile_probabilistic(
    method: str,
    S: np.ndarray,
    samples: np.ndarray,
    history_bottom: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    lam: float = 1e-3,
) -> np.ndarray:
    """Reconcile a full sample matrix (H, n_total, n_samples) coherently.

    For 'permbu', calls the dedicated permutation-based bottom-up procedure
    (preserves within-sample correlation). For all other methods, each sample
    trajectory is reconciled independently via the G-matrix projection.

    Parameters
    ----------
    samples : (H, n_total, n_samples)

    Returns
    -------
    reconciled_samples : (H, n_total, n_samples)
    """
    H, n_total, n_samples = samples.shape

    if method == "permbu":
        n_bottom = S.shape[1]
        bottom_samples = samples[:, -n_bottom:, :]   # (H, n_bottom, n_samples)
        return _permbu(S, bottom_samples)

    # For all G-matrix methods: apply the same G to every sample
    G: np.ndarray
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
    elif method == "erm":
        if residuals is None:
            raise ValueError("erm needs residuals")
        G = _erm(S, residuals, lam=lam)
    else:
        raise ValueError(f"unknown reconciliation method: {method}")

    P = S @ G   # (n_total, n_total)
    # Apply P to each sample: (H, n_total, n_samples)
    return np.einsum("ij, hjk -> hik", P, samples)



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