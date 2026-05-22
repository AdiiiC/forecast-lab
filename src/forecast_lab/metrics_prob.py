"""Proper scoring rules for probabilistic forecasts."""
from __future__ import annotations
import numpy as np


def pinball_loss(y, q_pred, q: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def quantile_loss_mean(y, quantiles: np.ndarray, q_levels: np.ndarray) -> float:
    """Average pinball loss across quantile levels — a common proxy for CRPS."""
    return float(np.mean([pinball_loss(y, quantiles[:, i], q)
                          for i, q in enumerate(q_levels)]))


def crps_sample(y: np.ndarray, samples: np.ndarray) -> float:
    """CRPS from samples (H, S). Uses the standard E|X-y| - 0.5 E|X-X'| identity."""
    y = np.asarray(y); s = np.asarray(samples)
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    # E|X - X'| via sorted-sample shortcut: O(S log S) per horizon
    s_sorted = np.sort(s, axis=1); S = s.shape[1]
    weights = (2 * np.arange(1, S + 1) - S - 1)
    term2 = (weights * s_sorted).sum(axis=1) / (S * S)
    return float(np.mean(term1 - 0.5 * term2))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Analytic CRPS for N(mu, sigma^2)."""
    from scipy.stats import norm
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1)
                                  + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def energy_score(y: np.ndarray, samples: np.ndarray, beta: float = 1.0) -> float:
    """Multivariate (here treated as horizon-vector) energy score.
    y: (H,), samples: (H, S). Reduces to CRPS when H=1.
    """
    s = samples  # (H, S)
    diffs_obs = np.linalg.norm(s - y[:, None], axis=0)  # (S,)
    term1 = diffs_obs.mean() ** beta
    S = s.shape[1]
    # E||X - X'|| via random pairing — O(S) memory
    idx = np.random.default_rng(0).permutation(S)
    diffs_pair = np.linalg.norm(s - s[:, idx], axis=0)
    term2 = diffs_pair.mean() ** beta
    return float(term1 - 0.5 * term2)


def log_score_gaussian(y, mu, sigma) -> float:
    from scipy.stats import norm
    return float(-np.mean(norm.logpdf(y, mu, sigma)))