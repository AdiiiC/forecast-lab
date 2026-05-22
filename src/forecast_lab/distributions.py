"""Unified predictive distribution interface.

A `PredictiveDistribution` can always answer three questions:
  * mean()              — point forecast
  * quantile(q)         — any quantile
  * sample(n)           — n Monte-Carlo trajectories (T, n) or (H, n)

Concrete implementations: Gaussian, StudentT, Quantile (interpolated), Empirical
(sample-based). All operate horizon-wise (vector per step).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


class PredictiveDistribution:
    def mean(self) -> np.ndarray: ...
    def quantile(self, q: float | np.ndarray) -> np.ndarray: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...
    def interval(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        return self.quantile(alpha / 2), self.quantile(1 - alpha / 2)


@dataclass
class Gaussian(PredictiveDistribution):
    mu: np.ndarray
    sigma: np.ndarray

    def mean(self): return self.mu
    def quantile(self, q): return self.mu + self.sigma * stats.norm.ppf(q)
    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return self.mu[:, None] + self.sigma[:, None] * rng.standard_normal((len(self.mu), n))


@dataclass
class StudentT(PredictiveDistribution):
    mu: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray | float

    def mean(self): return self.mu
    def quantile(self, q):
        return self.mu + self.sigma * stats.t.ppf(q, df=self.nu)
    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        nu = np.broadcast_to(np.asarray(self.nu, dtype=float), self.mu.shape)
        # sample t via normal/chi2 trick
        z = rng.standard_normal((len(self.mu), n))
        g = rng.chisquare(nu[:, None], (len(self.mu), n)) / nu[:, None]
        return self.mu[:, None] + self.sigma[:, None] * z / np.sqrt(g)


@dataclass
class Quantile(PredictiveDistribution):
    """Quantile forecast: q_levels (K,), q_values (H, K). Linear interpolation in q."""
    q_levels: np.ndarray
    q_values: np.ndarray

    def mean(self):
        # trapezoidal integral of quantile fn over [0,1] → mean (if 0/1 not present, extrap nearest)
        q, v = self.q_levels, self.q_values
        return np.trapz(v, q, axis=1) / (q[-1] - q[0] + 1e-12)

    def quantile(self, q):
        q = np.atleast_1d(q).astype(float)
        out = np.empty((self.q_values.shape[0], len(q)))
        for i, qi in enumerate(q):
            out[:, i] = np.array([np.interp(qi, self.q_levels, self.q_values[h])
                                  for h in range(self.q_values.shape[0])])
        return out.squeeze()

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        u = rng.uniform(size=(self.q_values.shape[0], n))
        return np.stack([np.interp(u[h], self.q_levels, self.q_values[h])
                         for h in range(self.q_values.shape[0])])


@dataclass
class Empirical(PredictiveDistribution):
    """Sample-based distribution. samples: (H, S)."""
    samples: np.ndarray

    def mean(self): return self.samples.mean(axis=1)
    def quantile(self, q): return np.quantile(self.samples, q, axis=1)
    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.samples.shape[1], size=n)
        return self.samples[:, idx]