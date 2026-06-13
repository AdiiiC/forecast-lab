"""Calibration diagnostics for probabilistic forecasts.

* PIT histogram          — should be uniform if the predictive distribution is calibrated
* Reliability diagram    — empirical coverage vs. nominal across α grid
* Sharpness vs. coverage — Pareto frontier across models / α levels
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def pit_values(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """PIT from samples: F̂(y) = fraction of samples ≤ y per step."""
    return np.mean(samples <= y[:, None], axis=1)


def pit_from_quantiles(y: np.ndarray, quantiles: np.ndarray,
                       q_levels: np.ndarray) -> np.ndarray:
    """Linear interpolation: find q such that Q(q) = y."""
    out = np.empty(len(y))
    for i, yi in enumerate(y):
        v = quantiles[i]
        # invert by linear interpolation in (v, q_levels)
        if yi <= v[0]:
            out[i] = q_levels[0] / 2
        elif yi >= v[-1]:
            out[i] = (1 + q_levels[-1]) / 2
        else:
            out[i] = np.interp(yi, v, q_levels)
    return out


def reliability(y, lo_fn, hi_fn, alphas=np.linspace(0.05, 0.5, 10)):
    """
    lo_fn, hi_fn : callables alpha -> array of lower/upper bounds.
    Returns DataFrame with nominal vs. empirical coverage and mean PI width.
    """
    rows = []
    for a in alphas:
        lo, hi = lo_fn(a), hi_fn(a)
        cov = float(np.mean((y >= lo) & (y <= hi)))
        rows.append(dict(nominal=1 - a, empirical=cov,
                         width=float(np.mean(hi - lo))))
    return pd.DataFrame(rows)


def plot_diagnostics(results: dict, out: Path):
    """Render PIT histograms + reliability diagrams + sharpness-coverage plots."""
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)

    # PIT histograms (only models with samples or quantiles)
    fig, axes = plt.subplots(1, max(1, len(results)),
                             figsize=(3.2 * max(1, len(results)), 3))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, folds) in zip(axes, results.items()):
        y_true = np.concatenate([f.y_true for f in folds])
        if all(f.samples is not None for f in folds):
            samp = np.concatenate([f.samples for f in folds], axis=0)
            pit = pit_values(y_true, samp)
        elif all(f.quantiles is not None for f in folds):
            q = np.concatenate([f.quantiles for f in folds], axis=0)
            pit = pit_from_quantiles(y_true, q, folds[0].q_levels)
        else:
            ax.set_visible(False)
            continue
        ax.hist(pit, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axhline(len(pit) / 20, color="black", ls="--", lw=1)
        ax.set_title(f"PIT — {name}")
        ax.set_xlabel("PIT")
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "pit_histograms.png", dpi=130)
    plt.close(fig)

    # Sharpness vs coverage scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, folds in results.items():
        if not all(np.isfinite(f.lo).all() for f in folds):
            continue
        y_true = np.concatenate([f.y_true for f in folds])
        lo = np.concatenate([f.lo for f in folds])
        hi = np.concatenate([f.hi for f in folds])
        cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
        width = float(np.mean(hi - lo))
        ax.scatter(cov, width, s=60)
        ax.annotate(name, (cov, width),
                    textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("empirical coverage")
    ax.set_ylabel("mean PI width")
    ax.set_title("sharpness vs. coverage (lower-right is better)")
    fig.tight_layout()
    fig.savefig(out / "sharpness_coverage.png", dpi=130)
    plt.close(fig)