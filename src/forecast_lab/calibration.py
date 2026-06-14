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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)

    # ── PIT histograms ────────────────────────────────────────────────────────
    # Layout: rows of up to 3 subplots, each 4×4 inches — keeps aspect ratio
    # sensible when Streamlit displays at container width.
    names = list(results.keys())
    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("#0f1117")

    for idx, (name, folds) in enumerate(results.items()):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#161b27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")

        y_true = np.concatenate([f.y_true for f in folds])

        if all(f.samples is not None for f in folds):
            samp = np.concatenate([f.samples for f in folds], axis=0)
            pit = pit_values(y_true, samp)
        elif all(f.quantiles is not None for f in folds):
            q = np.concatenate([f.quantiles for f in folds], axis=0)
            pit = pit_from_quantiles(y_true, q, folds[0].q_levels)
        elif all(f.lo is not None and np.isfinite(f.lo).all() for f in folds):
            # Conformal / point models: approximate PIT via Gaussian using PI width.
            # 90% PI → z=1.645; sigma ≈ (hi-lo)/(2*1.645)
            from scipy.stats import norm
            lo = np.concatenate([f.lo for f in folds])
            hi = np.concatenate([f.hi for f in folds])
            mean = np.concatenate([f.y_pred for f in folds])
            sigma = np.maximum((hi - lo) / (2 * 1.645), 1e-9)
            pit = norm.cdf(y_true, loc=mean, scale=sigma)
        else:
            ax.text(0.5, 0.5, "No distributional\noutput", ha="center",
                    va="center", color="#8b949e", transform=ax.transAxes)
            ax.set_title(f"PIT — {name}", fontsize=10)
            continue

        ax.hist(pit, bins=20, range=(0, 1),
                color="#388bfd", edgecolor="#0f1117", alpha=0.85)
        ax.axhline(len(pit) / 20, color="#f0f6fc", ls="--", lw=1)
        ax.set_title(f"PIT — {name}", fontsize=10)
        ax.set_xlabel("PIT")
        ax.set_xlim(0, 1)

    # Hide unused subplot cells
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(out / "pit_histograms.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Sharpness vs coverage scatter ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")

    colors = plt.cm.tab10.colors
    for i, (name, folds) in enumerate(results.items()):
        if not all(np.isfinite(f.lo).all() for f in folds):
            continue
        y_true = np.concatenate([f.y_true for f in folds])
        lo = np.concatenate([f.lo for f in folds])
        hi = np.concatenate([f.hi for f in folds])
        cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
        width = float(np.mean(hi - lo))
        ax.scatter(cov, width, s=80, color=colors[i % len(colors)], zorder=3)
        ax.annotate(name, (cov, width),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color="#e6edf3")
    ax.set_xlabel("empirical coverage")
    ax.set_ylabel("mean PI width")
    ax.set_title("sharpness vs. coverage (lower-right is better)")
    fig.tight_layout()
    fig.savefig(out / "sharpness_coverage.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)