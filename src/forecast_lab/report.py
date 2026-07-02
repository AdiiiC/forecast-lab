"""Aggregation + reporting with probabilistic metrics, interval scores,
business-cost metrics, DM-test significance, and calibration diagnostics.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .metrics import (
    mae, rmse, smape, mase, coverage, interval_width,
    winkler_score, asymmetric_mae,
)
from .metrics_prob import crps_sample, energy_score, quantile_loss_mean
from .stats_tests import diebold_mariano
from .calibration import plot_diagnostics
from .backtest import FoldResult


def _finite_list(arr) -> list:
    """Convert an array to a JSON-safe list, mapping non-finite values to None."""
    out = []
    for v in np.asarray(arr, dtype=float).ravel():
        out.append(float(v) if np.isfinite(v) else None)
    return out


def _gather(folds):
    return (
        np.concatenate([f.y_true for f in folds]),
        np.concatenate([f.y_pred for f in folds]),
        np.concatenate([f.lo     for f in folds]),
        np.concatenate([f.hi     for f in folds]),
    )


def aggregate(results: dict[str, list[FoldResult]], season: int,
              alpha: float, baseline: str = "seasonal_naive",
              cost_over: float = 1.0, cost_under: float = 3.0) -> pd.DataFrame:
    """Build the full leaderboard.

    Columns: MAE, RMSE, sMAPE, MASE, coverage, PI_width, Winkler, NV_cost,
    [CRPS, energy, QLoss when distributional outputs are available],
    skill_vs_naive_%, DM_p_vs_naive, sig.
    """
    rows = []
    for name, folds in results.items():
        y_true, y_pred, lo, hi = _gather(folds)
        train_ref = folds[0].train_tail

        row = dict(
            model=name,
            MAE=mae(y_true, y_pred),
            RMSE=rmse(y_true, y_pred),
            sMAPE=smape(y_true, y_pred),
            MASE=mase(y_true, y_pred, train_ref, season=season),
            coverage=coverage(y_true, lo, hi) if np.isfinite(lo).all() else np.nan,
            PI_width=interval_width(lo, hi)   if np.isfinite(lo).all() else np.nan,
        )

        # Interval score + asymmetric business-cost loss
        if np.isfinite(lo).all():
            row["Winkler"] = winkler_score(y_true, lo, hi, alpha=alpha)
        row[f"NV_cost({cost_under:.0f}:{cost_over:.0f})"] = asymmetric_mae(
            y_true, y_pred, over=cost_over, under=cost_under)

        # Probabilistic metrics (when samples / quantiles are present)
        samples_list = [getattr(f, "samples", None) for f in folds]
        if all(s is not None and s.size for s in samples_list):
            samp = np.concatenate(samples_list, axis=0)
            row["CRPS"]   = crps_sample(y_true, samp)
            row["energy"] = energy_score(y_true, samp)

        q_list  = [getattr(f, "quantiles", None) for f in folds]
        ql_list = [getattr(f, "q_levels",  None) for f in folds]
        if all(q is not None for q in q_list):
            row["QLoss"] = quantile_loss_mean(
                y_true, np.concatenate(q_list, axis=0), ql_list[0])

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    # Skill score + Diebold–Mariano significance vs. baseline
    if baseline in df.index:
        df["skill_vs_naive_%"] = (1 - df["MAE"] / df.loc[baseline, "MAE"]) * 100
        base_pred = np.concatenate([f.y_pred for f in results[baseline]])
        y_true    = np.concatenate([f.y_true for f in results[baseline]])
        pvals, stars = [], []
        for name in df.index:
            if name == baseline:
                pvals.append(np.nan)
                stars.append("—")
                continue
            yp = np.concatenate([f.y_pred for f in results[name]])
            dm = diebold_mariano(y_true, yp, base_pred, h=1, loss="mae")
            pvals.append(dm.p_value)
            stars.append(
                "***" if dm.p_value < 0.01 else
                "**"  if dm.p_value < 0.05 else
                "*"   if dm.p_value < 0.10 else "")
        df["DM_p_vs_naive"] = pvals
        df["sig"] = stars

    return df.sort_values("MAE")


def print_table(df: pd.DataFrame, alpha: float):
    console = Console()
    t = Table(title=f"Walk-forward results  (nominal PI = {1-alpha:.0%}, "
                    f"DM stars: *<0.10  **<0.05  ***<0.01)",
              show_lines=False)
    t.add_column("model", style="bold cyan")
    for c in df.columns:
        t.add_column(c, justify="right")
    for name, row in df.iterrows():
        cells = [name]
        for c in df.columns:
            v = row[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(f"{v:.3f}")
        t.add_row(*cells)
    console.print(t)


def plot_folds(results: dict, out: Path, alpha: float):
    """Per-model forecast plot for the most recent fold, plus a reliability
    table for any model that produced intervals."""
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)

    series_export: dict = {}
    for name, folds in results.items():
        f = folds[-1]
        h = len(f.y_true)
        x = np.arange(h)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(x, f.y_true, label="actual", lw=2, color="black")
        ax.plot(x, f.y_pred, label="forecast", lw=2)
        has_pi = np.isfinite(f.lo).all()
        if has_pi:
            ax.fill_between(x, f.lo, f.hi, alpha=0.2,
                            label=f"{int((1-alpha)*100)}% PI")
        ax.set_title(f"{name} — last fold")
        ax.set_xlabel("horizon step")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)

        series_export[name] = {
            "actual": _finite_list(f.y_true),
            "forecast": _finite_list(f.y_pred),
            "lo": _finite_list(f.lo) if has_pi else None,
            "hi": _finite_list(f.hi) if has_pi else None,
        }

    (out.parent / "forecasts.json").write_text(
        json.dumps({"alpha": float(alpha), "series": series_export})
    )

    # Reliability table (nominal vs. empirical coverage at several α levels)
    rel_rows = []
    for name, folds in results.items():
        if not all(np.isfinite(f.lo).all() for f in folds):
            continue
        y_true = np.concatenate([f.y_true for f in folds])
        lo     = np.concatenate([f.lo     for f in folds])
        hi     = np.concatenate([f.hi     for f in folds])
        # Re-use the central PI we have (single α); for multi-α reliability,
        # models with quantile/sample outputs can be re-queried elsewhere.
        rel_rows.append(dict(
            model=name,
            nominal=1 - alpha,
            empirical=coverage(y_true, lo, hi),
            mean_width=interval_width(lo, hi),
        ))
    if rel_rows:
        pd.DataFrame(rel_rows).to_csv(out.parent / "reliability.csv", index=False)


__all__ = ["aggregate", "print_table", "plot_folds", "plot_diagnostics"]