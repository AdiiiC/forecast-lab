"""Hierarchical walk-forward backtest.

Strategy:
  1. For each fold, fit the chosen base model on every node series independently
     (this is what production code typically does for grouped retail/energy data).
  2. Collect base forecasts y_hat ∈ R^{H × n_total}.
  3. Reconcile via the requested method using in-sample residuals.
  4. Evaluate node-level errors AND aggregate-level errors for both base and
     reconciled forecasts so the impact of reconciliation is visible.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtest import walk_forward_splits
from .hierarchy import Hierarchy
from .reconciliation import reconcile


@dataclass
class HierFoldResult:
    fold: int
    train_end: pd.Timestamp
    y_true: np.ndarray        # (H, n_total)
    y_base: np.ndarray        # (H, n_total) base forecasts (node-level)
    y_recon: np.ndarray       # (H, n_total) reconciled


def _node_actuals(h: Hierarchy) -> pd.DataFrame:
    return h.stack_actuals()


def _residuals(actuals: pd.DataFrame, in_sample_pred: pd.DataFrame) -> np.ndarray:
    return (actuals - in_sample_pred).values


def hier_backtest(model_factory, hierarchy: Hierarchy, *,
                  horizon: int, n_folds: int, min_train_size: int,
                  stride: int, mode: str, alpha: float,
                  reconciliation: str = "mint_shrink") -> list[HierFoldResult]:
    """
    `model_factory` is a 0-arg callable returning a fresh BaseModel instance per node.
    """
    actuals = _node_actuals(hierarchy)        # (T, n_total)
    T = len(actuals)
    splits = list(walk_forward_splits(T, min_train_size, horizon,
                                      n_folds, stride, mode))
    results: list[HierFoldResult] = []
    S = hierarchy.S

    for i, (te, ee) in enumerate(tqdm(splits, desc="hier-folds")):
        tr_lo = 0 if mode == "expanding" else max(0, te - min_train_size)
        train_a = actuals.iloc[tr_lo:te]
        test_a  = actuals.iloc[te:ee]

        # Fit one model per node, forecast horizon, plus produce in-sample one-step
        y_hat = np.zeros((horizon, S.shape[0]))
        residuals = np.zeros_like(train_a.values)
        for j, col in enumerate(actuals.columns):
            m = copy.deepcopy(model_factory())
            s = train_a[col]
            m.fit(s)
            fc = m.predict(horizon, alpha=alpha)
            y_hat[:, j] = fc.mean
            # cheap in-sample residual: 1-step seasonal-naive proxy avoids
            # forcing every model to expose .fitted_values
            residuals[:, j] = (s.values
                               - np.r_[np.full(min(168, len(s)), s.mean()),
                                       s.values[:-min(168, len(s))]])

        y_recon = reconcile(reconciliation, S, y_hat,
                            history_bottom=train_a.values[:, -S.shape[1]:],
                            residuals=residuals)

        results.append(HierFoldResult(
            fold=i, train_end=train_a.index[-1],
            y_true=test_a.values, y_base=y_hat, y_recon=y_recon,
        ))
    return results


def hier_report(results: list[HierFoldResult], hierarchy: Hierarchy) -> pd.DataFrame:
    """Per-level MAE for base vs. reconciled — the key reconciliation diagnostic."""
    levels_of_node = [0 if n == () else len(n) for n in hierarchy.node_order]
    rows = []
    for lvl in sorted(set(levels_of_node)):
        mask = np.array([lv == lvl for lv in levels_of_node])
        yt = np.concatenate([r.y_true[:, mask] for r in results], axis=0)
        yb = np.concatenate([r.y_base[:, mask] for r in results], axis=0)
        yr = np.concatenate([r.y_recon[:, mask] for r in results], axis=0)
        rows.append(dict(
            level=("TOP" if lvl == 0 else
                   "BOTTOM" if lvl == len(hierarchy.levels) else
                   f"L{lvl}"),
            n_series=int(mask.sum()),
            MAE_base=float(np.mean(np.abs(yt - yb))),
            MAE_recon=float(np.mean(np.abs(yt - yr))),
        ))
    df = pd.DataFrame(rows)
    df["delta_%"] = (1 - df["MAE_recon"] / df["MAE_base"]) * 100
    return df