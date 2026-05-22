"""Walk-forward backtester with covariate-aware fit/predict and no leakage."""
from __future__ import annotations
import copy
import inspect
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
from .models.base import BaseModel, Forecast
from .covariates import Covariates


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    train_tail: np.ndarray = field(repr=False)
    samples: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    q_levels: np.ndarray | None = None


def walk_forward_splits(n: int, min_train: int, horizon: int,
                        n_folds: int, stride: int, mode: str = "rolling"):
    """Yield (train_end_idx_exclusive, test_end_idx_exclusive)."""
    last_train_end = n - horizon
    first_train_end = max(min_train, last_train_end - (n_folds - 1) * stride)
    train_ends = list(range(first_train_end, last_train_end + 1, stride))[:n_folds]
    for te in train_ends:
        yield te, te + horizon


def _supports_cov(method) -> bool:
    return "cov" in inspect.signature(method).parameters


def backtest(model: BaseModel, y: pd.Series, *, horizon: int, n_folds: int,
             min_train_size: int, stride: int, mode: str, alpha: float,
             cov: Covariates | None = None, desc: str = ""):
    results: list[FoldResult] = []
    splits = list(walk_forward_splits(len(y), min_train_size, horizon,
                                      n_folds, stride, mode))
    cov = cov or Covariates()

    for i, (te, ee) in enumerate(tqdm(splits, desc=desc, leave=False)):
        tr_lo = 0 if mode == "expanding" else max(0, te - min_train_size)
        train = y.iloc[tr_lo:te]
        test  = y.iloc[te:ee]
        train_cov = cov.slice(train.index[0], train.index[-1]) if cov.has_any else None
        test_cov  = cov.slice(test.index[0],  test.index[-1])  if cov.has_any else None

        m = copy.deepcopy(model)
        if _supports_cov(m.fit) and train_cov is not None:
            m.fit(train, cov=train_cov)
        else:
            m.fit(train)
        if _supports_cov(m.predict) and test_cov is not None:
            fc: Forecast = m.predict(horizon, alpha=alpha, cov=test_cov)
        else:
            fc = m.predict(horizon, alpha=alpha)

        results.append(FoldResult(
            fold=i, train_end=train.index[-1],
            y_true=test.values, y_pred=fc.mean,
            lo=fc.lo if fc.lo is not None else np.full(horizon, np.nan),
            hi=fc.hi if fc.hi is not None else np.full(horizon, np.nan),
            train_tail=train.values[-max(min_train_size, horizon * 2):],
            samples=fc.samples, quantiles=fc.quantiles, q_levels=fc.q_levels,
        ))
    return results