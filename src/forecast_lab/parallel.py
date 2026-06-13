"""Parallel walk-forward fold executor.

Backends: 'serial' | 'thread' | 'process' | 'joblib' | 'ray'. GPU-aware: DL
models (deepar/tft/patchtst/tide/nbeats) auto-fall-back to thread/serial to
avoid CUDA-fork issues; tabular/classical models use process pools.
"""
from __future__ import annotations
import copy
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtest import walk_forward_splits, FoldResult
from .covariates import Covariates


_DL_MODELS = {"nbeats", "deepar", "tft", "patchtst", "tide", "chronos"}
_STATSMODELS_MODELS = {"arima", "prophet", "aci", "enbpi"}  # Statsmodels memory-intensive; avoid process fork


def _safe_backend(model_name: str, requested: str) -> str:
    if requested in ("serial", "thread"):    return requested
    if model_name in _DL_MODELS:             return "thread"   # CUDA + fork is unsafe
    if model_name in _STATSMODELS_MODELS:    return "thread"   # Statsmodels Kalman smoother memory-intensive
    return requested


def _run_fold(args):
    import inspect
    model, train, test, train_cov, test_cov, horizon, alpha, i = args
    m = copy.deepcopy(model)
    if "cov" in inspect.signature(m.fit).parameters and train_cov is not None:
        m.fit(train, cov=train_cov)
    else:
        m.fit(train)
    if "cov" in inspect.signature(m.predict).parameters and test_cov is not None:
        fc = m.predict(horizon, alpha=alpha, cov=test_cov)
    else:
        fc = m.predict(horizon, alpha=alpha)
    return FoldResult(
        fold=i, train_end=train.index[-1],
        y_true=test.values, y_pred=fc.mean,
        lo=fc.lo if fc.lo is not None else np.full(horizon, np.nan),
        hi=fc.hi if fc.hi is not None else np.full(horizon, np.nan),
        train_tail=train.values[-max(len(train) // 4, horizon * 2):],
        samples=fc.samples, quantiles=fc.quantiles, q_levels=fc.q_levels,
    )


def parallel_backtest(model, y: pd.Series, *, horizon, n_folds, min_train_size,
                      stride, mode, alpha, cov: Covariates | None = None,
                      backend: str = "process", n_workers: int | None = None,
                      desc: str = ""):
    splits = list(walk_forward_splits(len(y), min_train_size, horizon,
                                      n_folds, stride, mode))
    cov = cov or Covariates()
    jobs = []
    for i, (te, ee) in enumerate(splits):
        tr_lo = 0 if mode == "expanding" else max(0, te - min_train_size)
        train = y.iloc[tr_lo:te]; test = y.iloc[te:ee]
        tc = cov.slice(train.index[0], train.index[-1]) if cov.has_any else None
        sc = cov.slice(test.index[0],  test.index[-1])  if cov.has_any else None
        jobs.append((model, train, test, tc, sc, horizon, alpha, i))

    backend = _safe_backend(getattr(model, "name", ""), backend)
    n_workers = n_workers or min(len(jobs), os.cpu_count() or 4)

    if backend == "serial":
        return [_run_fold(j) for j in tqdm(jobs, desc=desc, leave=False)]
    if backend == "ray":
        import ray
        if not ray.is_initialized(): ray.init(ignore_reinit_error=True)
        remote = ray.remote(_run_fold)
        return ray.get([remote.remote(j) for j in jobs])
    if backend == "joblib":
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_workers)(delayed(_run_fold)(j) for j in jobs)
    Exec = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with Exec(max_workers=n_workers) as ex:
        futures = [ex.submit(_run_fold, j) for j in jobs]
        out = [None] * len(jobs)
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=desc, leave=False):
            r = f.result(); out[r.fold] = r
        return out