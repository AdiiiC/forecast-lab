"""Optuna-driven hyperparameter optimisation.

Enhancements over the original single-objective version:
  * Multi-objective Pareto front (MAE + Winkler) via NSGA-II sampler
  * MedianPruner integration — kill unpromising trials mid-fold
  * Config export — best params written to ``out_dir/best_hparams.yaml``
  * Backwards-compatible ``tune()`` entry-point still returns a dict
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import mae, winkler_score, coverage as cov_fn
from .models import build

ObjectiveMode = Literal["single", "pareto"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal objective factories
# ──────────────────────────────────────────────────────────────────────────────

def _single_objective(base_spec, y, season_length, bt_cfg, alpha, search_space):
    def objective(trial):
        spec = _sample_spec(trial, base_spec, search_space)
        folds = _run_backtest(spec, y, season_length, bt_cfg, alpha, trial)
        if folds is None:
            raise optuna.TrialPruned()
        y_true = np.concatenate([f.y_true for f in folds])
        y_pred = np.concatenate([f.y_pred for f in folds])
        return mae(y_true, y_pred)
    return objective


def _pareto_objective(base_spec, y, season_length, bt_cfg, alpha, search_space):
    """Returns (MAE, Winkler) for NSGA-II multi-objective optimisation."""
    def objective(trial):
        spec = _sample_spec(trial, base_spec, search_space)
        folds = _run_backtest(spec, y, season_length, bt_cfg, alpha, trial)
        if folds is None:
            raise optuna.TrialPruned()
        y_true = np.concatenate([f.y_true for f in folds])
        y_pred = np.concatenate([f.y_pred for f in folds])
        lo = np.concatenate([f.lo for f in folds])
        hi = np.concatenate([f.hi for f in folds])
        valid = np.isfinite(lo) & np.isfinite(hi)
        m_mae = mae(y_true, y_pred)
        m_winkler = (winkler_score(y_true[valid], lo[valid], hi[valid], alpha)
                     if valid.any() else m_mae * 2)
        return m_mae, m_winkler
    return objective


def _sample_spec(trial, base_spec, search_space):
    import optuna  # noqa: F401 — imported lazily so the module is importable without optuna
    spec = copy.deepcopy(base_spec)
    for k, dom in search_space.items():
        kind = dom["type"]
        if kind == "int":
            spec[k] = trial.suggest_int(k, dom["low"], dom["high"],
                                         step=dom.get("step", 1))
        elif kind == "float":
            spec[k] = trial.suggest_float(k, dom["low"], dom["high"],
                                           log=dom.get("log", False))
        elif kind == "cat":
            spec[k] = trial.suggest_categorical(k, dom["choices"])
    return spec


def _run_backtest(spec, y, season_length, bt_cfg, alpha, trial):
    """Run backtest; report intermediate values for pruning."""
    import optuna
    model = build(spec, season_length=season_length)
    folds = backtest(
        model, y,
        horizon=bt_cfg["horizon"],
        n_folds=bt_cfg["n_folds"],
        min_train_size=bt_cfg["min_train_size"],
        stride=bt_cfg["stride"],
        mode=bt_cfg["mode"],
        alpha=alpha,
        desc=f"trial-{trial.number}",
    )
    # Report per-fold intermediate value for the pruner
    for step, fold in enumerate(folds):
        intermediate = float(mae(fold.y_true, fold.y_pred))
        trial.report(intermediate, step=step)
        if trial.should_prune():
            return None
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def tune(
    base_spec: dict,
    y: pd.Series,
    season_length: int,
    bt_cfg: dict,
    alpha: float,
    search_space: dict,
    n_trials: int = 25,
    seed: int = 0,
    mlflow_run=None,
    mode: ObjectiveMode = "single",
    out_dir: str | Path | None = None,
) -> dict:
    """Run HPO and return best param dict.

    Parameters
    ----------
    mode    : 'single' (minimise MAE) or 'pareto' (minimise MAE + Winkler)
    out_dir : if set, writes ``best_hparams.yaml`` into this directory
    """
    import optuna  # noqa: F401
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    if mode == "pareto":
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=seed),
            pruner=pruner,
        )
        obj = _pareto_objective(base_spec, y, season_length, bt_cfg, alpha, search_space)
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner,
        )
        obj = _single_objective(base_spec, y, season_length, bt_cfg, alpha, search_space)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    # Pick best params
    if mode == "pareto":
        # Pareto-front: pick trial with lowest MAE (first objective) on the front
        pareto = study.best_trials
        best_trial = min(pareto, key=lambda t: t.values[0])
    else:
        best_trial = study.best_trial

    best = {**base_spec, **best_trial.params}

    # MLflow logging
    if mlflow_run is not None:
        key = base_spec.get("name", "model")
        mlflow_run.log_metric(f"hpo_best_mae.{key}", best_trial.values[0])
        if mode == "pareto":
            mlflow_run.log_metric(f"hpo_best_winkler.{key}", best_trial.values[1])
        mlflow_run.log_dict(best, f"hpo_best_{key}.json")

    # Config export
    if out_dir is not None:
        import yaml  # type: ignore
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_name = base_spec.get("name", "model")
        with open(out_path / f"best_hparams_{model_name}.yaml", "w") as fh:
            yaml.safe_dump(best, fh, default_flow_style=False)

    return best


def tune_all(
    specs: list[dict],
    y: pd.Series,
    season_length: int,
    bt_cfg: dict,
    alpha: float,
    search_spaces: dict[str, dict],
    n_trials: int = 25,
    seed: int = 0,
    out_dir: str | Path | None = None,
    mode: ObjectiveMode = "single",
) -> dict[str, dict]:
    """Tune every spec that has an entry in ``search_spaces``; return mapping name→best_params."""
    results = {}
    for spec in specs:
        name = spec.get("name", "")
        ss = search_spaces.get(name)
        if ss is None:
            continue
        best = tune(spec, y, season_length, bt_cfg, alpha, ss,
                    n_trials=n_trials, seed=seed, out_dir=out_dir, mode=mode)
        results[name] = best
    return results
