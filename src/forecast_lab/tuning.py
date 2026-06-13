"""Optuna-driven hyperparameter optimization with walk-forward CV as the objective."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import mae
from .models import build


def _objective_factory(base_spec: dict, y: pd.Series, season_length: int,
                       bt_cfg: dict, alpha: float, search_space: dict):
    def objective(trial):
        spec = copy.deepcopy(base_spec)
        for k, dom in search_space.items():
            kind = dom["type"]
            if kind == "int":
                spec[k] = trial.suggest_int(k, dom["low"], dom["high"], step=dom.get("step", 1))
            elif kind == "float":
                spec[k] = trial.suggest_float(k, dom["low"], dom["high"],
                                              log=dom.get("log", False))
            elif kind == "cat":
                spec[k] = trial.suggest_categorical(k, dom["choices"])
        model = build(spec, season_length=season_length)
        folds = backtest(model, y,
                         horizon=bt_cfg["horizon"], n_folds=bt_cfg["n_folds"],
                         min_train_size=bt_cfg["min_train_size"],
                         stride=bt_cfg["stride"], mode=bt_cfg["mode"],
                         alpha=alpha, desc=f"trial-{trial.number}")
        y_true = np.concatenate([f.y_true for f in folds])
        y_pred = np.concatenate([f.y_pred for f in folds])
        return mae(y_true, y_pred)
    return objective


def tune(base_spec: dict, y: pd.Series, season_length: int, bt_cfg: dict,
         alpha: float, search_space: dict, n_trials: int = 25,
         seed: int = 0, mlflow_run=None) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    obj = _objective_factory(base_spec, y, season_length, bt_cfg, alpha, search_space)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = {**base_spec, **study.best_params}
    if mlflow_run is not None:
        mlflow_run.log_metric(f"hpo_best_mae.{base_spec['name']}", study.best_value)
        mlflow_run.log_dict(best, f"hpo_best_{base_spec['name']}.json")
    return best