"""MLflow tracking wrapper. No-ops cleanly if mlflow isn't installed.

Enhancements
------------
* ``save_artifact``    — pickle a trained model alongside the run artifacts
* ``save_run_config``  — write config.yaml snapshot into the run directory
* ``load_artifact``    — reload a pickled model from a run directory
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import mlflow  # type: ignore
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False


def repro_hash(cfg: dict) -> str:
    """Stable hash of the resolved config — pin this to a model run."""
    b = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:12]


@contextmanager
def run(experiment: str, run_name: str, cfg: dict, enabled: bool = True):
    if not (enabled and _HAS_MLFLOW):
        yield _NullRun()
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.log_params({"_repro_hash": repro_hash(cfg)})
        _log_flat(cfg, prefix="cfg")
        yield _MLflowRun(r)


def _log_flat(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _log_flat(v, key)
        elif isinstance(v, (list, tuple)):
            mlflow.log_param(key, json.dumps(v))
        else:
            mlflow.log_param(key, v)


# ──────────────────────────────────────────────────────────────────────────────
# Artifact persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_artifact(run_dir: str | Path, model: Any, name: str = "model") -> Path:
    """Pickle ``model`` into ``run_dir/artefacts/<name>.pkl``.

    The artefact subdirectory is created if it does not exist.

    Returns the path where the model was saved.
    """
    out = Path(run_dir) / "artefacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifact(run_dir: str | Path, name: str = "model") -> Any:
    """Load a pickled artefact from ``run_dir/artefacts/<name>.pkl``."""
    path = Path(run_dir) / "artefacts" / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Config snapshot
# ──────────────────────────────────────────────────────────────────────────────

def save_run_config(run_dir: str | Path, cfg: dict, name: str = "config") -> Path:
    """Write the resolved config as YAML into ``run_dir/<name>.yaml``.

    This creates a reproducibility snapshot so any run can be re-executed
    from its directory alone.
    """
    import yaml  # type: ignore
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
    return path


def load_run_config(run_dir: str | Path, name: str = "config") -> dict:
    """Load a YAML config snapshot from a run directory."""
    import yaml  # type: ignore
    path = Path(run_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Run wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _NullRun:
    def log_metric(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def log_dict(self, *a, **k): pass


class _MLflowRun:
    def __init__(self, r): self.r = r

    def log_metric(self, k, v, step=None):
        safe_key = re.sub(r"[^a-zA-Z0-9_.\-/ ]", "_", k)
        mlflow.log_metric(safe_key, float(v), step=step)

    def log_metrics(self, d: dict[str, Any], step=None):
        safe_metrics = {
            re.sub(r"[^a-zA-Z0-9_.\-/ ]", "_", k): float(v)
            for k, v in d.items()
            if v is not None
        }
        mlflow.log_metrics(safe_metrics, step=step)

    def log_artifact(self, path):
        mlflow.log_artifact(str(path))

    def log_dict(self, d, name):
        mlflow.log_dict(d, name)



def repro_hash(cfg: dict) -> str:
    """Stable hash of the resolved config — pin this to a model run."""
    b = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:12]


@contextmanager
def run(experiment: str, run_name: str, cfg: dict, enabled: bool = True):
    if not (enabled and _HAS_MLFLOW):
        yield _NullRun()
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.log_params({"_repro_hash": repro_hash(cfg)})
        _log_flat(cfg, prefix="cfg")
        yield _MLflowRun(r)


def _log_flat(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _log_flat(v, key)
        elif isinstance(v, (list, tuple)):
            mlflow.log_param(key, json.dumps(v))
        else:
            mlflow.log_param(key, v)


class _NullRun:
    def log_metric(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def log_dict(self, *a, **k): pass


class _MLflowRun:
    def __init__(self, r): self.r = r
    def log_metric(self, k, v, step=None):
        safe_key = re.sub(r"[^a-zA-Z0-9_.\-/ ]", "_", k)
        mlflow.log_metric(safe_key, float(v), step=step)
    def log_metrics(self, d: dict[str, Any], step=None):
        safe_metrics = {
            re.sub(r"[^a-zA-Z0-9_.\-/ ]", "_", k): float(v)
            for k, v in d.items()
            if v is not None
        }
        mlflow.log_metrics(safe_metrics, step=step)
    def log_artifact(self, path):     mlflow.log_artifact(str(path))
    def log_dict(self, d, name):      mlflow.log_dict(d, name)