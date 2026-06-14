"""MLflow tracking wrapper. No-ops cleanly if mlflow isn't installed."""
from __future__ import annotations
import hashlib
import json
import os
import re
from contextlib import contextmanager
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
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","sqlite:///mlflow.db"))
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