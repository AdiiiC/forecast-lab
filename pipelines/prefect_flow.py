"""Prefect flows for Forecast Lab.

Flows
-----
daily_flow          — retrain + score (original schedule)
retrain_flow        — triggered automatically when MonitorReport.retrain_needed
validate_data_flow  — schema + drift validation before any model fits
versioned_run_flow  — wrap any run with artifact versioning (unique hash)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

try:
    from prefect import flow, task              # type: ignore
    from prefect.schedules import CronSchedule  # type: ignore
    _HAS_PREFECT = True
except ImportError:
    def flow(fn=None, **_):
        return fn if fn else (lambda f: f)
    def task(fn=None, **_):
        return fn if fn else (lambda f: f)
    class CronSchedule:
        def __init__(self, cron): pass
    _HAS_PREFECT = False


@task(retries=2, retry_delay_seconds=600)
def train(config: str, track: bool = False):
    import subprocess
    cmd = ["python", "-m", "forecast_lab.cli", "--config", config]
    if track:
        cmd.append("--track")
    subprocess.check_call(cmd)


@task
def score(endpoint: str = "http://forecast-lab:8000", horizon: int = 24):
    import requests
    import pandas as pd
    df = pd.read_parquet("/data/latest.parquet").tail(720)
    r = requests.post(
        f"{endpoint}/forecast",
        json={"history": df["y"].tolist(), "horizon": horizon},
    ).json()
    return r


@flow(name="forecast-lab-daily")
def daily_flow(config: str = "/opt/configs/energy_v2.yaml"):
    train(config)
    return score()


# ── Data validation ──────────────────────────────────────────────────────────

@task
def validate_data(data_path: str, schema_spec: dict | None = None,
                  ref_path: str | None = None) -> dict:
    """Validate schema and optionally check for covariate drift."""
    import pandas as pd
    from forecast_lab.schema import validate, drift_report

    ext = Path(data_path).suffix.lower()
    df = pd.read_parquet(data_path) if ext == ".parquet" else pd.read_csv(data_path)

    result: dict = {"ok": True, "errors": [], "drift": {}}

    if schema_spec:
        sr = validate(df, schema_spec)
        result["ok"] = sr.ok
        result["errors"] = sr.errors

    if ref_path:
        ext_r = Path(ref_path).suffix.lower()
        ref_df = (pd.read_parquet(ref_path) if ext_r == ".parquet"
                  else pd.read_csv(ref_path))
        drift_df = drift_report(ref_df, df)
        result["drift"] = drift_df.to_dict(orient="records")
        flagged = drift_df[drift_df["flag"] != "ok"]
        if not flagged.empty:
            result["ok"] = False
            result["errors"].append(
                f"Drift detected in columns: {list(flagged['feature'])}"
            )
    return result


@flow(name="forecast-lab-validate")
def validate_data_flow(data_path: str, schema_spec: dict | None = None,
                       ref_path: str | None = None) -> dict:
    return validate_data(data_path, schema_spec=schema_spec, ref_path=ref_path)


# ── Automatic retraining flow ────────────────────────────────────────────────

@task
def check_drift(run_dir: str, ref_residuals_path: str,
                nominal_coverage: float = 0.9,
                mae_threshold: float | None = None) -> dict:
    """Evaluate MonitorReport for the latest production window."""
    import numpy as np
    from forecast_lab.monitoring.monitor import evaluate

    ref = np.load(ref_residuals_path)
    p = Path(run_dir)
    cur_r = p / "cur_residuals.npy"
    cur_i = p / "cur_in_interval.npy"

    if not cur_r.exists() or not cur_i.exists():
        return {"retrain_needed": False, "reason": "no current window data"}

    report = evaluate(ref, np.load(cur_r), np.load(cur_i).astype(bool),
                      nominal_coverage, mae_threshold=mae_threshold)
    return report.to_dict()


@task
def conditional_retrain(report: dict, config: str, track: bool = False) -> dict:
    if not report.get("retrain_needed", False):
        return {"action": "skipped", "reason": "no drift detected"}
    import subprocess
    cmd = ["python", "-m", "forecast_lab.cli", "--config", config]
    if track:
        cmd.append("--track")
    subprocess.check_call(cmd)
    return {"action": "retrained", "config": config}


@flow(name="forecast-lab-retrain")
def retrain_flow(run_dir: str, ref_residuals_path: str, config: str,
                 nominal_coverage: float = 0.9,
                 mae_threshold: float | None = None,
                 track: bool = False):
    report = check_drift(run_dir, ref_residuals_path,
                         nominal_coverage=nominal_coverage,
                         mae_threshold=mae_threshold)
    return conditional_retrain(report, config=config, track=track)


# ── Versioned run flow ───────────────────────────────────────────────────────

@task
def snapshot_run(run_dir: str, config_path: str) -> str:
    """Copy run artifacts into a versioned sub-directory; return version tag."""
    cfg_text = Path(config_path).read_text()
    version = hashlib.sha256(cfg_text.encode()).hexdigest()[:10]
    versioned_dir = Path(run_dir).parent / f"{Path(run_dir).name}_v{version}"

    if versioned_dir.exists():
        return version   # idempotent

    shutil.copytree(str(run_dir), str(versioned_dir))
    (versioned_dir / "VERSION").write_text(
        json.dumps({"version": version, "config": config_path,
                    "snapshotted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())}, indent=2)
    )
    return version


@flow(name="forecast-lab-versioned-run")
def versioned_run_flow(config: str, track: bool = False) -> dict:
    train(config, track=track)
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(config).read_text())
    run_name = cfg.get("name", "run")
    version = snapshot_run(f"runs/{run_name}", config)
    return {"run": run_name, "version": version}


if __name__ == "__main__":
    if _HAS_PREFECT:
        daily_flow.serve(name="prod", schedule=CronSchedule(cron="0 2 * * *"))
