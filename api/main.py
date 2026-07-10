"""Read-only + async-run API over walk-forward backtest run artifacts.

New endpoints
-------------
POST /api/jobs            — trigger a backtest run asynchronously
GET  /api/jobs/{job_id}   — poll async job status + progress
GET  /api/runs/{a}/compare/{b}  — leaderboard diff between two runs
GET  /health              — liveness probe

Security
--------
``FL_API_KEY`` environment variable — if set, all mutating routes
(POST) require the header ``X-API-Key: <value>``. Read routes remain
public so the dashboard can operate without credentials.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticModel

RUNS_DIR = Path(os.environ.get("FL_RUNS_DIR", "runs")).resolve()
_API_KEY  = os.environ.get("FL_API_KEY")          # None = no auth required

app = FastAPI(title="Forecast Lab API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Auth helper
# ──────────────────────────────────────────────────────────────────────────────

def _require_key(x_api_key: str | None):
    if _API_KEY is None:
        return          # auth disabled
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_dir(run: str) -> Path:
    candidate = (RUNS_DIR / run).resolve()
    if candidate.parent != RUNS_DIR or not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run}' not found.")
    return candidate


def _safe_image(directory: Path, filename: str) -> Path:
    if not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG assets are served.")
    target = (directory / filename).resolve()
    if target.parent != directory.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return target


# ──────────────────────────────────────────────────────────────────────────────
# Async job registry
# ──────────────────────────────────────────────────────────────────────────────

_JOBS: dict[str, dict] = {}    # job_id → {status, config, started, finished, log}
_JOBS_LOCK = threading.Lock()


class RunRequest(PydanticModel):
    config: str                    # path to YAML config file
    track: bool = False            # pass --track to the CLI


def _run_job(job_id: str, config: str, track: bool):
    cmd = ["python", "-m", "forecast_lab.cli", "--config", config]
    if track:
        cmd.append("--track")
    log_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:              # type: ignore[union-attr]
            log_lines.append(line.rstrip())
            with _JOBS_LOCK:
                _JOBS[job_id]["log"] = log_lines[-200:]   # keep last 200 lines
        proc.wait()
        exit_code = proc.returncode
    except Exception as exc:
        exit_code = -1
        log_lines.append(f"error: {exc}")

    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "success" if exit_code == 0 else "failed"
        _JOBS[job_id]["exit_code"] = exit_code
        _JOBS[job_id]["finished"] = time.time()
        _JOBS[job_id]["log"] = log_lines[-200:]


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "runs_dir": str(RUNS_DIR)}


@app.get("/api/runs")
def list_runs() -> dict:
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "metrics.csv").exists()
    )
    return {"runs": runs}


@app.get("/api/runs/{run}/metrics")
def get_metrics(run: str) -> dict:
    path = _run_dir(run) / "metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.csv not found.")
    try:
        df = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read metrics: {exc}")
    rows = []
    for model, record in zip(df.index, df.to_dict(orient="records")):
        clean = {"model": str(model)}
        for key, value in record.items():
            clean[str(key)] = None if (isinstance(value, float) and math.isnan(value)) else value
        rows.append(clean)
    return {"columns": list(df.columns), "rows": rows}


@app.get("/api/runs/{run_a}/compare/{run_b}")
def compare_runs(run_a: str, run_b: str) -> dict:
    """Return a diff of two runs' leaderboards (shared metrics, delta values)."""
    def _load(run: str) -> pd.DataFrame:
        path = _run_dir(run) / "metrics.csv"
        if not path.exists():
            raise HTTPException(404, f"{run}/metrics.csv not found")
        return pd.read_csv(path, index_col=0)

    df_a, df_b = _load(run_a), _load(run_b)
    shared_metrics = [c for c in df_a.columns if c in df_b.columns
                      and df_a[c].dtype != object and df_b[c].dtype != object]
    shared_models  = df_a.index.intersection(df_b.index)

    diff_rows = []
    for model in shared_models:
        row = {"model": model}
        for metric in shared_metrics:
            va = df_a.loc[model, metric]
            vb = df_b.loc[model, metric]
            if pd.notna(va) and pd.notna(vb):
                row[f"{metric}_{run_a}"] = round(float(va), 5)
                row[f"{metric}_{run_b}"] = round(float(vb), 5)
                row[f"{metric}_delta"]   = round(float(vb) - float(va), 5)
        diff_rows.append(row)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "shared_models": list(shared_models),
        "shared_metrics": shared_metrics,
        "diff": diff_rows,
    }


@app.get("/api/runs/{run}/plots")
def list_plots(run: str) -> dict:
    directory = _run_dir(run) / "plots"
    if not directory.exists():
        return {"plots": []}
    return {"plots": [{"name": f.stem, "file": f.name}
                      for f in sorted(directory.glob("*.png"))]}


@app.get("/api/runs/{run}/plots/{filename}")
def get_plot(run: str, filename: str) -> FileResponse:
    return FileResponse(_safe_image(_run_dir(run) / "plots", filename),
                        media_type="image/png")


@app.get("/api/runs/{run}/diagnostics")
def list_diagnostics(run: str) -> dict:
    directory = _run_dir(run) / "diagnostics"
    if not directory.exists():
        return {"diagnostics": []}
    return {"diagnostics": [{"name": f.stem, "file": f.name}
                              for f in sorted(directory.glob("*.png"))]}


@app.get("/api/runs/{run}/diagnostics/{filename}")
def get_diagnostic(run: str, filename: str) -> FileResponse:
    return FileResponse(_safe_image(_run_dir(run) / "diagnostics", filename),
                        media_type="image/png")


@app.get("/api/runs/{run}/decisions")
def get_decisions(run: str) -> dict:
    path = _run_dir(run) / "decisions.json"
    if not path.exists():
        return {"exists": False, "decisions": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse decisions: {exc}")
    return {"exists": True, "decisions": data}


@app.get("/api/runs/{run}/reliability")
def get_reliability(run: str) -> dict:
    path = _run_dir(run) / "reliability.csv"
    if not path.exists():
        return {"exists": False, "rows": []}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read reliability: {exc}")
    rows = [
        {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
         for k, v in rec.items()}
        for rec in df.to_dict(orient="records")
    ]
    return {"exists": True, "rows": rows}


@app.get("/api/runs/{run}/forecasts")
def get_forecasts(run: str) -> dict:
    path = _run_dir(run) / "forecasts.json"
    if not path.exists():
        return {"exists": False, "alpha": None, "series": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse forecasts: {exc}")
    return {"exists": True, "alpha": data.get("alpha"), "series": data.get("series", {})}


# ──────────────────────────────────────────────────────────────────────────────
# Async job endpoints (mutating — require API key when FL_API_KEY is set)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/jobs", status_code=202)
def create_job(req: RunRequest, x_api_key: str | None = Header(default=None)) -> dict:
    """Trigger a backtest run asynchronously.

    Returns a ``job_id`` which can be polled via ``GET /api/jobs/{job_id}``.
    """
    _require_key(x_api_key)
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "config": req.config,
            "track": req.track,
            "started": time.time(),
            "finished": None,
            "exit_code": None,
            "log": [],
        }
    t = threading.Thread(target=_run_job, args=(job_id, req.config, req.track),
                         daemon=True)
    t.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll async job status, progress log, and exit code."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    elapsed = None
    if job["started"]:
        end = job["finished"] or time.time()
        elapsed = round(end - job["started"], 1)
    return {
        "job_id": job_id,
        "status": job["status"],
        "config": job["config"],
        "elapsed_seconds": elapsed,
        "exit_code": job["exit_code"],
        "log_tail": job["log"][-50:],
    }


@app.get("/api/jobs")
def list_jobs() -> dict:
    """List all known jobs and their statuses."""
    with _JOBS_LOCK:
        summary = [
            {"job_id": jid, "status": j["status"], "config": j["config"]}
            for jid, j in _JOBS.items()
        ]
    return {"jobs": summary}



def _run_dir(run: str) -> Path:
    """Resolve a run directory, rejecting anything outside ``RUNS_DIR``."""
    candidate = (RUNS_DIR / run).resolve()
    if candidate.parent != RUNS_DIR or not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run}' not found.")
    return candidate


def _safe_image(directory: Path, filename: str) -> Path:
    """Resolve an image file inside ``directory`` with no path traversal."""
    if not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG assets are served.")
    target = (directory / filename).resolve()
    if target.parent != directory.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return target


@app.get("/api/runs")
def list_runs() -> dict:
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = sorted(
        d.name
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "metrics.csv").exists()
    )
    return {"runs": runs}


@app.get("/api/runs/{run}/metrics")
def get_metrics(run: str) -> dict:
    path = _run_dir(run) / "metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.csv not found.")
    try:
        df = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read metrics: {exc}")

    columns = [str(c) for c in df.columns]
    rows = []
    for model, record in zip(df.index, df.to_dict(orient="records")):
        clean = {"model": str(model)}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                clean[str(key)] = None
            else:
                clean[str(key)] = value
        rows.append(clean)
    return {"columns": columns, "rows": rows}


@app.get("/api/runs/{run}/plots")
def list_plots(run: str) -> dict:
    directory = _run_dir(run) / "plots"
    if not directory.exists():
        return {"plots": []}
    plots = [
        {"name": f.stem, "file": f.name}
        for f in sorted(directory.glob("*.png"))
    ]
    return {"plots": plots}


@app.get("/api/runs/{run}/plots/{filename}")
def get_plot(run: str, filename: str) -> FileResponse:
    directory = _run_dir(run) / "plots"
    return FileResponse(_safe_image(directory, filename), media_type="image/png")


@app.get("/api/runs/{run}/diagnostics")
def list_diagnostics(run: str) -> dict:
    directory = _run_dir(run) / "diagnostics"
    if not directory.exists():
        return {"diagnostics": []}
    imgs = [
        {"name": f.stem, "file": f.name}
        for f in sorted(directory.glob("*.png"))
    ]
    return {"diagnostics": imgs}


@app.get("/api/runs/{run}/diagnostics/{filename}")
def get_diagnostic(run: str, filename: str) -> FileResponse:
    directory = _run_dir(run) / "diagnostics"
    return FileResponse(_safe_image(directory, filename), media_type="image/png")


@app.get("/api/runs/{run}/decisions")
def get_decisions(run: str) -> dict:
    path = _run_dir(run) / "decisions.json"
    if not path.exists():
        return {"exists": False, "decisions": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse decisions: {exc}")
    return {"exists": True, "decisions": data}


@app.get("/api/runs/{run}/reliability")
def get_reliability(run: str) -> dict:
    path = _run_dir(run) / "reliability.csv"
    if not path.exists():
        return {"exists": False, "rows": []}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read reliability: {exc}")

    rows = []
    for record in df.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                clean[str(key)] = None
            else:
                clean[str(key)] = value
        rows.append(clean)
    return {"exists": True, "rows": rows}


@app.get("/api/runs/{run}/forecasts")
def get_forecasts(run: str) -> dict:
    path = _run_dir(run) / "forecasts.json"
    if not path.exists():
        return {"exists": False, "alpha": None, "series": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse forecasts: {exc}")
    return {
        "exists": True,
        "alpha": data.get("alpha"),
        "series": data.get("series", {}),
    }

