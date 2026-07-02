"""Read-only API over walk-forward backtest run artifacts.

This service exposes the same ``runs/<name>/`` artifacts the dashboard has
always read (``metrics.csv``, ``plots/``, ``diagnostics/``, ``decisions.json``)
so the web client can render them. It contains no modelling logic; the backtest
pipeline remains the single source of truth for how these files are produced.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

RUNS_DIR = Path(os.environ.get("FL_RUNS_DIR", "runs")).resolve()

app = FastAPI(title="Forecast Lab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


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

