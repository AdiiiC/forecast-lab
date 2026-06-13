"""FastAPI inference service.

Endpoints
---------
GET  /health                              service heartbeat
POST /forecast   {history, horizon}       point + interval forecast
POST /intervals  {history, horizon, alpha}
POST /reload     {model_uri}              hot-swap a serialized model

Models are loaded from MLflow if MLFLOW_TRACKING_URI is set and FL_MODEL_URI
points to an MLflow run, else from a pickled local model.
"""
from __future__ import annotations
import os
import pickle
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="forecast-lab", version="1.0")


class ForecastRequest(BaseModel):
    history: List[float] = Field(..., min_length=10)
    timestamps: List[str] | None = None
    horizon: int = 24
    alpha: float = 0.1


class ForecastResponse(BaseModel):
    mean: List[float]
    lo: List[float] | None
    hi: List[float] | None
    model: str


_MODEL = None
_MODEL_NAME = "unset"


def _load(uri: str):
    if uri.startswith("models:") or uri.startswith("runs:"):
        import mlflow  # type: ignore
        return mlflow.pyfunc.load_model(uri)
    with open(uri, "rb") as fh:
        return pickle.load(fh)


@app.on_event("startup")
def _startup():
    global _MODEL, _MODEL_NAME
    uri = os.environ.get("FL_MODEL_URI")
    if uri:
        _MODEL = _load(uri)
        _MODEL_NAME = uri


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL_NAME}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if _MODEL is None:
        raise HTTPException(503, "no model loaded; set FL_MODEL_URI or POST /reload")
    idx = (pd.to_datetime(req.timestamps) if req.timestamps
           else pd.date_range("2024-01-01", periods=len(req.history), freq="H"))
    y = pd.Series(req.history, index=idx)
    m = _MODEL
    m.fit(y)
    fc = m.predict(req.horizon, alpha=req.alpha)
    return ForecastResponse(
        mean=fc.mean.tolist(),
        lo=fc.lo.tolist() if fc.lo is not None else None,
        hi=fc.hi.tolist() if fc.hi is not None else None,
        model=_MODEL_NAME,
    )


@app.post("/reload")
def reload(payload: dict):
    global _MODEL, _MODEL_NAME
    uri = payload.get("model_uri")
    if not uri:
        raise HTTPException(400, "model_uri required")
    _MODEL = _load(uri)
    _MODEL_NAME = uri
    return {"status": "loaded", "model": _MODEL_NAME}