# forecast-lab

> A forecasting platform that refuses to lie to you.

Most "forecasting projects" skip the baseline and look impressive but mean
nothing. `forecast-lab` is built on the opposite principle: **every model is
graded against a seasonal-naive baseline under a walk-forward backtest, with
Diebold–Mariano significance tests and calibrated prediction intervals**. If
nothing beats seasonal-naive, the CLI says so out loud.

---

## What's in the box

**Models — three families, ten architectures**

| Family | Models |
|---|---|
| Baseline | `SeasonalNaive` |
| Classical | `SARIMAX`, `Prophet` |
| Intermittent | `Croston`, `SBA`, `TSB`, `ADIDA` |
| ML | `LightGBM` (mean + quantile heads, lag + calendar + Fourier + holidays + exogenous covariates) |
| Deep learning | `N-BEATS`, `DeepAR`, `TFT`, `PatchTST`, `TiDE` |
| Zero-shot foundation | `Chronos`, `TimesFM` |

**Evaluation — research-grade honesty**

- Walk-forward backtest (rolling / expanding origin, configurable horizon, stride, folds)
- Calibrated prediction intervals: native (Prophet, quantile-LGBM, MC-dropout)
  **or** distribution-free via **split conformal**, **ACI**, or **EnbPI** wrappers
- Proper probabilistic scores: **CRPS**, energy score, quantile loss, Winkler interval score
- **MASE**, sMAPE, MAE, RMSE, empirical coverage, mean PI width
- **Diebold–Mariano** test vs. baseline with significance stars
- **PIT histograms**, reliability diagrams, sharpness-vs-coverage Pareto plots
- Asymmetric / newsvendor / dispatch-cost loss for business-aware comparison

**Engineering — production-shaped**

- Optuna HPO whose objective *is* walk-forward MAE — no train/val mismatch
- MLflow tracking with reproducibility hash, params, metrics, artifacts
- Parallel folds (`serial` / `thread` / `process` / `joblib` / `ray`) with
  GPU-aware safety fallback
- Pluggable I/O: Parquet, Snowflake, BigQuery, Kafka
- Schema validation + drift detection (PSI, KS)
- Streaming refits and an incremental forecaster interface
- FastAPI inference service + Dockerfile
- Airflow DAG and Prefect flow for daily train → score → monitor
- React + Vite dashboard (served by a read-only FastAPI artifact API) for
  interactive backtest inspection
- pytest + ruff + CI workflow

**Hierarchical forecasting**

- Build a summation matrix `S` from any (region, store, sku)-style key
- Reconcile with bottom-up, top-down, OLS, WLS-struct, **MinT-shrink**
- Per-level base vs. reconciled MAE so you can *see* where reconciliation helps

**Robustness layer**

- Kalman / seasonal imputation for missing values
- Hampel + STL-residual outlier detection (with flags)
- `ruptures`-based changepoint detection (CUSUM fallback)
- Newsvendor, safety-stock, and dispatch-threshold decision rules that convert
  forecasts + quantiles into procurement / dispatch policies

---

## Quickstart

```bash
git clone <repo> && cd forecast-lab
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Smoke test
PYTHONPATH=src python -m forecast_lab.cli --config configs/energy.yaml

# Full v2 run: Optuna HPO + MLflow + every model family
PYTHONPATH=src python -m forecast_lab.cli \
    --config configs/energy_v2.yaml --tune --track

mlflow ui --backend-store-uri file:./mlruns          # browse runs
```

## Dashboard (React + FastAPI)

The dashboard is a React (Vite + TypeScript) single-page app backed by a
read-only FastAPI service that streams the `runs/<name>/` artifacts. Start both
processes from the project root:

```bash
# 1. Artifact API — must run from the project root so runs/ resolves
make api          # uvicorn api.main:app --port 8001

# 2. Web client — installs deps and starts Vite on http://localhost:5173
make web          # cd web && npm install && npm run dev
```

Vite proxies `/api` to the FastAPI service, so the browser talks to a single
origin. The API contains no modelling logic; the backtest pipeline remains the
single source of truth for how the artifacts are produced.

## Ready-made configs

| Config | What it exercises |
|---|---|
| `configs/energy.yaml`          | Minimal v1 run — 5 model families |
| `configs/energy_v2.yaml`       | + DeepAR, TFT, Optuna HPO, CRPS, DM test, MLflow |
| `configs/energy_cov.yaml`      | + PatchTST/TiDE/Chronos and exogenous covariates |
| `configs/energy_adaptive.yaml` | Adaptive conformal (ACI, EnbPI) wrappers |
| `configs/intermittent.yaml`    | Preprocessing + Croston-family + decision artifacts |
| `configs/retail_hier.yaml`     | Hierarchical reconciliation across region/store/sku |

## The honest verdict

At the end of every run, the CLI prints one of:

```
Beat seasonal-naive at p<0.05: lightgbm, tft, patchtst
```
or
```
Nothing beats seasonal-naive at p<0.05. That's the honest answer.
```

That line is the whole point of the project.

---

## Project layout

```
forecast-lab/
├── Makefile
├── pyproject.toml
├── requirements.txt
├── README.md
├── .github/workflows/ci.yml
├── configs/
│   ├── energy.yaml
│   ├── energy_v2.yaml
│   ├── energy_cov.yaml
│   ├── energy_adaptive.yaml
│   ├── intermittent.yaml
│   └── retail_hier.yaml
├── api/
│   └── main.py                      # read-only FastAPI artifact service
├── web/                             # React + Vite dashboard
│   ├── index.html
│   └── src/
├── pipelines/
│   ├── airflow_dag.py               # daily train → score → monitor
│   └── prefect_flow.py
├── tests/
│   ├── test_splits.py
│   ├── test_metrics.py
│   ├── test_models_smoke.py
│   └── test_reconciliation.py
└── src/forecast_lab/
    ├── __init__.py
    ├── cli.py                       # main entrypoint
    ├── cli_hier.py                  # hierarchical entrypoint
    │
    ├── data.py                      # dataset loaders (+ covariates)
    ├── data_hier.py                 # synthetic grouped retail dataset
    ├── features.py                  # lag + calendar + Fourier features
    ├── calendars.py                 # holidays + known-future calendar feats
    ├── covariates.py                # known_future vs. observed covariates
    │
    ├── metrics.py                   # MAE, RMSE, sMAPE, MASE, Winkler, NV cost
    ├── metrics_prob.py              # CRPS, energy score, quantile loss
    ├── stats_tests.py               # Diebold–Mariano, Giacomini–White
    ├── distributions.py             # Gaussian / StudentT / Quantile / Empirical
    ├── calibration.py               # PIT, reliability, sharpness diagnostics
    │
    ├── conformal.py                 # split-conformal wrapper
    ├── conformal_adaptive.py        # ACI + EnbPI wrappers
    ├── preprocessing.py             # Kalman, Hampel, STL, changepoints
    ├── decision.py                  # newsvendor / safety stock / dispatch
    │
    ├── backtest.py                  # walk-forward backtester
    ├── backtest_hier.py             # hierarchical walk-forward
    ├── parallel.py                  # joblib / ray / process pool executor
    ├── hierarchy.py                 # summation matrix S from group spec
    ├── reconciliation.py            # BU / TD / OLS / WLS / MinT-shrink
    │
    ├── streaming.py                 # incremental refit interface
    ├── schema.py                    # validation + PSI/KS drift detection
    ├── report.py                    # leaderboard + plots + DM stars
    ├── tuning.py                    # Optuna HPO over walk-forward
    ├── tracking.py                  # MLflow wrapper + repro hash
    │
    ├── io/
    │   ├── __init__.py
    │   ├── base.py                  # Connector protocol
    │   ├── parquet.py
    │   ├── snowflake.py
    │   ├── bigquery.py
    │   └── kafka.py
    │
    ├── serving/
    │   ├── __init__.py
    │   ├── app.py                   # FastAPI inference service
    │   └── Dockerfile
    │
    ├── monitoring/
    │   ├── __init__.py
    │   └── monitor.py               # residual + coverage drift alerts
    │
    └── models/
        ├── __init__.py              # REGISTRY + recursive build()
        ├── base.py                  # BaseModel, Forecast dataclass
        ├── naive.py                 # SeasonalNaive (the baseline)
        ├── arima.py                 # SARIMAX
        ├── prophet_model.py         # Prophet
        ├── lgbm.py                  # LightGBM (mean + quantile heads)
        ├── nbeats.py                # N-BEATS (MC-dropout intervals)
        ├── deepar.py                # DeepAR (Gaussian / Student-t)
        ├── tft.py                   # Temporal Fusion Transformer
        ├── patchtst.py              # PatchTST
        ├── tide.py                  # TiDE
        ├── chronos.py               # Chronos / TimesFM zero-shot adapter
        └── croston.py               # Croston, SBA, TSB, ADIDA
```

## Citation / acknowledgements

Implements ideas from: Hyndman & Athanasopoulos (FPP3), Wickramasuriya et al.
(MinT, 2019), Salinas et al. (DeepAR, 2020), Lim et al. (TFT, 2021), Nie et al.
(PatchTST, 2023), Das et al. (TiDE, 2023), Gibbs & Candès (ACI, 2021), Xu & Xie
(EnbPI, 2021), Ansari et al. (Chronos, 2024), Das et al. (TimesFM, 2024).

## License

MIT