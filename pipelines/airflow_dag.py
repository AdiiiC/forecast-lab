"""Daily batch scoring DAG."""
from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore


def _train(**ctx):
    import subprocess
    subprocess.check_call(["python", "-m", "forecast_lab.cli",
                           "--config", "/opt/configs/energy_v2.yaml",
                           "--track"])


def _score(**ctx):
    import requests, json, pandas as pd
    df = pd.read_parquet("/data/latest.parquet").tail(720)
    r = requests.post("http://forecast-lab:8000/forecast",
                      json={"history": df["y"].tolist(), "horizon": 24})
    r.raise_for_status()
    with open(f"/data/forecast_{ctx['ds']}.json", "w") as fh:
        json.dump(r.json(), fh)


def _monitor(**ctx):
    from forecast_lab.monitoring.monitor import evaluate
    # ... pull recent residuals from your store, call evaluate(...) ...
    pass


with DAG(
    dag_id="forecast_lab_daily",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
) as dag:
    train   = PythonOperator(task_id="train",   python_callable=_train)
    score   = PythonOperator(task_id="score",   python_callable=_score)
    monitor = PythonOperator(task_id="monitor", python_callable=_monitor)
    train >> score >> monitor