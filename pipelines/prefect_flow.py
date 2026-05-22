from __future__ import annotations
from prefect import flow, task  # type: ignore
from prefect.schedules import CronSchedule  # type: ignore

@task(retries=2, retry_delay_seconds=600)
def train(config: str):
    import subprocess
    subprocess.check_call(["python", "-m", "forecast_lab.cli",
                           "--config", config, "--track"])

@task
def score():
    import requests, pandas as pd, json
    df = pd.read_parquet("/data/latest.parquet").tail(720)
    r = requests.post("http://forecast-lab:8000/forecast",
                      json={"history": df["y"].tolist(), "horizon": 24}).json()
    return r


@flow(name="forecast-lab-daily")
def daily_flow(config: str = "/opt/configs/energy_v2.yaml"):
    train(config)
    return score()


if __name__ == "__main__":
    daily_flow.serve(name="prod", schedule=CronSchedule(cron="0 2 * * *"))