from __future__ import annotations
import pandas as pd

class BigQueryConnector:
    def __init__(self, project: str, dataset: str, table: str):
        from google.cloud import bigquery  # type: ignore
        self.client = bigquery.Client(project=project)
        self.table_id = f"{project}.{dataset}.{table}"

    def read(self, *, where: dict | None = None) -> pd.DataFrame:
        clause = ""
        if where:
            clause = " WHERE " + " AND ".join(f"{k}=@{k}" for k in where)
        return self.client.query(f"SELECT * FROM `{self.table_id}`{clause}").to_dataframe()

    def write(self, df: pd.DataFrame, mode: str = "append") -> None:
        from google.cloud import bigquery  # type: ignore
        cfg = bigquery.LoadJobConfig(
            write_disposition=("WRITE_TRUNCATE" if mode == "overwrite"
                               else "WRITE_APPEND"))
        self.client.load_table_from_dataframe(df, self.table_id, job_config=cfg).result()