from __future__ import annotations
import pandas as pd

class SnowflakeConnector:
    def __init__(self, account: str, user: str, password: str, warehouse: str,
                 database: str, schema: str, table: str, **kw):
        import snowflake.connector  # type: ignore
        self.table = f"{database}.{schema}.{table}"
        self.conn = snowflake.connector.connect(
            account=account, user=user, password=password,
            warehouse=warehouse, database=database, schema=schema, **kw)

    def read(self, *, where: dict | None = None) -> pd.DataFrame:
        clause = ""
        params = []
        if where:
            clause = " WHERE " + " AND ".join(f"{k}=%s" for k in where)
            params = list(where.values())
        return pd.read_sql(f"SELECT * FROM {self.table}{clause}", self.conn, params=params)

    def write(self, df: pd.DataFrame, mode: str = "append") -> None:
        from snowflake.connector.pandas_tools import write_pandas  # type: ignore
        if mode == "overwrite":
            self.conn.cursor().execute(f"TRUNCATE TABLE {self.table}")
        write_pandas(self.conn, df, self.table.split(".")[-1])