from __future__ import annotations
from pathlib import Path
import pandas as pd

class ParquetConnector:
    def __init__(self, path: str, partition_cols: list[str] | None = None):
        self.path = Path(path); self.partition_cols = partition_cols

    def read(self, *, where: dict | None = None) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        if where:
            for k, v in where.items():
                df = df[df[k] == v]
        return df

    def write(self, df: pd.DataFrame, mode: str = "append") -> None:
        if mode == "overwrite" and self.path.exists():
            if self.path.is_dir():
                import shutil; shutil.rmtree(self.path)
            else:
                self.path.unlink()
        df.to_parquet(self.path, partition_cols=self.partition_cols)