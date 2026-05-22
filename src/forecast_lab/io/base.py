from __future__ import annotations
from typing import Protocol
import pandas as pd

class Connector(Protocol):
    def read(self, *, where: dict | None = None) -> pd.DataFrame: ...
    def write(self, df: pd.DataFrame, *, mode: str = "append") -> None: ...