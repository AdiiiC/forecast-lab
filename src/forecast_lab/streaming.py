"""Incremental refit interface.

Models that support warm-starts (LightGBM via `init_model`, statsmodels via
`extend`, PyTorch by gradient steps) override `partial_fit`. Default behavior
is a full refit on the most recent rolling window — still useful, but expensive.
"""
from __future__ import annotations
import copy
import pandas as pd
from .models.base import BaseModel


class StreamingForecaster:
    def __init__(self, model: BaseModel, window: int):
        self.model = model
        self.window = window
        self.buffer: pd.Series | None = None

    def update(self, new_obs: pd.Series) -> "StreamingForecaster":
        if self.buffer is None:
            self.buffer = new_obs.copy()
        else:
            self.buffer = pd.concat([self.buffer, new_obs])
        self.buffer = self.buffer.iloc[-self.window:]
        # try partial_fit, else fall back to a fresh fit on the buffer
        m = self.model
        if hasattr(m, "partial_fit"):
            m.partial_fit(new_obs)
        else:
            self.model = copy.deepcopy(m).fit(self.buffer)
        return self

    def predict(self, horizon: int, alpha: float = 0.1):
        return self.model.predict(horizon, alpha=alpha)