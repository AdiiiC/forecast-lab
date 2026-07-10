"""Online best-of-N model selection.

Maintains a rolling-MAE scoreboard across folds and always delegates
to the current champion. After each fold the scoreboard is updated;
the champion may change over time.

This makes it easy to hedge across a diverse pool (e.g. statistical +
neural) without committing up front.
"""
from __future__ import annotations

import copy
import inspect

import numpy as np
import pandas as pd

from .base import BaseModel, Forecast


class BestOfN(BaseModel):
    """Online champion selection from a pool of models.

    Parameters
    ----------
    models         : candidate pool (list of BaseModel)
    window         : number of recent observations over which rolling
                     MAE is tracked; ``None`` = expanding (all folds)
    burn_in        : number of predictions before selection activates;
                     before burn-in all models are averaged
    """

    produces_intervals = True

    def __init__(
        self,
        models: list[BaseModel],
        window: int | None = 168,
        burn_in: int = 0,
    ):
        self.models = models
        self.window = window
        self.burn_in = burn_in
        names = "|".join(getattr(m, "name", type(m).__name__) for m in models)
        self.name = f"best_of_n({names})"

    # ------------------------------------------------------------------
    def fit(self, y: pd.Series, cov=None) -> "BestOfN":
        self.models_: list[BaseModel] = []
        for m in self.models:
            mc = copy.deepcopy(m)
            sig = inspect.signature(mc.fit)
            if "cov" in sig.parameters and cov is not None:
                mc.fit(y, cov=cov)
            else:
                mc.fit(y)
            self.models_.append(mc)

        # Rolling MAE buffers: list of deque-like arrays per model
        self._error_history: list[list[float]] = [[] for _ in self.models_]
        self._n_obs: int = 0
        return self

    def _champion_idx(self) -> int:
        if self._n_obs < self.burn_in or all(len(h) == 0 for h in self._error_history):
            return -1  # -1 means "average"
        maes = []
        for hist in self._error_history:
            window_hist = hist[-self.window :] if self.window else hist
            maes.append(np.mean(window_hist) if window_hist else np.inf)
        return int(np.argmin(maes))

    def update(self, y_true: np.ndarray, forecasts: list[Forecast]) -> None:
        """Record prediction errors after new observations arrive."""
        for i, fc in enumerate(forecasts):
            h = min(len(y_true), len(fc.mean))
            self._error_history[i].extend(np.abs(y_true[:h] - fc.mean[:h]).tolist())
        self._n_obs += len(y_true)

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        champ = self._champion_idx()

        if champ >= 0:
            m = self.models_[champ]
            sig = inspect.signature(m.predict)
            fc: Forecast = (
                m.predict(horizon, alpha=alpha, cov=cov)
                if "cov" in sig.parameters and cov is not None
                else m.predict(horizon, alpha=alpha)
            )
            fc.meta["champion"] = getattr(m, "name", f"model_{champ}")
            fc.meta["champion_idx"] = champ
            return fc

        # Burn-in: return equal-weight average
        all_fc = []
        for m in self.models_:
            sig = inspect.signature(m.predict)
            all_fc.append(
                m.predict(horizon, alpha=alpha, cov=cov)
                if "cov" in sig.parameters and cov is not None
                else m.predict(horizon, alpha=alpha)
            )

        mean = np.mean([f.mean for f in all_fc], axis=0)
        has_lo = all(f.lo is not None for f in all_fc)
        lo = np.mean([f.lo for f in all_fc], axis=0) if has_lo else None
        hi = np.mean([f.hi for f in all_fc], axis=0) if has_lo else None
        return Forecast(mean=mean, lo=lo, hi=hi,
                        meta={"champion": "burn-in-average"})

    # scoreboard property — useful for dashboards / logging
    @property
    def scoreboard(self) -> dict[str, float]:
        result = {}
        for i, (m, hist) in enumerate(zip(self.models_, self._error_history)):
            window_hist = hist[-self.window :] if self.window else hist
            name = getattr(m, "name", f"model_{i}")
            result[name] = float(np.mean(window_hist)) if window_hist else float("inf")
        return result
