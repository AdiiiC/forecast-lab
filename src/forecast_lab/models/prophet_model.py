"""Prophet wrapper with native posterior-sample prediction intervals."""
from __future__ import annotations
import logging
import pandas as pd
from .base import BaseModel, Forecast

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetModel(BaseModel):
    name = "prophet"
    produces_intervals = True
    accepts_covariates = False     # extra regressors could be wired via cov later

    def __init__(self, weekly: bool = True, daily: bool = True,
                 yearly: bool = False, interval_width: float = 0.9):
        self.kw = dict(
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            yearly_seasonality=yearly,
        )
        self.default_iw = float(interval_width)

    def fit(self, y: pd.Series, cov=None) -> "ProphetModel":
        from prophet import Prophet
        self.freq_ = pd.infer_freq(y.index) or "H"
        self.m_ = Prophet(interval_width=self.default_iw, **self.kw)
        self.m_.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        # Prophet's interval_width is fixed at fit time; re-set it per call.
        self.m_.interval_width = 1 - alpha
        future = self.m_.make_future_dataframe(
            periods=horizon, freq=self.freq_, include_history=False)
        fc = self.m_.predict(future)
        return Forecast(
            mean=fc["yhat"].to_numpy(),
            lo=fc["yhat_lower"].to_numpy(),
            hi=fc["yhat_upper"].to_numpy(),
        )