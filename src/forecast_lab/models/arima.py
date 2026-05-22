"""SARIMAX wrapper with native analytic prediction intervals."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .base import BaseModel, Forecast


class ARIMAModel(BaseModel):
    name = "arima"
    produces_intervals = True
    accepts_covariates = False

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)

    def fit(self, y: pd.Series, cov=None) -> "ARIMAModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        f = self.res_.get_forecast(steps=horizon)
        ci = f.conf_int(alpha=alpha)
        return Forecast(
            mean=np.asarray(f.predicted_mean),
            lo=np.asarray(ci.iloc[:, 0]),
            hi=np.asarray(ci.iloc[:, 1]),
        )