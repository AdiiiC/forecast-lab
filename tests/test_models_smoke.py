import numpy as np, pandas as pd
from forecast_lab.models import build

def _series():
    idx = pd.date_range("2024-01-01", periods=1000, freq="H")
    t = np.arange(1000)
    return pd.Series(10 + 5*np.sin(2*np.pi*t/24) + np.random.default_rng(0).normal(size=1000),
                     index=idx)

def test_naive_runs():
    m = build({"name": "seasonal_naive"}, season_length=24).fit(_series())
    fc = m.predict(24)
    assert fc.mean.shape == (24,)
    assert (fc.hi >= fc.lo).all()

def test_lightgbm_runs():
    m = build({"name": "lightgbm", "lags": [1, 24], "n_estimators": 50},
              season_length=24).fit(_series())
    fc = m.predict(24)
    assert fc.mean.shape == (24,)