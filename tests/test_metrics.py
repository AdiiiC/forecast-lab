import numpy as np
from forecast_lab.metrics import mae, mase, coverage
from forecast_lab.metrics_prob import crps_sample

def test_basic_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.0, 2.5])
    assert abs(mae(y, p) - 1/3) < 1e-9

def test_mase_naive_is_one_ish():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200) + 5
    score = mase(y[24:], np.roll(y, 24)[24:], y, season=24)
    assert 0.5 < score < 2.0

def test_coverage():
    y = np.array([1, 2, 3, 4])
    lo = np.array([0, 1, 2, 3])
    hi = np.array([2, 3, 5, 5])
    assert coverage(y, lo, hi) == 1.0

def test_crps_perfect_forecast():
    y = np.array([3.0, 4.0])
    samples = np.tile(y[:, None], (1, 200))
    assert crps_sample(y, samples) < 1e-6