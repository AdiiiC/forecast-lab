import numpy as np
from forecast_lab.reconciliation import reconcile

def test_bu_is_coherent():
    S = np.array([[1, 1, 1],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    y_hat = np.array([[10, 3, 4, 2]])  # incoherent: 10 ≠ 3+4+2
    rec = reconcile("bu", S, y_hat)
    assert abs(rec[0, 0] - (rec[0, 1] + rec[0, 2] + rec[0, 3])) < 1e-9

def test_mint_shrink_reduces_top_level_variance():
    rng = np.random.default_rng(0)
    S = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
    resid = rng.normal(size=(500, 3))
    y_hat = rng.normal(size=(10, 3))
    rec = reconcile("mint_shrink", S, y_hat, residuals=resid)
    # coherent
    assert np.allclose(rec[:, 0], rec[:, 1] + rec[:, 2])