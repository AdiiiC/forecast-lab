from forecast_lab.backtest import walk_forward_splits

def test_no_overlap_and_monotonic():
    splits = list(walk_forward_splits(n=1000, min_train=200, horizon=24,
                                      n_folds=5, stride=50))
    assert len(splits) == 5
    last_te = -1
    for te, ee in splits:
        assert te >= 200
        assert ee == te + 24
        assert te > last_te
        last_te = te


def test_train_strictly_before_test():
    for te, ee in walk_forward_splits(500, 100, 10, 3, 50):
        assert te <= ee - 10