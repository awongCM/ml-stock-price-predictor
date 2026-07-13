import numpy as np

from app.evaluation.metrics import mape, rmse


def test_rmse_zero_error():
    values = np.array([1.0, 2.0, 3.0])
    assert rmse(values, values) == 0.0


def test_mape_known():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    assert mape(actual, predicted) == 10.0
