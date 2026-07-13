import pandas as pd

from app.evaluation.backtest import run_backtest


def test_backtest_buy_and_hold_on_rising_series():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    actual = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=idx)
    predicted = pd.Series([101.0, 111.0, 121.0, 131.0, 141.0], index=idx)
    result = run_backtest(actual, predicted)
    assert result["buy_hold_return_pct"] > 0
