from __future__ import annotations

import pandas as pd


def run_backtest(
    actual: pd.Series,
    predicted: pd.Series,
    initial_cash: float = 10_000.0,
) -> dict[str, float]:
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if aligned.empty:
        return {"strategy_return_pct": 0.0, "buy_hold_return_pct": 0.0}

    cash = initial_cash
    shares = 0.0
    closes = aligned["actual"].values
    preds = aligned["predicted"].values

    for i in range(1, len(aligned)):
        signal_long = preds[i - 1] > closes[i - 1]
        if signal_long and shares == 0:
            shares = cash / closes[i]
            cash = 0.0
        elif not signal_long and shares > 0:
            cash = shares * closes[i]
            shares = 0.0

    final_value = cash + shares * closes[-1]
    strategy_return = (final_value / initial_cash - 1) * 100
    buy_hold_return = (closes[-1] / closes[0] - 1) * 100
    return {
        "strategy_return_pct": float(strategy_return),
        "buy_hold_return_pct": float(buy_hold_return),
    }
