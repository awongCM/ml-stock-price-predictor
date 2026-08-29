from __future__ import annotations

import pandas as pd

from app.models.base import BaseModel


class SMAModel(BaseModel):
    name = "sma"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        return None

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        series = df["Close"].rolling(window=self.window).mean().dropna()
        last = float(series.iloc[-1])
        idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([last] * horizon, index=idx, name="prediction")
