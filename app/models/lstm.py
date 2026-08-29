from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from app.models.base import BaseModel

MIN_ROWS = 60
MODELS_DIR = Path("models")


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(
        self,
        sequence_length: int = 20,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> None:
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model: keras.Model | None = None

    def _build_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in range(len(values) - self.sequence_length):
            x.append(values[i : i + self.sequence_length])
            y.append(values[i + self.sequence_length])
        return np.array(x), np.array(y)

    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        if len(df) < MIN_ROWS:
            raise ValueError(f"LSTM requires at least {MIN_ROWS} rows")
        scaled = self.scaler.fit_transform(df[[target_col]].values)
        x, y = self._build_sequences(scaled)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        self.model = keras.Sequential(
            [
                layers.LSTM(32, input_shape=(self.sequence_length, 1)),
                layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        scaled = self.scaler.transform(df[["Close"]].values)
        current = scaled[-self.sequence_length :].reshape(1, self.sequence_length, 1)
        preds = []
        for _ in range(horizon):
            nxt = self.model.predict(current, verbose=0)
            preds.append(float(nxt[0, 0]))
            current = np.append(current[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
        inv = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(inv, index=idx, name="prediction")

    def save(self, ticker: str) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{ticker.upper()}_lstm.keras"
        if self.model is not None:
            self.model.save(path)
        return path
