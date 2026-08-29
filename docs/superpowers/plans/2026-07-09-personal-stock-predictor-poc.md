# Personal Stock Predictor PoC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, personal Streamlit app that fetches stock data, runs SMA/EMA/LSTM forecasts, evaluates with RMSE/MAPE and backtest, and manages a personal watchlist.

**Architecture:** Python package under `app/` with data, models, evaluation, UI, and CLI modules. Streamlit calls core modules directly (no API). Local Parquet cache and JSON watchlist. Chronological train/holdout splits.

**Tech Stack:** Python 3.11+, yfinance, pandas, numpy, tensorflow/keras, streamlit, plotly, pytest

**Spec:** `docs/superpowers/specs/2026-07-09-personal-stock-predictor-poc-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies |
| `app/data/cache.py` | Read/write Parquet cache, TTL check |
| `app/data/fetcher.py` | Download OHLCV via yfinance |
| `app/models/base.py` | Abstract model interface |
| `app/models/sma.py` | Simple moving average model |
| `app/models/ema.py` | Exponential moving average model |
| `app/models/lstm.py` | Keras LSTM train/predict |
| `app/evaluation/metrics.py` | RMSE, MAPE |
| `app/evaluation/backtest.py` | Signal-based backtest vs buy-and-hold |
| `app/cli.py` | argparse CLI: fetch, predict, backtest |
| `app/ui/main.py` | Streamlit dashboard |
| `config/watchlist.json` | Default personal tickers |
| `tests/` | Unit and integration tests |

---

## Phase 1: Project Scaffold & Data Pipeline

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/data/__init__.py`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Create requirements.txt**

```text
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.15.0
streamlit>=1.30.0
plotly>=5.18.0
scikit-learn>=1.3.0
pytest>=7.4.0
pyarrow>=14.0.0
```

- [ ] **Step 2: Append to .gitignore**

```text
cache/
models/
.venv/
*.parquet
```

- [ ] **Step 3: Create empty package inits**

```python
# app/__init__.py
"""Personal stock price predictor PoC."""

# app/data/__init__.py
"""Data fetching and caching."""
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Successful install

- [ ] **Step 5: Commit**

```bash
git add requirements.txt app/ .gitignore
git commit -m "chore: add project scaffold and dependencies"
```

---

### Task 2: Cache module

**Files:**
- Create: `app/data/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache.py
import pandas as pd
from pathlib import Path
from app.data.cache import cache_path, is_stale, save_ohlcv, load_ohlcv

def test_cache_path_normalizes_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.cache.CACHE_DIR", tmp_path)
    assert cache_path("aapl") == tmp_path / "AAPL.parquet"

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.cache.CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
    save_ohlcv("AAPL", df)
    loaded = load_ohlcv("AAPL")
    assert len(loaded) == 3
    assert loaded["Close"].iloc[-1] == 102.0

def test_is_stale_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.cache.CACHE_DIR", tmp_path)
    assert is_stale("MSFT", max_age_hours=24) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement cache.py**

```python
# app/data/cache.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("cache")
DEFAULT_MAX_AGE_HOURS = 24


def cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def save_ohlcv(ticker: str, df: pd.DataFrame) -> Path:
    path = cache_path(ticker)
    df.to_parquet(path)
    return path


def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = cache_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {ticker.upper()}")
    return pd.read_parquet(path)


def is_stale(ticker: str, max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> bool:
    path = cache_path(ticker)
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    return age_hours > max_age_hours
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cache.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/data/cache.py tests/test_cache.py
git commit -m "feat: add local Parquet OHLCV cache"
```

---

### Task 3: Fetcher module

**Files:**
- Create: `app/data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing test with mocked yfinance**

```python
# tests/test_fetcher.py
import pandas as pd
import pytest
from app.data import fetcher

def test_normalize_columns_keeps_ohlcv():
    raw = pd.DataFrame(
        {
            ("Close", "AAPL"): [150.0, 151.0],
            ("Open", "AAPL"): [149.0, 150.0],
            ("High", "AAPL"): [151.0, 152.0],
            ("Low", "AAPL"): [148.0, 149.0],
            ("Volume", "AAPL"): [1e6, 1.1e6],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    out = fetcher._normalize_yfinance_frame(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 2

def test_fetch_ohlcv_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher.cache, "CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cached = pd.DataFrame(
        {
            "Open": range(5),
            "High": range(5),
            "Low": range(5),
            "Close": range(5),
            "Volume": range(5),
        },
        index=idx,
    )
    fetcher.cache.save_ohlcv("AAPL", cached)
    monkeypatch.setattr(fetcher.cache, "is_stale", lambda *a, **k: False)
    result = fetcher.fetch_ohlcv("AAPL", period="1y")
    assert len(result) == 5
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_fetcher.py -v`

- [ ] **Step 3: Implement fetcher.py**

```python
# app/data/fetcher.py
from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.data import cache

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.dropna()


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    ticker = ticker.upper()
    if not force_refresh and not cache.is_stale(ticker):
        return cache.load_ohlcv(ticker)

    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    df = _normalize_yfinance_frame(raw)
    cache.save_ohlcv(ticker, df)
    return df
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add yfinance OHLCV fetcher with cache integration"
```

---

### Task 4: CLI fetch command

**Files:**
- Create: `app/cli.py`

- [ ] **Step 1: Implement CLI fetch**

```python
# app/cli.py
from __future__ import annotations

import argparse

from app.data.fetcher import fetch_ohlcv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal stock predictor CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Fetch and cache OHLCV data")
    fetch_p.add_argument("ticker")
    fetch_p.add_argument("--period", default="1y")
    fetch_p.add_argument("--force", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        df = fetch_ohlcv(args.ticker, period=args.period, force_refresh=args.force)
        print(f"Fetched {len(df)} rows for {args.ticker.upper()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run: `python -m app.cli fetch AAPL --period 6mo`
Expected: `Fetched N rows for AAPL` with N > 100

- [ ] **Step 3: Commit**

```bash
git add app/cli.py
git commit -m "feat: add CLI fetch command"
```

---

## Phase 2: Forecast Models

### Task 5: Base model interface

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/base.py`

- [ ] **Step 1: Implement base.py**

```python
# app/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        ...
```

- [ ] **Step 2: Commit**

```bash
git add app/models/
git commit -m "feat: add BaseModel interface"
```

---

### Task 6: SMA model

**Files:**
- Create: `app/models/sma.py`
- Create: `tests/test_sma.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sma.py
import pandas as pd
from app.models.sma import SMAModel

def test_sma_predict_constant_series():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Close": [100.0] * 30}, index=idx)
    model = SMAModel(window=5)
    model.fit(df)
    preds = model.predict(df, horizon=3)
    assert len(preds) == 3
    assert preds.iloc[0] == 100.0
```

- [ ] **Step 2: Implement sma.py**

```python
# app/models/sma.py
from __future__ import annotations

import pandas as pd

from app.models.base import BaseModel


class SMAModel(BaseModel):
    name = "sma"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._last_close: float | None = None

    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        self._last_close = float(df[target_col].iloc[-1])

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        series = df["Close"].rolling(window=self.window).mean().dropna()
        last = float(series.iloc[-1])
        idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([last] * horizon, index=idx, name="prediction")
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/test_sma.py -v`

```bash
git add app/models/sma.py tests/test_sma.py
git commit -m "feat: add SMA forecast model"
```

---

### Task 7: EMA model

**Files:**
- Create: `app/models/ema.py`
- Create: `tests/test_ema.py`

- [ ] **Step 1: Write test and implement (mirror SMA pattern)**

```python
# app/models/ema.py
from __future__ import annotations

import pandas as pd

from app.models.base import BaseModel


class EMAModel(BaseModel):
    name = "ema"

    def __init__(self, span: int = 20) -> None:
        self.span = span

    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None:
        self._ema = df[target_col].ewm(span=self.span, adjust=False).mean()

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        series = df["Close"].ewm(span=self.span, adjust=False).mean()
        last = float(series.iloc[-1])
        idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([last] * horizon, index=idx, name="prediction")
```

- [ ] **Step 2: Test and commit**

Run: `pytest tests/test_ema.py -v`

```bash
git add app/models/ema.py tests/test_ema.py
git commit -m "feat: add EMA forecast model"
```

---

### Task 8: LSTM model

**Files:**
- Create: `app/models/lstm.py`
- Create: `tests/test_lstm.py`

- [ ] **Step 1: Write integration test (small synthetic data)**

```python
# tests/test_lstm.py
import pandas as pd
import pytest
from app.models.lstm import LSTMModel, MIN_ROWS

def test_lstm_requires_minimum_rows():
    idx = pd.date_range("2024-01-01", periods=MIN_ROWS - 1, freq="D")
    df = pd.DataFrame({"Close": range(MIN_ROWS - 1)}, index=idx)
    model = LSTMModel(epochs=1, sequence_length=10)
    with pytest.raises(ValueError):
        model.fit(df)

@pytest.mark.slow
def test_lstm_predict_shape():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"Close": (pd.Series(range(120)) + 100).astype(float)}, index=idx)
    model = LSTMModel(epochs=1, sequence_length=10)
    model.fit(df)
    preds = model.predict(df, horizon=3)
    assert len(preds) == 3
```

- [ ] **Step 2: Implement lstm.py**

```python
# app/models/lstm.py
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
        window = scaled[-self.sequence_length :].reshape(1, self.sequence_length, 1)
        preds = []
        current = window.copy()
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
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/test_lstm.py -v -m "not slow"` (mark slow test optional in CI)

```bash
git add app/models/lstm.py tests/test_lstm.py
git commit -m "feat: add LSTM forecast model"
```

---

### Task 9: Model registry + CLI predict

**Files:**
- Modify: `app/models/__init__.py`
- Modify: `app/cli.py`

- [ ] **Step 1: Add get_model factory**

```python
# app/models/__init__.py
from app.models.ema import EMAModel
from app.models.lstm import LSTMModel
from app.models.sma import SMAModel

MODELS = {"sma": SMAModel, "ema": EMAModel, "lstm": LSTMModel}


def get_model(name: str):
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model {name}. Choose from {list(MODELS)}")
    return MODELS[key]()
```

- [ ] **Step 2: Add predict subcommand to cli.py**

```python
# In build_parser():
pred_p = sub.add_parser("predict", help="Run forecast")
pred_p.add_argument("ticker")
pred_p.add_argument("--model", default="sma", choices=["sma", "ema", "lstm"])
pred_p.add_argument("--horizon", type=int, default=5)
pred_p.add_argument("--period", default="1y")

# In main():
elif args.command == "predict":
    from app.models import get_model
    df = fetch_ohlcv(args.ticker, period=args.period)
    model = get_model(args.model)
    model.fit(df)
    preds = model.predict(df, horizon=args.horizon)
    print(preds)
```

- [ ] **Step 3: Smoke test and commit**

Run: `python -m app.cli predict AAPL --model sma --horizon 3`

```bash
git add app/models/__init__.py app/cli.py
git commit -m "feat: add model registry and CLI predict command"
```

---

## Phase 3: Evaluation & Backtest

### Task 10: Metrics module

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Implement metrics with tests**

```python
# app/evaluation/metrics.py
import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
```

```python
# tests/test_metrics.py
import numpy as np
from app.evaluation.metrics import rmse, mape

def test_rmse_zero_error():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0

def test_mape_known():
    a = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    assert mape(a, p) == 5.0
```

- [ ] **Step 2: Run tests and commit**

Run: `pytest tests/test_metrics.py -v`

```bash
git add app/evaluation/ tests/test_metrics.py
git commit -m "feat: add RMSE and MAPE metrics"
```

---

### Task 11: Holdout split helper

**Files:**
- Create: `app/evaluation/split.py`
- Create: `tests/test_split.py`

- [ ] **Step 1: Implement chronological split**

```python
# app/evaluation/split.py
import pandas as pd


def train_holdout_split(df: pd.DataFrame, holdout_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1")
    split_idx = int(len(df) * (1 - holdout_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
```

- [ ] **Step 2: Test and commit**

```bash
git add app/evaluation/split.py tests/test_split.py
git commit -m "feat: add chronological train/holdout split"
```

---

### Task 12: Backtest module

**Files:**
- Create: `app/evaluation/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Implement simple long/flat backtest**

```python
# app/evaluation/backtest.py
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
```

- [ ] **Step 2: Test and commit**

```bash
git add app/evaluation/backtest.py tests/test_backtest.py
git commit -m "feat: add simple backtest vs buy-and-hold"
```

---

### Task 13: Evaluation orchestration + CLI backtest

**Files:**
- Create: `app/evaluation/runner.py`
- Modify: `app/cli.py`

- [ ] **Step 1: Implement runner that fits on train, predicts holdout**

```python
# app/evaluation/runner.py
from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.backtest import run_backtest
from app.evaluation.metrics import mape, rmse
from app.evaluation.split import train_holdout_split
from app.models.base import BaseModel


def evaluate_model(model: BaseModel, df: pd.DataFrame, horizon: int = 1) -> dict:
    train, holdout = train_holdout_split(df)
    model.fit(train)
    # one-step rolling predictions on holdout for metrics
    preds = []
    actuals = []
    rolling = train.copy()
    for i in range(len(holdout)):
        p = model.predict(rolling, horizon=1)
        preds.append(float(p.iloc[0]))
        actuals.append(float(holdout["Close"].iloc[i]))
        rolling = pd.concat([rolling, holdout.iloc[[i]]])
    pred_arr = np.array(preds)
    act_arr = np.array(actuals)
    bt = run_backtest(holdout["Close"], pd.Series(pred_arr, index=holdout.index))
    return {
        "rmse": rmse(act_arr, pred_arr),
        "mape": mape(act_arr, pred_arr),
        **bt,
    }
```

- [ ] **Step 2: Add backtest CLI subcommand and smoke test**

Run: `python -m app.cli backtest AAPL --model sma`

- [ ] **Step 3: Commit**

```bash
git add app/evaluation/runner.py app/cli.py
git commit -m "feat: add evaluation runner and CLI backtest"
```

---

## Phase 4: Streamlit UI

### Task 14: Watchlist config

**Files:**
- Create: `config/watchlist.json`
- Create: `app/ui/watchlist.py`

- [ ] **Step 1: Create default watchlist**

```json
["AAPL", "MSFT", "GOOGL"]
```

- [ ] **Step 2: Implement load/save helpers**

```python
# app/ui/watchlist.py
import json
from pathlib import Path

WATCHLIST_PATH = Path("config/watchlist.json")


def load_watchlist() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return ["AAPL"]
    return json.loads(WATCHLIST_PATH.read_text())


def save_watchlist(tickers: list[str]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(sorted(set(t.upper() for t in tickers)), indent=2))
```

- [ ] **Step 3: Commit**

```bash
git add config/watchlist.json app/ui/watchlist.py
git commit -m "feat: add personal watchlist persistence"
```

---

### Task 15: Streamlit dashboard

**Files:**
- Create: `app/ui/__init__.py`
- Create: `app/ui/main.py`

- [ ] **Step 1: Implement main Streamlit app**

```python
# app/ui/main.py
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from app.data.fetcher import fetch_ohlcv
from app.evaluation.runner import evaluate_model
from app.models import get_model
from app.ui.watchlist import load_watchlist, save_watchlist

st.set_page_config(page_title="Stock Predictor PoC", layout="wide")
st.title("ml-stock-price-predictor")

watchlist = load_watchlist()
with st.sidebar:
    st.header("Watchlist")
    ticker = st.selectbox("Ticker", watchlist)
    new_ticker = st.text_input("Add ticker").upper().strip()
    if st.button("Add to watchlist") and new_ticker:
        watchlist = sorted(set(watchlist + [new_ticker]))
        save_watchlist(watchlist)
        st.rerun()
    model_name = st.selectbox("Model", ["sma", "ema", "lstm"])
    horizon = st.slider("Forecast horizon (days)", 1, 14, 5)
    period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
    force_refresh = st.button("Refresh data")

df = fetch_ohlcv(ticker, period=period, force_refresh=force_refresh)
model = get_model(model_name)
with st.spinner("Running forecast..."):
    metrics = evaluate_model(model, df, horizon=horizon)
    model.fit(df)
    forecast = model.predict(df, horizon=horizon)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast"))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE", f"{metrics['rmse']:.2f}")
c2.metric("MAPE", f"{metrics['mape']:.2f}%")
c3.metric("Strategy return", f"{metrics['strategy_return_pct']:.2f}%")
c4.metric("Buy & hold", f"{metrics['buy_hold_return_pct']:.2f}%")

st.caption("Educational proof of concept. Not financial advice.")
```

- [ ] **Step 2: Manual smoke test**

Run: `streamlit run app/ui/main.py`
Expected: Dashboard loads, chart renders for AAPL

- [ ] **Step 3: Commit**

```bash
git add app/ui/
git commit -m "feat: add Streamlit personal dashboard"
```

---

## Phase 5: Polish & Documentation

### Task 16: README and pytest config

**Files:**
- Modify: `README.md`
- Create: `pytest.ini`

- [ ] **Step 1: Update README with setup and usage**

Include:
- Project purpose (personal PoC)
- `pip install -r requirements.txt`
- `python -m app.cli fetch AAPL`
- `streamlit run app/ui/main.py`
- Disclaimer

- [ ] **Step 2: Add pytest.ini**

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -v -m "not slow"`
Expected: All non-slow tests PASS

- [ ] **Step 4: Commit**

```bash
git add README.md pytest.ini
git commit -m "docs: add README and pytest configuration"
```

---

## Plan Self-Review

| Spec requirement | Covered by |
|------------------|------------|
| yfinance fetch + cache | Tasks 2–4 |
| SMA, EMA, LSTM | Tasks 5–9 |
| RMSE, MAPE | Task 10 |
| Backtest vs buy-and-hold | Tasks 12–13 |
| Streamlit UI | Tasks 14–15 |
| Personal watchlist | Task 14 |
| CLI | Tasks 4, 9, 13 |
| README / disclaimer | Task 16 |
| Error handling (invalid ticker) | fetcher raises ValueError; UI should wrap in try/except during Task 15 polish |
| Min 60 rows for LSTM | Task 8 |

**Gap to address during Task 15:** Wrap `fetch_ohlcv` and model calls in `st.error()` handlers for invalid tickers.

---

## Execution Order

Implement phases sequentially: 1 → 2 → 3 → 4 → 5. Each phase produces testable, committable increments.

**Estimated commands for final verification:**

```bash
pip install -r requirements.txt
pytest -v -m "not slow"
python -m app.cli fetch AAPL
python -m app.cli predict AAPL --model ema --horizon 5
python -m app.cli backtest AAPL --model sma
streamlit run app/ui/main.py
```
