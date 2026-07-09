# Personal Stock Predictor PoC — Design Spec

**Date:** 2026-07-09  
**Status:** Approved  
**Owner:** Personal use (Option A — single user, local only)

---

## 1. Purpose

Transform `ml-stock-price-predictor` from an empty placeholder into a **local, personal Proof of Concept** for stock price forecasting. The project follows Path A (tutorial-inspired ML pipeline) but delivers a **usable app**, not notebooks-only learning material.

**Primary user:** The repo owner only. No authentication, no multi-tenant features, no production deployment required for v1.

---

## 2. Goals

### In scope

- Fetch and cache historical OHLCV data for any US ticker via `yfinance`
- Implement three forecast models with a shared interface: SMA, EMA, LSTM
- Evaluate models with RMSE and MAPE
- Run a simple backtest comparing model signals vs buy-and-hold
- Streamlit dashboard for interactive exploration
- Personal watchlist persisted in a local JSON file
- CLI for headless operations (fetch, predict, backtest)
- Reproducible local setup documented in README

### Out of scope (v1)

- User authentication or accounts
- Cloud deployment (optional later)
- Email/Slack alerts
- Real brokerage or live trading integration
- News/sentiment data
- Multi-user or shared watchlists

---

## 3. Success Criteria

1. Add a ticker (e.g. `AAPL`) to watchlist and view 1+ year of price history
2. Generate next-day (or next-N-day) forecast using LSTM, SMA, or EMA
3. Switch models in the UI without restarting the app
4. View RMSE/MAPE and backtest return vs buy-and-hold for the selected model
5. Refresh data with one click after markets update
6. Fresh clone → `pip install -r requirements.txt` → `streamlit run app/ui/main.py` works per README

---

## 4. Architecture

```
Streamlit UI  →  Python core modules  →  local cache + saved model weights
                      ├── data (fetch, cache)
                      ├── models (SMA, EMA, LSTM)
                      └── evaluation (metrics, backtest)
```

**Design decision:** No FastAPI layer for v1. Streamlit imports core modules directly. Reduces complexity for a single-user local app.

### Data flow

1. User selects ticker in Streamlit (or CLI passes ticker argument)
2. `data.fetcher` checks local cache; fetches from Yahoo Finance if stale or missing
3. `data.cache` stores OHLCV as Parquet under `cache/{ticker}.parquet`
4. Selected model loads history, trains if needed (LSTM), returns forecast series
5. `evaluation.metrics` computes RMSE/MAPE on holdout window
6. `evaluation.backtest` simulates simple long-only strategy vs buy-and-hold
7. UI renders price chart, forecast overlay, and metric cards

---

## 5. Repository Layout

```
ml-stock-price-predictor/
├── app/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py
│   │   └── cache.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sma.py
│   │   ├── ema.py
│   │   └── lstm.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── backtest.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main.py
│   └── cli.py
├── config/
│   └── watchlist.json
├── cache/                  # gitignored
├── models/                 # gitignored trained weights
├── notebooks/              # optional experiments
├── tests/
├── docs/superpowers/
├── requirements.txt
├── pyproject.toml          # optional; requirements.txt sufficient for v1
└── README.md
```

---

## 6. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Matches original intent and ML ecosystem |
| Data source | `yfinance` | Free, no API key, adequate for personal PoC |
| DataFrames | `pandas` | Standard for OHLCV manipulation |
| Baselines | `pandas` / `numpy` | SMA and EMA are rolling computations |
| LSTM | `tensorflow` + `keras` | Aligns with original tutorial reference |
| Metrics | `numpy`, `sklearn.metrics` | RMSE; MAPE custom |
| UI | `streamlit` | Fastest path to personal dashboard |
| Charts | `plotly` or `matplotlib` | Interactive charts in Streamlit |
| Storage | Local Parquet + JSON | No database for single user |
| Testing | `pytest` | Unit tests for data, models, metrics |
| CLI | `argparse` or `typer` | Headless fetch/predict/backtest |

---

## 7. Phased Delivery

### Phase 1 — Data pipeline

- `yfinance` fetch for configurable date range
- Parquet cache with TTL (default: refresh if older than 1 day)
- CLI: `python -m app.cli fetch AAPL --period 1y`

### Phase 2 — Models

- Abstract `BaseModel` with `fit(df)`, `predict(steps)`, `name`
- SMA: window parameter (default 20)
- EMA: span parameter (default 20)
- LSTM: sequence length, epochs, saved weights to `models/{ticker}_lstm.keras`
- CLI: `python -m app.cli predict AAPL --model lstm --horizon 5`

### Phase 3 — Evaluation

- RMSE and MAPE on last 20% holdout (time-series split, no shuffle)
- Backtest: go long when forecast > previous close; flat otherwise
- Report total return vs buy-and-hold over holdout period
- CLI: `python -m app.cli backtest AAPL --model ema`

### Phase 4 — Streamlit UI

- Sidebar: watchlist, add/remove ticker, model selector, horizon slider
- Main: candlestick/line chart with historical close + forecast
- Metrics panel: RMSE, MAPE, backtest summary
- Buttons: Refresh data, Run forecast

### Phase 5 — Personal polish

- Default watchlist: `["AAPL", "MSFT", "GOOGL"]`
- Session remembers last viewed ticker
- README with setup, screenshots, disclaimer
- `.gitignore` updates for `cache/`, `models/`, `.venv`

---

## 8. Model Interface

```python
class BaseModel(ABC):
    name: str

    def fit(self, df: pd.DataFrame, target_col: str = "Close") -> None: ...
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series: ...
```

- Input `df`: DatetimeIndex, columns at minimum `Open`, `High`, `Low`, `Close`, `Volume`
- LSTM uses MinMax-scaled `Close` sequences; inverse-transform for output
- SMA/EMA use rolling logic on `Close`; extend forecast by repeating last computed trend or iterative one-step

---

## 9. Evaluation Rules

- **Train/holdout split:** chronological 80/20; never shuffle
- **RMSE:** `sqrt(mean((actual - predicted)^2))` on holdout
- **MAPE:** `mean(|actual - predicted| / actual) * 100` on holdout (exclude zero prices)
- **Backtest:** starting cash 10,000; fully invested when signal bullish; compare to buy-and-hold of same ticker over holdout window; assume no transaction costs in v1

---

## 10. UI Wireframe (logical)

```
┌─────────────────────────────────────────────────────────┐
│  ml-stock-price-predictor                    [Refresh]  │
├──────────────┬──────────────────────────────────────────┤
│ Watchlist    │  AAPL — Close price + forecast chart     │
│  • AAPL      │                                          │
│  • MSFT      │  ┌────────────────────────────────────┐  │
│  • GOOGL     │  │     📈 line chart                  │  │
│ [+ Add]      │  └────────────────────────────────────┘  │
│              │                                          │
│ Model: LSTM  │  RMSE: 2.34   MAPE: 1.2%                 │
│ Horizon: 5   │  Backtest: +3.2%  vs B&H: +5.1%         │
│              │                                          │
│              │  ⚠ Educational PoC — not financial advice│
└──────────────┴──────────────────────────────────────────┘
```

---

## 11. Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid ticker | Show clear error in UI/CLI; do not crash app |
| Empty yfinance response | Retry once; then surface "no data for ticker" |
| Insufficient history for LSTM | Require minimum 60 rows; message if below |
| Missing cached file | Auto-fetch on first access |
| LSTM training slow | Show Streamlit spinner; cache trained model |

---

## 12. Testing Strategy

- **Unit:** cache read/write, SMA/EMA known values on synthetic series, RMSE/MAPE on fixed arrays
- **Integration:** fetch mock or recorded fixture CSV; end-to-end predict pipeline
- **Manual:** run Streamlit locally; verify chart and metrics for AAPL

---

## 13. Disclaimer

All UI surfaces include footer text:

> Educational proof of concept. Not financial advice. Past performance does not guarantee future results.

---

## 14. Future Extensions (not v1)

- Optional Render deploy for remote access
- Volatility forecasting
- Sentiment from news headlines
- Transaction costs in backtest
- FastAPI layer if UI is split from core

---

## 15. Approval

Design reviewed and approved by user on 2026-07-09 (Option A — personal local use).
