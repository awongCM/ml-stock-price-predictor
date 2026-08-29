# ml-stock-price-predictor

A personal, local proof-of-concept app for stock price forecasting using machine learning.

## What it does

- Fetches and caches historical OHLCV data via Yahoo Finance (`yfinance`)
- Runs three forecast models: **SMA**, **EMA**, and **LSTM**
- Evaluates models with **RMSE**, **MAPE**, and a simple backtest vs buy-and-hold
- Provides a **Streamlit dashboard** with a personal watchlist

> Educational proof of concept. Not financial advice.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage

```bash
# Fetch and cache data
python -m app.cli fetch AAPL --period 1y

# Run a forecast
python -m app.cli predict AAPL --model ema --horizon 5

# Evaluate on holdout data
python -m app.cli backtest AAPL --model sma
```

## Run the dashboard

```bash
streamlit run app/ui/main.py
```

Open the URL shown in your terminal. Use the sidebar to pick a ticker, model, and forecast horizon.

## Project structure

```
app/
  data/         # fetch + cache
  models/       # SMA, EMA, LSTM
  evaluation/   # metrics + backtest
  ui/           # Streamlit app
  cli.py        # command-line interface
config/
  watchlist.json
```

## Tests

```bash
pytest -v -m "not slow"
```

## Docs

- Design spec: `docs/superpowers/specs/2026-07-09-personal-stock-predictor-poc-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-09-personal-stock-predictor-poc.md`

## Reference

Original inspiration: [Stock Price Prediction using Machine Learning (ProjectPro)](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571)
