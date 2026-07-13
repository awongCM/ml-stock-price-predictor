from __future__ import annotations

import pandas as pd
import yfinance as yf

from app.data import cache

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
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
