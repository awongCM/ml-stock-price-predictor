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
