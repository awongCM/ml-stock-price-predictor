import pandas as pd

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
    monkeypatch.setattr(fetcher.cache, "is_stale", lambda *args, **kwargs: False)
    result = fetcher.fetch_ohlcv("AAPL", period="1y")
    assert len(result) == 5
