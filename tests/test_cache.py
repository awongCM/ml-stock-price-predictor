import pandas as pd

from app.data.cache import cache_path, is_stale, load_ohlcv, save_ohlcv


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
