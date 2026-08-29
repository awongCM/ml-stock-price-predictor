import json
from pathlib import Path

WATCHLIST_PATH = Path("config/watchlist.json")


def load_watchlist() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return ["AAPL"]
    return json.loads(WATCHLIST_PATH.read_text())


def save_watchlist(tickers: list[str]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = sorted({ticker.upper() for ticker in tickers if ticker})
    WATCHLIST_PATH.write_text(json.dumps(normalized, indent=2))
