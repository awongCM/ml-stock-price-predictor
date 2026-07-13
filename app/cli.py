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

    pred_p = sub.add_parser("predict", help="Run forecast")
    pred_p.add_argument("ticker")
    pred_p.add_argument("--model", default="sma", choices=["sma", "ema", "lstm"])
    pred_p.add_argument("--horizon", type=int, default=5)
    pred_p.add_argument("--period", default="1y")

    bt_p = sub.add_parser("backtest", help="Evaluate model on holdout data")
    bt_p.add_argument("ticker")
    bt_p.add_argument("--model", default="sma", choices=["sma", "ema", "lstm"])
    bt_p.add_argument("--period", default="1y")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        df = fetch_ohlcv(args.ticker, period=args.period, force_refresh=args.force)
        print(f"Fetched {len(df)} rows for {args.ticker.upper()}")
        return

    if args.command == "predict":
        from app.models import get_model

        df = fetch_ohlcv(args.ticker, period=args.period)
        model = get_model(args.model)
        model.fit(df)
        preds = model.predict(df, horizon=args.horizon)
        print(preds)
        return

    if args.command == "backtest":
        from app.evaluation.runner import evaluate_model
        from app.models import get_model

        df = fetch_ohlcv(args.ticker, period=args.period)
        model = get_model(args.model)
        metrics = evaluate_model(model, df)
        print(
            f"RMSE: {metrics['rmse']:.4f}  MAPE: {metrics['mape']:.2f}%  "
            f"Strategy: {metrics['strategy_return_pct']:.2f}%  "
            f"Buy&Hold: {metrics['buy_hold_return_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
