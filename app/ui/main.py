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
    new_ticker = st.text_input("Add ticker").strip().upper()
    if st.button("Add to watchlist") and new_ticker:
        save_watchlist(watchlist + [new_ticker])
        st.rerun()

    model_name = st.selectbox("Model", ["sma", "ema", "lstm"])
    horizon = st.slider("Forecast horizon (days)", 1, 14, 5)
    period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
    force_refresh = st.button("Refresh data")

try:
    df = fetch_ohlcv(ticker, period=period, force_refresh=force_refresh)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

model = get_model(model_name)

with st.spinner("Running forecast..."):
    try:
        metrics = evaluate_model(model, df, horizon=horizon)
        model.fit(df)
        forecast = model.predict(df, horizon=horizon)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", mode="lines+markers"))
fig.update_layout(
    title=f"{ticker} — Close price and forecast ({model_name.upper()})",
    xaxis_title="Date",
    yaxis_title="Price",
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{metrics['rmse']:.2f}")
col2.metric("MAPE", f"{metrics['mape']:.2f}%")
col3.metric("Strategy return", f"{metrics['strategy_return_pct']:.2f}%")
col4.metric("Buy & hold", f"{metrics['buy_hold_return_pct']:.2f}%")

st.caption("Educational proof of concept. Not financial advice.")
