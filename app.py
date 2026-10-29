"""
COMP4145 PD2 - Enhanced Single Stock / Portfolio Trading App
Extra features:
  1. Multi-stock portfolio analysis
  2. SQLite holdings persistence
  3. Plotly interactive visualisation
"""

import concurrent.futures
import sqlite3
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import trading_strategy as ts

DB_PATH = "holdings.db"
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "JNJ",
]

st.set_page_config(page_title="COMP4145 PD2 - Trading App", layout="wide")

# Render a visible shell immediately, before any network call.
st.title("COMP4145 PD2 - Trading App")
st.caption("Use the sidebar to navigate. Price data may take a few seconds to load.")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            ticker TEXT PRIMARY KEY,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


@st.cache_resource
def get_connection():
    return init_db()


def get_holdings():
    return pd.read_sql_query(
        "SELECT ticker, shares, avg_cost, updated_at FROM holdings ORDER BY ticker",
        get_connection(),
    )


def upsert_holding(ticker, shares, avg_cost):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO holdings (ticker, shares, avg_cost, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            shares = excluded.shares,
            avg_cost = excluded.avg_cost,
            updated_at = CURRENT_TIMESTAMP
        """,
        (ticker.upper().strip(), float(shares), float(avg_cost)),
    )
    conn.commit()


def delete_holding(ticker):
    conn = get_connection()
    conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
    conn.commit()


# ---------------------------------------------------------------------------
# Cached data loader with timeout
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers, start, end):
    """
    Download prices in a worker thread with a hard timeout.
    Raises RuntimeError on timeout or empty data so Streamlit can show an error
    instead of hanging on a blank page.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ts.download_close, list(tickers), start, end)
        try:
            df = future.result(timeout=20)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                "Data download timed out after 20 seconds. "
                "Yahoo Finance may be slow, blocked, or rate-limiting. "
                "Try a shorter date range or fewer tickers."
            )

    if df is None or df.empty:
        raise RuntimeError(
            "No price data returned. Check ticker symbols, date range, "
            "or network access to Yahoo Finance."
        )
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Page",
    ["Portfolio Overview", "Stock Analysis", "Risk & Allocation", "Holdings"],
)

today = date.today()
default_start = today - timedelta(days=365 * 5)
date_range = st.sidebar.date_input(
    "Date range", value=(default_start, today)
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, today

st.sidebar.caption("Data source: Yahoo Finance. Demo only, not investment advice.")


# ---------------------------------------------------------------------------
# Page 1: Portfolio Overview
# ---------------------------------------------------------------------------
if page == "Portfolio Overview":
    st.header("Portfolio Overview")
    selected = st.multiselect(
        "Select stocks (max 10)",
        DEFAULT_TICKERS,
        default=["AAPL", "MSFT", "GOOGL"],
    )

    if not selected:
        st.info("Select at least one ticker to continue.")
    elif len(selected) > 10:
        st.warning("Please select at most 10 tickers (Yahoo Finance limit).")
    else:
        with st.spinner("Downloading price data from Yahoo Finance..."):
            try:
                close = load_prices(tuple(selected), start_date, end_date)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        _, port_ret, cum = ts.portfolio_returns(close)
        metrics = ts.risk_metrics(port_ret)

        c1, c2, c3, c4 = st.columns(4)
        total_return = (cum.iloc[-1] - 1) * 100 if not cum.empty else 0.0
        c1.metric("Total Return (%)", f"{total_return:.2f}")
        c2.metric("Annual Return (%)", metrics["Annual Return (%)"])
        c3.metric("Annual Volatility (%)", metrics["Annual Volatility (%)"])
        c4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])

        st.subheader("Portfolio Cumulative Return")
        fig_cum = px.line(
            x=cum.index,
            y=(cum - 1) * 100,
            labels={"x": "Date", "y": "Cumulative Return (%)"},
        )
        fig_cum.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Normalised Price (Base = 100)")
        normalised = close / close.iloc[0] * 100
        fig_norm = px.line(normalised, labels={"value": "Index", "variable": "Ticker"})
        fig_norm.update_layout(height=420)
        st.plotly_chart(fig_norm, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Stock Analysis (Golden Cross)
# ---------------------------------------------------------------------------
elif page == "Stock Analysis":
    st.header("Single Stock Analysis - Golden Cross")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)

    with st.spinner("Downloading price data from Yahoo Finance..."):
        try:
            close = load_prices((ticker,), start_date, end_date)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    if ticker not in close.columns:
        st.error("No price data returned for this ticker.")
        st.stop()

    df = close.rename(columns={ticker: "Close"})
    df = ts.add_moving_averages(df)

    if df["MA_long"].isna().all():
        st.warning("Not enough data to compute the 200-day moving average. Extend the date range.")

    df, trades = ts.backtest_golden_cross(df)
    stats = ts.trade_statistics(trades)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", stats["Total Trades"])
    c2.metric("Win Rate (%)", stats["Win Rate (%)"])
    c3.metric("Average Profit", stats["Average Profit"])
    c4.metric("Total Profit", stats["Total Profit"])

    st.subheader(f"{ticker} Price Chart with Moving Averages and Trades")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_short"], name="SMA 50", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_long"], name="SMA 200", line=dict(color="#2ca02c")))

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="green"),
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="red"),
            )
        )
    fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Trades")
    if trades.empty:
        st.info("No Golden Cross trades found in this period.")
    else:
        st.dataframe(trades, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: Risk & Allocation
# ---------------------------------------------------------------------------
elif page == "Risk & Allocation":
    st.header("Risk & Allocation")

    holdings = get_holdings()
    if holdings.empty:
        st.info("No holdings yet. Showing equal weights for the selected tickers.")
        allocation_tickers = st.multiselect(
            "Tickers for allocation",
            DEFAULT_TICKERS,
            default=["AAPL", "MSFT", "GOOGL"],
        )
    else:
        allocation_tickers = holdings["ticker"].tolist()

    if allocation_tickers:
        with st.spinner("Downloading price data from Yahoo Finance..."):
            try:
                close = load_prices(tuple(allocation_tickers), start_date, end_date)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        if not holdings.empty:
            latest = close.iloc[-1]
            market_value = holdings.set_index("ticker")["shares"] * latest
            market_value = market_value.dropna()
            if not market_value.empty:
                pie_df = market_value.reset_index()
                pie_df.columns = ["Ticker", "Market Value"]
                fig_pie = px.pie(
                    pie_df, values="Market Value", names="Ticker",
                    title="Allocation by Market Value",
                )
            else:
                st.warning("Could not compute market values from holdings.")
                fig_pie = None
        else:
            fig_pie = px.pie(
                names=allocation_tickers,
                values=[1] * len(allocation_tickers),
                title="Equal-Weight Allocation",
            )

        if fig_pie is not None:
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr_tickers = st.multiselect(
            "Tickers for correlation",
            DEFAULT_TICKERS,
            default=["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
        )
        if len(corr_tickers) >= 2:
            with st.spinner("Downloading correlation data..."):
                try:
                    corr_close = load_prices(tuple(corr_tickers), start_date, end_date)
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

            returns = corr_close.pct_change().dropna(how="all")
            corr = returns.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title="Daily Return Correlation",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select at least two tickers.")


# ---------------------------------------------------------------------------
# Page 4: Holdings (SQLite)
# ---------------------------------------------------------------------------
elif page == "Holdings":
    st.header("Holdings Database")
    holdings = get_holdings()

    st.subheader("Current Holdings")
    if holdings.empty:
        st.info("No holdings yet. Add one below.")
    else:
        st.dataframe(holdings, use_container_width=True)

    st.subheader("Add or Update Holding")
    with st.form("holding_form"):
        ticker = st.text_input("Ticker", placeholder="e.g. AAPL")
        shares = st.number_input("Shares", min_value=0.0, step=1.0, value=1.0)
        avg_cost = st.number_input("Average cost per share (USD)", min_value=0.0, step=0.01, value=100.0)
        submitted = st.form_submit_button("Save Holding")

        if submitted:
            if not ticker.strip():
                st.error("Ticker cannot be empty.")
            else:
                upsert_holding(ticker, shares, avg_cost)
                st.success(f"Saved {ticker.upper()}.")
                st.rerun()

    st.subheader("Delete Holding")
    if not holdings.empty:
        to_delete = st.selectbox("Select ticker to delete", holdings["ticker"].tolist())
        if st.button("Delete"):
            delete_holding(to_delete)
            st.success(f"Deleted {to_delete}.")
            st.rerun()
    else:
        st.caption("Nothing to delete.")