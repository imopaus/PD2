"""
Trading strategy utilities for COMP4145 PD2 (Idea 1).
Strategy: Golden Cross
  - Buy  when the short-term SMA crosses above the long-term SMA.
  - Sell when the short-term SMA crosses below the long-term SMA.
"""

import numpy as np
import pandas as pd
import yfinance as yf

SHORT_WINDOW = 50
LONG_WINDOW = 200
TRADING_DAYS = 252


def download_close(tickers, start, end):
    """
    Download adjusted close prices for one or more tickers.
    Returns a DataFrame with one column per ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].copy()
        close.columns = tickers

    close = close.dropna(how="all")
    return close


def add_moving_averages(df, short=SHORT_WINDOW, long=LONG_WINDOW):
    """Add short-term and long-term simple moving averages to a Close-price DataFrame."""
    df = df.copy()
    df["MA_short"] = df["Close"].rolling(short).mean()
    df["MA_long"] = df["Close"].rolling(long).mean()
    return df


def backtest_golden_cross(df):
    """
    Run the Golden Cross backtest.
    `df` must contain the columns: Close, MA_short, MA_long.
    Returns (signals_df, trades_df).
    """
    df = df.copy()
    df["signal"] = 0

    short = df["MA_short"]
    long = df["MA_long"]
    prev_short = short.shift(1)
    prev_long = long.shift(1)

    # 1 = buy, -1 = sell, 0 = hold
    df.loc[(short > long) & (prev_short <= prev_long), "signal"] = 1
    df.loc[(short < long) & (prev_short >= prev_long), "signal"] = -1

    trades = []
    position = 0
    entry_price = None
    entry_date = None

    for current_date, row in df.iterrows():
        price = row["Close"]
        if pd.isna(price):
            continue

        if row["signal"] == 1 and position == 0:
            position = 1
            entry_price = float(price)
            entry_date = current_date
        elif row["signal"] == -1 and position == 1:
            exit_price = float(price)
            trades.append(
                {
                    "BuyDate": entry_date,
                    "BuyPrice": entry_price,
                    "SellDate": current_date,
                    "SellPrice": exit_price,
                    "HoldingDays": (current_date - entry_date).days,
                    "Profit": exit_price - entry_price,
                    "ReturnPct": (exit_price / entry_price - 1) * 100,
                    "Open": False,
                }
            )
            position = 0
            entry_price = None
            entry_date = None

    # Close any open position at the last available price
    if position == 1 and entry_price is not None:
        last_date = df.index[-1]
        last_price = float(df["Close"].iloc[-1])
        trades.append(
            {
                "BuyDate": entry_date,
                "BuyPrice": entry_price,
                "SellDate": last_date,
                "SellPrice": last_price,
                "HoldingDays": (last_date - entry_date).days,
                "Profit": last_price - entry_price,
                "ReturnPct": (last_price / entry_price - 1) * 100,
                "Open": True,
            }
        )

    trades_df = pd.DataFrame(trades)
    return df, trades_df


def trade_statistics(trades_df):
    """Summarise a trades DataFrame."""
    if trades_df is None or trades_df.empty:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate (%)": 0.0,
            "Average Profit": 0.0,
            "Total Profit": 0.0,
        }

    wins = trades_df[trades_df["Profit"] > 0]
    losses = trades_df[trades_df["Profit"] <= 0]
    return {
        "Total Trades": int(len(trades_df)),
        "Winning Trades": int(len(wins)),
        "Losing Trades": int(len(losses)),
        "Win Rate (%)": round(len(wins) / len(trades_df) * 100, 2),
        "Average Profit": round(float(trades_df["Profit"].mean()), 2),
        "Total Profit": round(float(trades_df["Profit"].sum()), 2),
    }


def portfolio_returns(close_df, weights=None):
    """
    Compute equal-weight (or custom-weight) portfolio daily returns.
    Returns (daily_returns_df, portfolio_return_series, cumulative_curve).
    """
    returns = close_df.pct_change().dropna(how="all")

    if weights is None:
        weights = pd.Series(1.0 / close_df.shape[1], index=close_df.columns)
    else:
        weights = weights.reindex(close_df.columns).fillna(0.0)
        total = weights.sum()
        weights = weights / total if total else weights

    portfolio_return = (returns * weights).sum(axis=1, skipna=True)
    cumulative = (1 + portfolio_return).cumprod()
    return returns, portfolio_return, cumulative


def risk_metrics(portfolio_return):
    """Annualised return, volatility, Sharpe ratio and maximum drawdown."""
    if portfolio_return is None or portfolio_return.empty:
        return {
            "Annual Return (%)": 0.0,
            "Annual Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
        }

    annual_return = portfolio_return.mean() * TRADING_DAYS
    annual_vol = portfolio_return.std() * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_vol if annual_vol else 0.0

    cumulative = (1 + portfolio_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return {
        "Annual Return (%)": round(float(annual_return) * 100, 2),
        "Annual Volatility (%)": round(float(annual_vol) * 100, 2),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Max Drawdown (%)": round(float(drawdown.min()) * 100, 2),
    }