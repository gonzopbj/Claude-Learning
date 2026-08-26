"""Load daily OHLC price data for a ticker.

Tries live data via yfinance first (works when this machine has open
internet access). Falls back to a local CSV in data/<TICKER>.csv, which
must have Date,Open,High,Low,Close[,Adj Close,Volume] columns.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_prices(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            return df
    except Exception:
        pass

    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No live data available for {ticker} and no local fallback at {csv_path}"
        )
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df
