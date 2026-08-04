"""
Intraday data ingestion: BTC (two venues) and ETH minute bars.

Data source note
-----------------
Same egress constraint as the daily experiment (Yahoo/FRED/exchange APIs all
blocked). No public GitHub-hosted repo bundles multi-asset intraday data as
plain files at usable scale -- but one does via Git LFS:
Gendo90/Crypto-Historical-Prices ships real Kaggle-sourced 1-minute OHLCV for
Bitcoin (Coinbase AND Bitstamp -- two venues of the *same* asset, useful for a
cross-exchange-arbitrage sanity check) and Ethereum. LFS blobs aren't served
by raw.githubusercontent.com (that returns the tiny pointer file) but are
served by media.githubusercontent.com, which the sandbox's egress policy does
allow.

Overlap window across all three series: 2016-05-09 to 2019-01-08 (~2.7 years
of 1-minute bars, ~1.4M rows) -- that's the universe below.
"""
import os

import numpy as np
import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

LFS_BASE = "https://media.githubusercontent.com/media/Gendo90/Crypto-Historical-Prices/master"
SOURCES = {
    "btc_coinbase": f"{LFS_BASE}/Bitcoin/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",
    "btc_bitstamp": f"{LFS_BASE}/Bitcoin/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv",
    "eth": f"{LFS_BASE}/Ethereum/ETH_1min.csv",
}

OVERLAP_START = "2016-05-09"
OVERLAP_END = "2019-01-08"

ALL_ASSETS = ["btc_coinbase", "btc_bitstamp", "eth"]
TRADABLE_ASSETS = ALL_ASSETS  # all three are directly tradable (spot)


def _fetch(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if os.path.exists(path):
        return path
    with requests.get(SOURCES[name], stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def _load_btc(path: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["Timestamp", "Close"])
    df["ts"] = pd.to_datetime(df["Timestamp"], unit="s")
    s = df.set_index("ts")["Close"].sort_index()
    return s


def _load_eth(path: str) -> pd.Series:
    # NB: the "Unix Timestamp" column switches units partway through the file
    # (seconds up to 2018-08-23, milliseconds from 2018-08-25 on) -- some
    # upstream export changed format mid-history. The "Date" string column is
    # consistent throughout, so parse that instead.
    df = pd.read_csv(path, usecols=["Date", "Close"])
    df["ts"] = pd.to_datetime(df["Date"])
    s = df.set_index("ts")["Close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def build_minute_panel(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh:
        for name in SOURCES:
            p = os.path.join(RAW_DIR, f"{name}.csv")
            if os.path.exists(p):
                os.remove(p)

    cb_path = _fetch("btc_coinbase")
    bs_path = _fetch("btc_bitstamp")
    eth_path = _fetch("eth")

    cb = _load_btc(cb_path)
    bs = _load_btc(bs_path)
    eth = _load_eth(eth_path)

    panel = pd.DataFrame({"btc_coinbase": cb, "btc_bitstamp": bs, "eth": eth})
    panel = panel.loc[OVERLAP_START:OVERLAP_END]

    # reindex to a complete 1-minute grid, then forward-fill short gaps
    # (illiquid minutes with no trade -- both BTC venues have plenty; ETH's
    # own file already carries price forward for no-trade minutes).
    full_idx = pd.date_range(panel.index.min(), panel.index.max(), freq="min")
    panel = panel.reindex(full_idx)
    panel = panel.ffill(limit=15)
    panel = panel.dropna(how="any")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    panel.to_parquet(os.path.join(PROCESSED_DIR, "minute_panel.parquet"))
    return panel


def build_hour_panel(minute_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if minute_panel is None:
        minute_panel = build_minute_panel()
    hourly = minute_panel.resample("h").last().dropna(how="any")
    hourly.to_parquet(os.path.join(PROCESSED_DIR, "hour_panel.parquet"))
    return hourly


RETURN_CLIP = 0.20  # generous vs. real 1-min crypto vol (~0.05-0.1%); only catches bad ticks


def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple returns, clipped at +-20%/bar.

    This Kaggle-sourced minute data has a handful of single-tick data errors
    (e.g. BTC printing $0.06 for one minute on 2017-04-15 before snapping
    back), which without clipping produce +-1,000,000% one-bar "returns"
    that dominate every correlation and backtest touching that asset.
    """
    rets = panel.pct_change().dropna(how="all")
    return rets.clip(-RETURN_CLIP, RETURN_CLIP)


if __name__ == "__main__":
    m = build_minute_panel()
    print("minute panel:", m.shape, m.index.min(), m.index.max())
    h = build_hour_panel(m)
    print("hour panel:", h.shape, h.index.min(), h.index.max())
    print(m.describe())
