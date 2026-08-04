"""
Data ingestion for the lag-propagation experiment.

Data source note
-----------------
This sandbox's egress policy blocks Yahoo Finance, FRED, Alpha Vantage, Stooq,
and general web hosts (verified: 403 at the proxy). The only reachable market
data we found is the `datasets` GitHub org, served over raw.githubusercontent.com
(itself ultimately sourced from FRED). That fixes our universe to what those
mirrors carry: daily USD exchange rates for 21 countries (1971/1999-2026),
WTI and Brent crude (1986/1987-2026), and VIX (1990-2026). No equities.

This means the experiment below tests the lag/propagation-delay hypothesis on
an FX + commodities + volatility graph, not stocks. That's a fine graph for
the question ("does information take measurable time to move between related
markets"), just flag the universe when interpreting results.
"""
import io
import os

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

SOURCES = {
    "fx": "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv",
    "wti": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv",
    "brent": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
    "vix": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
}

# Countries with a continuous USD exchange-rate series covering 1999-2026.
# Venezuela excluded (starts 2000-01-03, and hyperinflation regime breaks
# make its return series non-comparable to the rest anyway).
FX_COUNTRIES = [
    "Australia", "Brazil", "Canada", "China", "Denmark", "Euro", "Hong Kong",
    "India", "Japan", "Malaysia", "Mexico", "New Zealand", "Norway",
    "Singapore", "South Africa", "South Korea", "Sweden", "Switzerland",
    "Taiwan", "Thailand", "United Kingdom",
]

START_DATE = "1999-01-04"  # Euro's first print; all other series pre-date this
END_DATE = None  # use everything available


def _fetch(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if os.path.exists(path):
        return path
    resp = requests.get(SOURCES[name], timeout=30)
    resp.raise_for_status()
    with open(path, "w") as f:
        f.write(resp.text)
    return path


def build_price_panel(force_refresh: bool = False) -> pd.DataFrame:
    """Return a wide DataFrame indexed by Date, one price-level column per asset.

    FX columns are named 'fx_<Country>' (USD exchange rate, FRED convention:
    for most countries this is local-currency-per-USD; EUR/GBP/AUD/NZD are
    USD-per-unit -- direction is irrelevant here since we only use returns of
    each series against itself/each other, not a global "USD strength" sign).
    """
    if force_refresh:
        for name in SOURCES:
            p = os.path.join(RAW_DIR, f"{name}.csv")
            if os.path.exists(p):
                os.remove(p)

    fx_path = _fetch("fx")
    wti_path = _fetch("wti")
    brent_path = _fetch("brent")
    vix_path = _fetch("vix")

    fx = pd.read_csv(fx_path, parse_dates=["Date"])
    fx = fx[fx["Country"].isin(FX_COUNTRIES)]
    fx_panel = fx.pivot(index="Date", columns="Country", values="Exchange rate")
    fx_panel.columns = [f"fx_{c.replace(' ', '_')}" for c in fx_panel.columns]

    wti = pd.read_csv(wti_path, parse_dates=["Date"]).set_index("Date")["Price"]
    wti.name = "cmd_WTI"

    brent = pd.read_csv(brent_path, parse_dates=["Date"]).set_index("Date")["Price"]
    brent.name = "cmd_Brent"

    vix = pd.read_csv(vix_path, parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    vix.index.name = "Date"
    vix.name = "vol_VIX"

    panel = fx_panel.join([wti, brent, vix], how="outer")
    panel = panel.sort_index()
    panel = panel.loc[START_DATE:END_DATE]

    # Different national holiday calendars mean occasional idiosyncratic gaps;
    # forward-fill short ones, then drop any date where a genuine gap remains.
    panel = panel.ffill(limit=2)
    panel = panel.dropna(how="any")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    panel.to_csv(os.path.join(PROCESSED_DIR, "prices.csv"))
    return panel


def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) daily returns.

    Not log returns: WTI printed a negative price on 2020-04-20 (the
    well-known negative-oil-futures day), which makes log(price) undefined.
    Simple returns handle that fine and are what we actually trade on.
    """
    rets = panel.pct_change().dropna(how="all")
    # WTI printed -$36.98 on 2020-04-20 (front-month contract-roll mechanics,
    # not a spot-price move); pct_change turns that into a +-300% one-day
    # "return" that would dominate every correlation/backtest touching oil.
    # Clip oil returns to a still-generous +-30% band (~11 std devs) so that
    # single mechanical artifact doesn't drive the results.
    for col in ("cmd_WTI", "cmd_Brent"):
        if col in rets.columns:
            rets[col] = rets[col].clip(-0.30, 0.30)
    return rets


TRADABLE_ASSETS = [c for c in [f"fx_{c.replace(' ', '_')}" for c in FX_COUNTRIES]] + [
    "cmd_WTI",
    "cmd_Brent",
]
# vol_VIX is kept as a driver-only node: not directly tradable without
# futures-roll/contango modeling we're not doing here.
DRIVER_ONLY_ASSETS = ["vol_VIX"]
ALL_ASSETS = TRADABLE_ASSETS + DRIVER_ONLY_ASSETS


if __name__ == "__main__":
    panel = build_price_panel()
    print(panel.shape, panel.index.min(), panel.index.max())
    print(panel.columns.tolist())
    rets = compute_returns(panel)
    print(rets.describe().T[["mean", "std"]])
