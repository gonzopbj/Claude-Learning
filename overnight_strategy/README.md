# Overnight ("buy close, sell open") strategy backtest

Decomposes each ticker's daily return into two legs and backtests them
separately:

- **overnight**: buy at the close, sell at next day's open
- **intraday**: buy at the open, sell at the close
- **buy&hold**: close-to-close (the two legs compound into this exactly)

## Data

`data_loader.py` tries live data via `yfinance` first. In this sandbox,
all outbound finance-data APIs (Yahoo Finance, Stooq, Alpha Vantage, IEX,
Polygon, Nasdaq Data Link, etc.) are blocked by the network egress proxy,
so it falls back to the CSVs in `data/`, sourced from the (MIT-licensed)
[`backtrader`](https://github.com/mementum/backtrader) sample-data set —
real, unadjusted daily OHLC for **ORCL** (1995-2014), **NVDA** (1999-2014)
and **YHOO** (1996-2015). These are the most recent tickers with genuine
multi-year daily open/close history reachable from this environment.

**Run this on a machine with normal internet access and it will pull
live, current data for any tickers you pass** (small-caps, meme stocks,
ETFs, whatever you want to test):

```bash
pip install -r requirements.txt
python3 run_backtest.py AAPL MSFT TSLA GME IWM SPY
```

## Methodology

- Daily overnight/intraday/close-close returns, compounded into equity curves.
- Sharpe, CAGR, t-stat, hit rate, max drawdown per leg.
- Cost sensitivity: Sharpe/CAGR net of round-trip costs from 0-20 bps/day
  (buy-close-sell-open needs two trades/day, so realistic frictions matter).
- Yearly Sharpe breakdown, to see if the edge is stable or decaying
  (published overnight-anomaly research shows decay after ~2008-2015).
- A transparent "robustness score" combining net-of-cost Sharpe, the
  t-stat, the fraction of profitable years, and first-half vs second-half
  Sharpe decay — used to rank which ticker's overnight edge looks most
  likely to persist, **not** a guarantee of future returns.

## Files

- `data_loader.py` — live/local data loading
- `backtest.py` — leg decomposition, stats, cost sensitivity
- `plots.py` — chart generation
- `run_backtest.py` — main entry point
- `results/` — generated CSVs, JSON stats, and PNG charts (git-ignored inputs
  regenerate on every run)
