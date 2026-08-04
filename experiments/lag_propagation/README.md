# Lag-propagation trading experiment

Tests one narrow, falsifiable piece of the dynamic-causal-graph trading idea:
does information take a measurable amount of time to propagate between
related markets, and is that delay tradable?

**Results: [results/report.md](results/report.md)** — full writeup with
plots. Short version: the effect is statistically real (permutation test,
~99.5th–100th percentile vs. random selection) but resolves within ~1
trading day and doesn't survive realistic transaction costs. Buy-and-hold
and plain momentum both beat it over 1999–2026.

## Layout

- `data.py` — downloads and caches a cross-asset daily panel (21 FX rates,
  WTI, Brent, VIX). See the module docstring for why this specific universe:
  the sandbox's egress policy blocks Yahoo Finance / FRED / Stooq / Alpha
  Vantage, so this uses the one reachable source (`datasets` GitHub org).
- `discovery.py` — full-sample lagged-correlation scan with Benjamini-
  Hochberg FDR correction across all pair×lag tests.
- `backtest.py` — walk-forward (no-lookahead) strategy that trades each
  asset off its best trailing-window lagged driver, refreshed monthly, plus
  momentum/buy-hold baselines and the permutation-null control.
- `run.py` — runs the whole pipeline end to end and writes `results/`.

## Run it

```
pip install -r ../../requirements.txt
python3 run.py
```

Takes about 2 minutes, dominated by the 200-permutation null test. Writes
`results/summary.json` and the PNGs referenced in the report.
