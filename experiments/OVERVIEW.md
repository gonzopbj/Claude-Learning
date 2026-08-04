# Overview: lag-propagation trading experiments

This document explains the setup shared by both experiments in this
directory — what question they test, where the data comes from, and exactly
how the discovery scan, backtest, confidence gating, event-driven sizing,
and validation checks work. It's the methodology reference; each
experiment's own `results/report.md` has the numbers and findings.

- [`lag_propagation/`](lag_propagation) — daily FX, commodities, VIX
  (1999–2026)
- [`lag_propagation_intraday/`](lag_propagation_intraday) — hourly and
  1-minute BTC (two venues) + ETH (2016–2019)

## 1. The question

A dynamic-causal-graph trading pitch usually reduces to one falsifiable
claim: **when one asset moves, related assets don't fully reprice
instantly — there's a lag, and that lag is tradable.** These experiments
test that claim directly, end to end: does a measurable propagation delay
exist in real market data, is it strong enough to survive multiple-testing
scrutiny, and does a strategy built on it survive realistic trading costs?
Everything below is built to give an honest answer to that last part
specifically — it would have been easy to report only the parts that look
good.

Two rounds:

1. **Daily** — cast a wide net (24 assets, cross-correlation scan, simple
   walk-forward backtest) to see if the effect exists at all.
2. **Intraday** — follow-up once the daily result showed a real-but-weak,
   1-day effect that didn't survive costs. Tests whether the true
   propagation window is actually shorter (hours/minutes, not days), adds
   confidence gating (only trade the strongest picks), and adds
   event-driven position sizing (react to specific triggers instead of
   continuously re-pricing every bar).

## 2. Data

### Why this specific data

This runs in a sandboxed environment whose network egress is restricted to
an allowlist. Direct checks (`curl` against the proxy) confirmed Yahoo
Finance, FRED, Alpha Vantage, Stooq, and crypto exchange APIs (Binance,
Kraken) all return `403` — blocked by policy, not down. `github.com`,
`raw.githubusercontent.com`, `api.github.com`, and `media.githubusercontent.com`
(GitHub's Git-LFS content host) are reachable. That constraint, not
convenience, is why both experiments source data from public GitHub repos
rather than a financial data API.

### Daily experiment: FX + commodities + volatility

| Asset class | Source | Assets | History |
|---|---|---|---|
| FX (USD exchange rates) | `datasets/exchange-rates` (GitHub org, FRED-sourced) | 21 countries: Australia, Brazil, Canada, China, Denmark, Euro, Hong Kong, India, Japan, Malaysia, Mexico, New Zealand, Norway, Singapore, South Africa, South Korea, Sweden, Switzerland, Taiwan, Thailand, UK | 1999-01-04 → 2026-07-29 |
| Commodities | `datasets/oil-prices` | WTI, Brent crude | same |
| Volatility | `datasets/finance-vix` | VIX | same |

24 assets total, daily bars, ~7,190 trading days. VIX is a **driver-only**
node — it can predict other assets but isn't traded itself (no futures-roll
model here). Venezuela's FX series is excluded (hyperinflation regime
breaks make its returns non-comparable). Data cleaning: WTI's famous
2020-04-20 negative-price print turns a single day's `pct_change()` into a
±300% "return"; oil returns are clipped to ±30% (still ~11 standard
deviations) so that one contract-roll mechanical artifact doesn't dominate
every correlation touching oil.

### Intraday experiment: BTC (two venues) + ETH

| Asset | Source | Venue | Granularity |
|---|---|---|---|
| `btc_coinbase` | `Gendo90/Crypto-Historical-Prices` (GitHub, Git LFS) | Coinbase | 1-minute |
| `btc_bitstamp` | same repo | Bitstamp | 1-minute |
| `eth` | same repo | (ETHUSD) | 1-minute |

Fetched via `media.githubusercontent.com` specifically — Git LFS blobs
aren't served by `raw.githubusercontent.com` (that returns a small pointer
file instead of the actual content). Overlap window across all three series:
2016-05-09 to 2019-01-08 (~2.7 years), giving 1,362,866 one-minute bars,
resampled to 22,743 hourly bars for the coarser-frequency run. Bitcoin on
two venues (Coinbase and Bitstamp) is deliberate: it's literally the same
asset on two order books, so any discovered lag between them has a known,
checkable real-world interpretation (cross-exchange arbitrage) — a built-in
sanity check on whether the method finds real structure or noise.

Cleaning: the raw minute data has a handful of single-tick errors (e.g. BTC
printing $0.06 for one minute on 2017-04-15 before snapping back) that
otherwise produce ~+1,000,000% one-bar "returns." All returns are clipped
to ±20%/bar (real 1-minute crypto volatility is ~0.05–0.1%, so this only
catches genuine data errors).

## 3. Step 1: lag discovery (does a delay exist at all?)

`discovery.py` (identical logic in both experiments) answers this in
isolation, before any trading question: for every **ordered** pair of
assets (driver → target, so A→B and B→A are tested separately) and every
lag *k* from 1 up to `max_lag` bars, it Pearson-correlates the driver's
return at time *t* against the target's return at time *t+k*, using the
**entire** sample (not walk-forward — this step is purely "is there
structure here," not a trading signal yet).

That's `n_assets × (n_assets−1) × max_lag` independent tests — 5,520 for
the daily 24-asset universe, 60 for the 3-asset intraday universe. Testing
that many hypotheses at once means some will look "significant" by chance
alone, so every p-value is corrected via **Benjamini-Hochberg false
discovery rate (FDR) control** (`statsmodels.stats.multitest.multipletests`,
q < 0.05) before anything is called significant. The headline outputs are:
the fraction of pairs with at least one FDR-significant lag, and the
distribution of each pair's *strongest* significant lag — this second
number is the direct, quantified answer to "how many bars does information
take to propagate here."

## 4. Step 2: the walk-forward backtest (is it tradable?)

This is the core piece shared by both experiments (`backtest.py`), and the
part built specifically to avoid the standard ways this kind of study lies
to itself.

### No lookahead, by construction

The backtest never uses the discovery scan's full-sample correlations to
trade. Instead, every `refresh` bars it **re-runs the same correlation scan
using only the trailing `window` bars** — data the strategy could actually
have seen at that point in time. `compute_rebalance_windows` walks forward
through the whole sample building one lagged-correlation tensor per
rebalance date; `select_assignments` then picks, for each tradable target,
the driver/lag combination with the strongest `|correlation|` in that
trailing window only (`mode="best"`; there's also `mode="random"` for the
permutation test below).

The correlation computation itself (`_lagged_corr_matrices`) is a small
piece of applied linear algebra worth noting: rather than looping over
every driver/lag/target triple with `scipy.stats.pearsonr` (way too slow at
scale — this was measured and rejected), it standardizes the trailing
window once per lag and computes the full driver×target correlation matrix
for that lag as a single matrix product (`standardized_early.T @
standardized_late`). That's what makes a 940-rebalance, 3-asset, 60-lag
minute-frequency walk-forward run in under a second instead of minutes.

### Turning a pick into a position: two different models

**Continuous sizing** (`signals_from_rebalances`, the original approach):
every bar, a traded target's position is resized to
`sign(correlation) × clip(z-score of the driver's return `lag` bars ago,
±3)`. This means the position changes *every single bar* the assignment is
active, since the driver's realized return input is different each bar.

**Event-driven sizing** (`signals_from_rebalances_event_driven`, added in
the intraday follow-up): instead of continuously resizing, a fixed-size
directional bet only *opens* when the driver's z-scored return crosses an
`entry_z` threshold, holds for exactly `lag` bars (the propagation window
the discovery scan found), then flattens. A new trigger while a position is
already open overwrites it. This was added specifically because continuous
sizing was found to generate turnover roughly proportional to bar count
regardless of whether there was a real signal that bar — expensive at daily
frequency, catastrophic at 1-minute frequency (see §6). It's implemented
without a per-bar Python loop: a forward-fill via `np.maximum.accumulate`
on trigger indices carries each trigger's value forward for exactly `lag`
bars, vectorized across the whole block at once.

### From position to portfolio return

`raw_signal_to_portfolio` turns either kind of raw per-target signal into a
backtested return series:

1. **Normalize** — each day's raw signal values are scaled so gross
   exposure (`Σ|weight|`) sums to 1 whenever at least one target is active.
2. **Vol-target** — the portfolio is leveraged up or down so its trailing
   realized volatility matches a target (8%/year daily experiment,
   20%/year intraday), using only *past* realized vol (`rolling().std()`
   shifted by one bar — no lookahead), capped at 3x leverage.
3. **Cost** — a flat cost per unit of turnover (`Σ|Δweight|` that day) is
   charged: 3bps in the daily experiment (FX/commodities), 5bps in the
   intraday one (crypto taker fees run higher). The intraday experiment
   also reports a **cost-sensitivity sweep** (0/1/2/3/5bps) and, for
   event-driven sizing, a **breakeven cost** computed directly from each
   config's actual annualized return and turnover — the flat per-turnover
   cost that would exactly cancel it out.

### Baselines every result has to beat

- **Time-series momentum**: each tradable asset traded off the sign of its
  *own* trailing cumulative return (21 days daily; 24h/60min intraday) — the
  same portfolio construction machinery, just a different signal. This is
  the direct answer to "is cross-asset lag actually adding anything over
  just trading each asset's own trend."
- **Buy-and-hold**: a static equal-weight basket, no rebalancing signal at
  all.

## 5. Step 3: confidence gating (should every pick get traded?)

Trading every target the discovery scan finds a "best" driver for dilutes
strong signals with weak ones — the daily experiment's ungated result
traded every pick uniformly regardless of how weak its correlation was.
`apply_confidence_gate(records, min_abs_corr)` fixes this by dropping any
rebalance-period assignment whose `|correlation|` falls below a threshold —
that target sits out (flat) for the period instead. Thresholds are set from
the **empirical distribution of picks actually made** over the run
(`pick_confidence_distribution`), not arbitrary numbers: median, 75th, and
90th percentile gates are tested against the ungated baseline, so "how
selective" is measured in the same units the strategy actually produces.

## 6. Step 4: is any of this more than multiple-testing luck?

Scanning ~230 candidate driver/lag combinations per target per rebalance
will find a "best" one even if nothing in the data is real — that's exactly
what the FDR correction in §3 is for at the discovery-scan level, but the
backtest needs its own version of the same check. The **permutation-null
control** re-runs the identical backtest mechanics, but replaces "pick the
best-correlated driver/lag" with "pick a *uniformly random* one from the
same candidate pool" (`select_assignments(mode="random")`), 200 times in
the daily experiment (50 at minute frequency, where each draw costs ~5s at
1.36M rows). If the real, best-pick strategy's Sharpe doesn't clearly separate
from this null distribution, the "edge" is indistinguishable from what
scanning that many candidates produces by chance alone. Where the real
strategy lands in that null distribution (e.g. "100th percentile of 200
random draws") is the headline number in both reports.

## 7. Parameters at a glance

| | Daily | Hourly (intraday) | Minute (intraday) |
|---|---:|---:|---:|
| Trailing window | 252 bars (~1yr) | 720 bars (30d) | 10,080 bars (1wk) |
| Rebalance frequency | every 21 bars (~monthly) | every 168 bars (weekly) | every 1,440 bars (daily) |
| Max lag scanned | 10 bars | 24 bars (1 day) | 60 bars (1 hour) |
| Cost assumption | 3bps/turnover | 5bps/turnover | 5bps/turnover |
| Vol target (annualized) | 8% | 20% | 20% |
| Max leverage | 3x | 3x | 3x |
| Permutation draws | 200 | 200 | 50 |
| Position sizing tested | continuous only | continuous + event-driven | continuous + event-driven |

## 8. What's *not* modeled (limitations to keep in mind)

- **Costs are a flat bps-of-turnover assumption**, not a real order-book/
  slippage/market-impact model. Real execution costs vary with size and
  urgency in ways this doesn't capture.
- **VIX and the intraday universe's driver-only status** aside, every
  tradable asset is assumed frictionlessly shortable at the same cost as
  going long — true for spot FX and most liquid crypto, not universally true.
- **3-node intraday universe**: BTC on two venues plus ETH is enough to get
  a mechanistically interpretable result (cross-exchange arbitrage, BTC→ETH
  lead-lag) but gives the permutation-null test very little room to
  distinguish "best pick" from "random pick" — noted explicitly in the
  intraday report as a reason its null-percentile result is weaker than the
  daily experiment's despite much stronger raw correlations.
- **No market-impact or capacity modeling** — nothing here says how much
  capital any of this could actually absorb even if it were profitable.
