# Claude testing

## Experiments

- [`experiments/lag_propagation`](experiments/lag_propagation) — does
  information take measurable time to propagate across a cross-asset graph,
  and is that lag tradable? Daily FX/commodities/VIX data. See
  [results/report.md](experiments/lag_propagation/results/report.md) for
  findings.
- [`experiments/lag_propagation_intraday`](experiments/lag_propagation_intraday) —
  follow-up at 1-hour and 1-minute resolution (BTC on two venues + ETH),
  with confidence-gated position taking (only trade high-confidence picks).
  See [results/report.md](experiments/lag_propagation_intraday/results/report.md).
