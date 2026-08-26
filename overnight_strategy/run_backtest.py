"""Backtest 'buy at close, sell at open' (overnight) vs intraday vs
buy-and-hold, across a list of tickers, and rank tickers by how
statistically robust / likely-to-persist the overnight edge looks.

Usage:
    python3 run_backtest.py                       # uses DEFAULT_TICKERS
    python3 run_backtest.py AAPL MSFT TSLA GME     # live data via yfinance
"""
import os
import sys
import json
import pandas as pd

from data_loader import load_prices
from backtest import (
    compute_legs, perf_stats, yearly_stats, cost_sensitivity,
    breakeven_cost_bps, equity_curve,
)
from plots import make_all_plots

DEFAULT_TICKERS = ["ORCL", "NVDA", "YHOO"]
COST_GRID_BPS = [0, 2, 5, 10, 20]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def analyze_ticker(ticker: str) -> dict:
    raw = load_prices(ticker)
    df = compute_legs(raw)

    legs = {
        "overnight (close->open)": df["overnight_ret"],
        "intraday (open->close)": df["intraday_ret"],
        "buy&hold (close->close)": df["closeclose_ret"],
    }
    stats = {name: perf_stats(r) for name, r in legs.items()}

    yr = yearly_stats(df["overnight_ret"])
    first_half = yr.iloc[: len(yr) // 2]["sharpe"].mean()
    second_half = yr.iloc[len(yr) // 2:]["sharpe"].mean()
    pct_years_positive = (yr["sharpe"] > 0).mean()

    costs = cost_sensitivity(df["overnight_ret"], COST_GRID_BPS)
    breakeven = breakeven_cost_bps(df["overnight_ret"])

    return {
        "ticker": ticker,
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_days": len(df),
        "stats": stats,
        "yearly": yr,
        "first_half_sharpe": first_half,
        "second_half_sharpe": second_half,
        "pct_years_positive": pct_years_positive,
        "cost_sensitivity": costs,
        "breakeven_bps": breakeven,
        "df": df,
    }


def print_summary(result: dict):
    t = result["ticker"]
    print(f"\n{'=' * 70}\n{t}  ({result['date_range']}, n={result['n_days']} trading days)\n{'=' * 70}")
    rows = []
    for leg, s in result["stats"].items():
        rows.append({
            "leg": leg,
            "CAGR": f"{s['cagr']:.2%}",
            "AnnVol": f"{s['ann_vol']:.2%}",
            "Sharpe": f"{s['sharpe']:.2f}",
            "t-stat": f"{s['tstat']:.2f}",
            "HitRate": f"{s['hit_rate']:.1%}",
            "MaxDD": f"{s['max_drawdown']:.1%}",
            "MeanDaily(bps)": f"{s['mean_daily_bps']:.2f}",
        })
    print(pd.DataFrame(rows).set_index("leg").to_string())
    print(f"\nOvernight leg stability: first-half Sharpe={result['first_half_sharpe']:.2f}  "
          f"second-half Sharpe={result['second_half_sharpe']:.2f}  "
          f"%years positive={result['pct_years_positive']:.0%}")
    print(f"Breakeven round-trip cost (overnight leg): {result['breakeven_bps']:.2f} bps/day")
    print("\nCost sensitivity (overnight leg):")
    print(result["cost_sensitivity"].to_string())


def build_ranking(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        s = r["stats"]["overnight (close->open)"]
        net10 = cost_sensitivity(r["df"]["overnight_ret"], [10]).loc[10]
        rows.append({
            "ticker": r["ticker"],
            "gross_sharpe": s["sharpe"],
            "tstat": s["tstat"],
            "net_sharpe_10bps": net10["sharpe"],
            "pct_years_positive": r["pct_years_positive"],
            "first_half_sharpe": r["first_half_sharpe"],
            "second_half_sharpe": r["second_half_sharpe"],
            "breakeven_bps": r["breakeven_bps"],
        })
    rank_df = pd.DataFrame(rows).set_index("ticker")

    def norm(col):
        c = rank_df[col]
        rng = c.max() - c.min()
        return (c - c.min()) / rng if rng > 0 else c * 0 + 0.5

    rank_df["robustness_score"] = (
        0.35 * norm("net_sharpe_10bps")
        + 0.25 * norm("tstat")
        + 0.25 * norm("pct_years_positive")
        + 0.15 * (1 - (rank_df["first_half_sharpe"] - rank_df["second_half_sharpe"]).clip(lower=0)
                  / (rank_df["first_half_sharpe"].abs().max() + 1e-9))
    )
    return rank_df.sort_values("robustness_score", ascending=False)


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for t in tickers:
        try:
            results.append(analyze_ticker(t))
        except Exception as e:
            print(f"Skipping {t}: {e}")

    for r in results:
        print_summary(r)

    ranking = build_ranking(results)
    print(f"\n{'=' * 70}\nROBUSTNESS RANKING (overnight leg, higher = more likely edge persists)\n{'=' * 70}")
    print(ranking.to_string())

    ranking.to_csv(os.path.join(RESULTS_DIR, "ranking.csv"))
    make_all_plots(results, ranking, RESULTS_DIR)
    for r in results:
        r["yearly"].to_csv(os.path.join(RESULTS_DIR, f"{r['ticker']}_yearly.csv"))
        r["cost_sensitivity"].to_csv(os.path.join(RESULTS_DIR, f"{r['ticker']}_cost_sensitivity.csv"))
        summary_rows = {leg: s for leg, s in r["stats"].items()}
        with open(os.path.join(RESULTS_DIR, f"{r['ticker']}_stats.json"), "w") as f:
            json.dump(summary_rows, f, indent=2, default=float)

    return results, ranking


if __name__ == "__main__":
    main()
