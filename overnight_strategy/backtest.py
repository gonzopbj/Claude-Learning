"""Core backtest engine for the close-to-open (overnight) strategy.

Decomposes each trading day into two legs:
  overnight_ret[t] = Open[t] / Close[t-1] - 1   (buy close, sell open)
  intraday_ret[t]  = Close[t] / Open[t]  - 1    (buy open, sell close)
  closeclose_ret[t]= Close[t] / Close[t-1] - 1  (buy & hold benchmark)

overnight_ret and intraday_ret compound exactly to closeclose_ret.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_legs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["Close"].shift(1)
    out["overnight_ret"] = out["Open"] / prev_close - 1
    out["intraday_ret"] = out["Close"] / out["Open"] - 1
    out["closeclose_ret"] = out["Close"] / prev_close - 1
    return out.dropna(subset=["overnight_ret", "intraday_ret", "closeclose_ret"])


def apply_cost(returns: pd.Series, round_trip_bps: float) -> pd.Series:
    """Subtract a round-trip transaction cost (in bps) from every trade."""
    return returns - round_trip_bps / 10_000.0


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def perf_stats(returns: pd.Series, freq: int = TRADING_DAYS) -> dict:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {}
    mean = returns.mean()
    std = returns.std(ddof=1)
    eq = equity_curve(returns)
    total_return = eq.iloc[-1] - 1
    years = n / freq
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    ann_vol = std * np.sqrt(freq)
    sharpe = (mean / std) * np.sqrt(freq) if std > 0 else np.nan
    tstat = (mean / std) * np.sqrt(n) if std > 0 else np.nan
    hit_rate = (returns > 0).mean()
    mdd = max_drawdown(eq)
    return {
        "n_days": n,
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "tstat": tstat,
        "hit_rate": hit_rate,
        "max_drawdown": mdd,
        "mean_daily_bps": mean * 10_000,
    }


def yearly_stats(returns: pd.Series) -> pd.DataFrame:
    """Per-calendar-year Sharpe/mean, used to check whether the edge is
    stable over time or decaying."""
    rows = []
    for year, chunk in returns.groupby(returns.index.year):
        s = perf_stats(chunk)
        if s:
            rows.append({"year": year, "sharpe": s["sharpe"], "mean_daily_bps": s["mean_daily_bps"],
                         "hit_rate": s["hit_rate"], "n_days": s["n_days"]})
    return pd.DataFrame(rows).set_index("year")


def cost_sensitivity(returns: pd.Series, bps_grid) -> pd.DataFrame:
    rows = []
    for bps in bps_grid:
        net = apply_cost(returns, bps)
        s = perf_stats(net)
        rows.append({"round_trip_bps": bps, "sharpe": s.get("sharpe"), "cagr": s.get("cagr"),
                     "total_return": s.get("total_return")})
    return pd.DataFrame(rows).set_index("round_trip_bps")


def breakeven_cost_bps(returns: pd.Series) -> float:
    """Round-trip cost (bps) at which mean daily return hits zero."""
    return returns.mean() * 10_000
