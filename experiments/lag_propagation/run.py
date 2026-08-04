"""
Orchestrates the full experiment and writes results/ (JSON summary, plots, report.md).

Run: python3 run.py
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as B
import discovery as D
from data import ALL_ASSETS, TRADABLE_ASSETS, build_price_panel, compute_returns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MANAGED_FLOAT = ["fx_Malaysia", "fx_China", "fx_Hong_Kong"]  # capital controls / hard or soft USD pegs for part of the sample
N_PERMUTATIONS = 200
SEED = 42


def run_discovery(rets: pd.DataFrame, assets: list[str], label: str) -> dict:
    scan = D.full_sample_scan(rets, assets, max_lag=D.MAX_LAG)
    scan_fdr = D.add_fdr(scan)
    n_pairs = len(assets) * (len(assets) - 1)
    summary = D.summarize(scan_fdr, scan, n_pairs, scan.shape[0])
    summary["label"] = label
    return summary, scan_fdr


def run_backtest_suite(rets: pd.DataFrame, assets: list[str], tradable_assets: list[str], label: str,
                        n_permutations: int = N_PERMUTATIONS) -> dict:
    records, R, tradable_idx = B.compute_rebalance_windows(rets, assets, tradable_assets)
    target_rets = R[:, tradable_idx]

    best_assign = B.select_assignments(records, tradable_idx, len(assets), mode="best")
    raw = B.signals_from_rebalances(best_assign, R, tradable_idx)
    port = B.raw_signal_to_portfolio(raw, target_rets)
    metrics_full = B.performance_metrics(port["strategy_return"])

    port_nocost = B.raw_signal_to_portfolio(raw, target_rets, cost_bps=0.0)
    metrics_nocost = B.performance_metrics(port_nocost["strategy_return"])

    raw_smooth = pd.DataFrame(raw).ewm(halflife=3, adjust=False).mean().to_numpy()
    port_smooth = B.raw_signal_to_portfolio(raw_smooth, target_rets)
    metrics_smooth = B.performance_metrics(port_smooth["strategy_return"])

    mom = B.momentum_baseline(rets, tradable_assets)
    metrics_mom = B.performance_metrics(mom["strategy_return"])

    bh = B.buy_hold_baseline(rets, tradable_assets)
    metrics_bh = B.performance_metrics(bh["strategy_return"])

    rng = np.random.default_rng(SEED)
    null_sharpes = []
    t0 = time.time()
    for i in range(n_permutations):
        rand_assign = B.select_assignments(records, tradable_idx, len(assets), mode="random", rng=rng)
        raw_r = B.signals_from_rebalances(rand_assign, R, tradable_idx)
        port_r = B.raw_signal_to_portfolio(raw_r, target_rets)
        null_sharpes.append(B.performance_metrics(port_r["strategy_return"])["sharpe"])
    perm_time = time.time() - t0

    null_sharpes = np.array(null_sharpes)
    real_sharpe = metrics_full["sharpe"]
    percentile = float((null_sharpes < real_sharpe).mean() * 100)

    return {
        "label": label,
        "strategy": metrics_full,
        "strategy_no_cost": metrics_nocost,
        "strategy_smoothed": metrics_smooth,
        "momentum_baseline": metrics_mom,
        "buy_hold_baseline": metrics_bh,
        "n_permutations": n_permutations,
        "permutation_runtime_sec": round(perm_time, 1),
        "null_sharpe_mean": float(null_sharpes.mean()),
        "null_sharpe_std": float(null_sharpes.std()),
        "null_sharpe_p05": float(np.percentile(null_sharpes, 5)),
        "null_sharpe_p50": float(np.percentile(null_sharpes, 50)),
        "null_sharpe_p95": float(np.percentile(null_sharpes, 95)),
        "real_sharpe_percentile_in_null": percentile,
        "_equity_curves": {
            "strategy": (1 + port["strategy_return"]).cumprod(),
            "strategy_smoothed": (1 + port_smooth["strategy_return"]).cumprod(),
            "momentum": (1 + mom["strategy_return"]).cumprod(),
            "buy_hold": (1 + bh["strategy_return"]).cumprod(),
        },
        "_null_sharpes": null_sharpes,
        "_turnover_mean": float(port["turnover"].mean()),
    }


def plot_equity_curves(equity_curves: dict, dates: pd.DatetimeIndex, path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in equity_curves.items():
        ax.plot(dates, curve, label=name, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lag_histogram(lag_dist: dict, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = sorted(int(k) for k in lag_dist.keys())
    counts = [lag_dist[str(k)] if str(k) in lag_dist else lag_dist.get(k, 0) for k in lags]
    ax.bar(lags, counts, color="#4C72B0")
    ax.set_xlabel("Best significant lag (trading days)")
    ax.set_ylabel("Number of asset pairs")
    ax.set_title(title)
    ax.set_xticks(lags)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_null_distribution(null_sharpes: np.ndarray, real_sharpe: float, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_sharpes, bins=30, color="#888888", alpha=0.8, label="Random driver/lag pick (null)")
    ax.axvline(real_sharpe, color="crimson", linewidth=2, label=f"Best-pick strategy (Sharpe={real_sharpe:.2f})")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Permutations")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    panel = build_price_panel()
    rets_all = compute_returns(panel)[ALL_ASSETS].dropna()

    universes = {
        "full": (ALL_ASSETS, TRADABLE_ASSETS),
        "ex_managed_fx": (
            [a for a in ALL_ASSETS if a not in MANAGED_FLOAT],
            [a for a in TRADABLE_ASSETS if a not in MANAGED_FLOAT],
        ),
    }

    all_results = {}
    for key, (assets, tradable) in universes.items():
        print(f"=== universe: {key} ({len(assets)} assets, {len(tradable)} tradable) ===")
        disc_summary, _ = run_discovery(rets_all[assets], assets, key)
        bt_summary = run_backtest_suite(rets_all[assets], assets, tradable, key)
        all_results[key] = {"discovery": disc_summary, "backtest": bt_summary}
        print(json.dumps(disc_summary, indent=2, default=str))
        print(f"strategy sharpe={bt_summary['strategy']['sharpe']:.3f}  "
              f"momentum sharpe={bt_summary['momentum_baseline']['sharpe']:.3f}  "
              f"buyhold sharpe={bt_summary['buy_hold_baseline']['sharpe']:.3f}  "
              f"null p50={bt_summary['null_sharpe_p50']:.3f}  "
              f"real percentile in null={bt_summary['real_sharpe_percentile_in_null']:.1f}%")

    # plots for the full universe (primary result)
    full_bt = all_results["full"]["backtest"]
    dates = rets_all.index[-len(full_bt["_equity_curves"]["strategy"]):]
    plot_equity_curves(full_bt["_equity_curves"], dates,
                        os.path.join(RESULTS_DIR, "equity_curves.png"),
                        "Lag-trading strategy vs. baselines (full FX+commodities+VIX universe)")
    plot_lag_histogram(all_results["full"]["discovery"]["lag_distribution_days"],
                        os.path.join(RESULTS_DIR, "lag_histogram.png"),
                        "Distribution of best significant lag, full sample (FDR q<0.05)")
    plot_null_distribution(full_bt["_null_sharpes"], full_bt["strategy"]["sharpe"],
                            os.path.join(RESULTS_DIR, "permutation_null.png"),
                            "Best-pick strategy vs. 200 random driver/lag permutations")

    ex_bt = all_results["ex_managed_fx"]["backtest"]
    plot_lag_histogram(all_results["ex_managed_fx"]["discovery"]["lag_distribution_days"],
                        os.path.join(RESULTS_DIR, "lag_histogram_ex_managed.png"),
                        "Lag distribution excluding managed-float currencies (MYR/CNY/HKD)")

    # strip non-serializable / heavy fields before dumping json
    def clean(d):
        d = dict(d)
        d.pop("_equity_curves", None)
        d.pop("_null_sharpes", None)
        return d

    json_out = {
        "full": {
            "discovery": all_results["full"]["discovery"],
            "backtest": clean(all_results["full"]["backtest"]),
        },
        "ex_managed_fx": {
            "discovery": all_results["ex_managed_fx"]["discovery"],
            "backtest": clean(all_results["ex_managed_fx"]["backtest"]),
        },
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    print(f"\nWrote results to {RESULTS_DIR}")
    return json_out


if __name__ == "__main__":
    main()
