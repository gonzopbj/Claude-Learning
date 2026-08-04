"""
Orchestrates the intraday experiment: BTC (Coinbase + Bitstamp) + ETH,
1-hour and 1-minute bars, with confidence-gated position taking.

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
from data import ALL_ASSETS, TRADABLE_ASSETS, build_hour_panel, build_minute_panel, compute_returns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_PERMUTATIONS = 200
SEED = 42

# bar-frequency-specific configuration
CONFIGS = {
    "hour": dict(
        window=24 * 30,       # 30-day trailing window
        refresh=24 * 7,        # re-pick weekly
        max_lag=24,             # scan up to 1 day ahead
        annualization=24 * 365,
        cost_bps=0.0005,        # 5bps: crypto taker fees are higher than FX
        mom_lookback=24,
        target_vol_annual=0.20,
        vol_lookback=24 * 30,
        max_leverage=3.0,
        n_permutations=200,
    ),
    "minute": dict(
        window=60 * 24 * 7,    # 1-week trailing window
        refresh=60 * 24,        # re-pick daily
        max_lag=60,              # scan up to 1 hour ahead
        annualization=60 * 24 * 365,
        cost_bps=0.0005,
        mom_lookback=60,
        target_vol_annual=0.20,
        vol_lookback=60 * 24 * 7,
        max_leverage=3.0,
        n_permutations=50,  # T~1.4M rows: each permutation costs ~5s, keep runtime sane
    ),
}


def run_discovery(rets: pd.DataFrame, assets: list[str], max_lag: int) -> dict:
    scan = D.full_sample_scan(rets, assets, max_lag=max_lag)
    scan_fdr = D.add_fdr(scan)
    n_pairs = len(assets) * (len(assets) - 1)
    return D.summarize(scan_fdr, n_pairs, scan.shape[0])


def run_backtest_suite(rets: pd.DataFrame, assets: list[str], tradable_assets: list[str], cfg: dict,
                        n_permutations: int = N_PERMUTATIONS) -> dict:
    records, R, tradable_idx = B.compute_rebalance_windows(
        rets, assets, tradable_assets, window=cfg["window"], refresh=cfg["refresh"], max_lag=cfg["max_lag"])
    target_rets = R[:, tradable_idx]
    port_kwargs = dict(annualization=cfg["annualization"], cost_bps=cfg["cost_bps"],
                        target_vol_annual=cfg["target_vol_annual"], vol_lookback=cfg["vol_lookback"],
                        max_leverage=cfg["max_leverage"])

    best_assign = B.select_assignments(records, tradable_idx, len(assets), mode="best")
    raw_ungated = B.signals_from_rebalances(best_assign, R, tradable_idx)
    cost_sensitivity = {}
    for cost in [0.0, 0.0001, 0.0002, 0.0003, cfg["cost_bps"]]:
        p = B.raw_signal_to_portfolio(raw_ungated, target_rets, annualization=cfg["annualization"], cost_bps=cost,
                                       target_vol_annual=cfg["target_vol_annual"], vol_lookback=cfg["vol_lookback"],
                                       max_leverage=cfg["max_leverage"])
        cost_sensitivity[f"{cost * 10000:.0f}bps"] = round(
            B.performance_metrics(p["strategy_return"], cfg["annualization"])["sharpe"], 3)

    confidences = B.pick_confidence_distribution(best_assign)
    thresholds = {
        "ungated (trade every pick)": 0.0,
        "median-confidence gate": float(np.percentile(confidences, 50)),
        "p75-confidence gate": float(np.percentile(confidences, 75)),
        "p90-confidence gate": float(np.percentile(confidences, 90)),
    }

    gated_results = {}
    equity_curves = {}
    for label, thr in thresholds.items():
        gated = B.apply_confidence_gate(best_assign, thr)
        raw = B.signals_from_rebalances(gated, R, tradable_idx)
        port = B.raw_signal_to_portfolio(raw, target_rets, **port_kwargs)
        metrics = B.performance_metrics(port["strategy_return"], cfg["annualization"])
        metrics["threshold_abs_corr"] = thr
        metrics["mean_n_active"] = float(port["n_active"].mean())
        metrics["mean_turnover"] = float(port["turnover"].mean())
        gated_results[label] = metrics
        equity_curves[label] = (1 + port["strategy_return"]).cumprod()

    mom = B.momentum_baseline(rets, tradable_assets, lookback=cfg["mom_lookback"], **port_kwargs)
    metrics_mom = B.performance_metrics(mom["strategy_return"], cfg["annualization"])
    equity_curves["momentum baseline"] = (1 + mom["strategy_return"]).cumprod()

    bh = B.buy_hold_baseline(rets, tradable_assets, cost_bps=cfg["cost_bps"])
    metrics_bh = B.performance_metrics(bh["strategy_return"], cfg["annualization"])
    equity_curves["buy-hold baseline"] = (1 + bh["strategy_return"]).cumprod()

    rng = np.random.default_rng(SEED)
    null_sharpes = []
    t0 = time.time()
    for i in range(n_permutations):
        rand_assign = B.select_assignments(records, tradable_idx, len(assets), mode="random", rng=rng)
        raw_r = B.signals_from_rebalances(rand_assign, R, tradable_idx)
        port_r = B.raw_signal_to_portfolio(raw_r, target_rets, **port_kwargs)
        null_sharpes.append(B.performance_metrics(port_r["strategy_return"], cfg["annualization"])["sharpe"])
    perm_time = time.time() - t0
    null_sharpes = np.array(null_sharpes)
    real_sharpe = gated_results["ungated (trade every pick)"]["sharpe"]
    percentile = float((null_sharpes < real_sharpe).mean() * 100)

    return {
        "gated_results": gated_results,
        "cost_sensitivity_sharpe": cost_sensitivity,
        "momentum_baseline": metrics_mom,
        "buy_hold_baseline": metrics_bh,
        "n_permutations": n_permutations,
        "permutation_runtime_sec": round(perm_time, 1),
        "null_sharpe_mean": float(null_sharpes.mean()),
        "null_sharpe_std": float(null_sharpes.std()),
        "null_sharpe_p50": float(np.percentile(null_sharpes, 50)),
        "real_sharpe_percentile_in_null": percentile,
        "_equity_curves": equity_curves,
        "_null_sharpes": null_sharpes,
    }


def find_pair(all_sig_pairs, driver, target):
    for row in all_sig_pairs:
        if row["driver"] == driver and row["target"] == target:
            return row
    return None


def plot_confidence_gating(gated_results: dict, path: str, title: str):
    labels = list(gated_results.keys())
    sharpes = [gated_results[l]["sharpe"] for l in labels]
    n_active = [gated_results[l]["mean_n_active"] for l in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(range(len(labels)), sharpes, color="#4C72B0")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([l.replace(" gate", "").replace(" (trade every pick)", "") for l in labels],
                         rotation=20, ha="right")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Annualized Sharpe")
    ax1.set_title("Sharpe by confidence gate")
    ax2.bar(range(len(labels)), n_active, color="#DD8452")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([l.replace(" gate", "").replace(" (trade every pick)", "") for l in labels],
                         rotation=20, ha="right")
    ax2.set_ylabel("Mean # assets held (of 3)")
    ax2.set_title("Selectivity by confidence gate")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity_curves(equity_curves: dict, dates, path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in equity_curves.items():
        ax.plot(dates, curve.to_numpy(), label=name, linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_null_distribution(null_sharpes, real_sharpe, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_sharpes, bins=30, color="#888888", alpha=0.8, label="Random driver/lag pick (null)")
    ax.axvline(real_sharpe, color="crimson", linewidth=2, label=f"Best-pick, ungated (Sharpe={real_sharpe:.2f})")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Permutations")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def clean(d):
    d = dict(d)
    d.pop("_equity_curves", None)
    d.pop("_null_sharpes", None)
    return d


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    minute_panel = build_minute_panel()
    hour_panel = build_hour_panel(minute_panel)

    results = {}
    for freq, panel in [("hour", hour_panel), ("minute", minute_panel)]:
        print(f"\n=== frequency: {freq} ({panel.shape[0]} bars) ===")
        cfg = CONFIGS[freq]
        rets = compute_returns(panel)[ALL_ASSETS].dropna()

        disc = run_discovery(rets, ALL_ASSETS, cfg["max_lag"])
        print(f"pct pairs w/ sig lag: {disc['pct_pairs_with_any_significant_lag']}  "
              f"median best lag: {disc['median_best_lag_bars']} bars")
        for row in disc["all_significant_pairs"]:
            print(f"  {row['driver']:15s} -> {row['target']:15s} lag={row['lag']:3d} corr={row['corr']:+.4f} q={row['qvalue']:.2e}")

        bt = run_backtest_suite(rets, ALL_ASSETS, TRADABLE_ASSETS, cfg, n_permutations=cfg["n_permutations"])
        for label, m in bt["gated_results"].items():
            print(f"  [{label}] sharpe={m['sharpe']:.3f} n_active={m['mean_n_active']:.2f} "
                  f"turnover={m['mean_turnover']:.3f} threshold={m['threshold_abs_corr']:.4f}")
        print(f"  cost sensitivity (ungated): {bt['cost_sensitivity_sharpe']}")
        print(f"  momentum sharpe={bt['momentum_baseline']['sharpe']:.3f}  "
              f"buyhold sharpe={bt['buy_hold_baseline']['sharpe']:.3f}  "
              f"null p50={bt['null_sharpe_p50']:.3f}  "
              f"real(ungated) percentile in null={bt['real_sharpe_percentile_in_null']:.1f}%")

        results[freq] = {"discovery": disc, "backtest": bt}

        dates = rets.index[-len(list(bt["_equity_curves"].values())[0]):]
        plot_equity_curves(bt["_equity_curves"], dates,
                            os.path.join(RESULTS_DIR, f"equity_curves_{freq}.png"),
                            f"Confidence-gated lag strategy vs. baselines ({freq} bars, BTC x2 + ETH)")
        plot_confidence_gating(bt["gated_results"],
                                os.path.join(RESULTS_DIR, f"confidence_gating_{freq}.png"),
                                f"Effect of confidence gating ({freq} bars)")
        plot_null_distribution(bt["_null_sharpes"], bt["gated_results"]["ungated (trade every pick)"]["sharpe"],
                                os.path.join(RESULTS_DIR, f"permutation_null_{freq}.png"),
                                f"Best-pick vs. {cfg['n_permutations']} random permutations ({freq} bars)")

    json_out = {freq: {"discovery": r["discovery"], "backtest": clean(r["backtest"])} for freq, r in results.items()}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\nWrote results to {RESULTS_DIR}")
    return json_out


if __name__ == "__main__":
    main()
