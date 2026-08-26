"""Chart generation for the overnight-strategy backtest results.
Static matplotlib PNGs; palette follows the repo's dataviz color rules
(fixed categorical hue order, single y-axis, no rainbow)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CAT = {
    "overnight (close->open)": "#2a78d6",   # slot 1 blue
    "intraday (open->close)": "#eb6834",    # slot 2 orange
    "buy&hold (close->close)": "#1baf7a",   # slot 3 aqua
}
TICKER_COLORS = {}  # filled in dynamically from CAT-like order
_TICKER_ORDER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_equity_curves(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), facecolor=SURFACE)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        df = r["df"]
        from backtest import equity_curve
        for leg_name, col in [("overnight (close->open)", "overnight_ret"),
                               ("intraday (open->close)", "intraday_ret"),
                               ("buy&hold (close->close)", "closeclose_ret")]:
            eq = equity_curve(df[col])
            ax.plot(eq.index, eq.values, color=CAT[leg_name], linewidth=1.6, label=leg_name)
        ax.set_yscale("log")
        ax.set_title(r["ticker"], fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        _style_ax(ax)
    axes[-1].legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle("Growth of $1: overnight vs intraday vs buy & hold (log scale)",
                 color=INK_PRIMARY, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_sharpe(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), facecolor=SURFACE, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        yr = r["yearly"]
        colors = ["#1baf7a" if v >= 0 else "#e34948" for v in yr["sharpe"]]
        ax.bar(yr.index.astype(str), yr["sharpe"], color=colors, width=0.65, zorder=2)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(r["ticker"], fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=90)
        _style_ax(ax)
    axes[0].set_ylabel("Overnight-leg Sharpe")
    fig.suptitle("Overnight-leg Sharpe by calendar year (stability / decay check)",
                 color=INK_PRIMARY, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_cost_sensitivity(results, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    for i, r in enumerate(results):
        cs = r["cost_sensitivity"]
        ax.plot(cs.index, cs["sharpe"], marker="o", markersize=5, linewidth=1.8,
                color=_TICKER_ORDER[i % len(_TICKER_ORDER)], label=r["ticker"])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Round-trip transaction cost (bps/day)")
    ax.set_ylabel("Overnight-leg Sharpe (net of cost)")
    ax.set_title("Cost sensitivity: overnight-leg Sharpe vs. round-trip cost",
                 fontsize=12, fontweight="bold")
    _style_ax(ax)
    ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(ranking, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURFACE)
    order = ranking.sort_values("robustness_score")
    colors = [_TICKER_ORDER[i % len(_TICKER_ORDER)] for i in range(len(order))]
    ax.barh(order.index, order["robustness_score"], color=colors, height=0.55, zorder=2)
    ax.set_xlabel("Robustness score (0-1, higher = edge more likely to persist)")
    ax.set_title("Overnight-strategy robustness ranking", fontsize=12, fontweight="bold")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(results, ranking, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_equity_curves(results, os.path.join(out_dir, "equity_curves.png"))
    plot_yearly_sharpe(results, os.path.join(out_dir, "yearly_sharpe.png"))
    plot_cost_sensitivity(results, os.path.join(out_dir, "cost_sensitivity.png"))
    plot_ranking(ranking, os.path.join(out_dir, "ranking.png"))
