"""Aggregate runs/*/eval.json into results/summary.md, results/summary.json and figures/.

  python aggregate.py [--prefix state_ --lam_A ...]
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import env as E

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
ORDER = ["dense", "dense_ent", "soft", "hard", "mult", "oracle", "random"]


def load_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval.json"))):
        r = json.load(open(f))
        if r["seed"] >= 100:          # lambda-selection runs
            continue
        rows.append(r)
    return rows


def ms(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "–"
    if len(x) == 1:
        return f"{x[0]:.3f}"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def paired(a, b):
    """mean of (a-b) with 95% t-CI over paired seeds."""
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2:
        return "–"
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]"


def metrics(r):
    p = r["pred"]
    m = dict(
        id1=p["id"]["h1"]["dec_nrmse"], id5=p["id"]["h5"]["dec_nrmse"],
        phys1=p["physics"]["h1"]["dec_nrmse"], dist1=p["distractor"]["h1"]["dec_nrmse"],
        dist5=p["distractor"]["h5"]["dec_nrmse"], do1_1=p["do1"]["h1"]["dec_nrmse"], do1_5=p["do1"]["h5"]["dec_nrmse"],
        id1_raw=p["id"]["h1"]["nrmse_nd"],
        jac_auroc=r["graph_jac"]["auroc"], jac_shdk=r["graph_jac"]["shd_oraclek"],
        attn_auroc=r["graph_attn"]["auroc"],
        jac20=np.array(r["graph_jac"]["matrix"])[2, 0],      # grandparent shortcut 0 -> 2 (normalized)
        jac24=np.array(r["graph_jac"]["matrix"])[2, 4],      # distractor -> 2
        jac_nonedge=np.mean([np.array(r["graph_jac"]["matrix"])[i, j] for i in range(5) for j in range(5)
                             if i != j and E.TRUE_PARENTS[i, j] == 0]),
        min_r2=r["alignment"]["min_matched_r2"], mean_r2=float(np.mean(r["alignment"]["matched_r2"])),
    )
    m["dist_ratio"] = m["dist1"] / m["id1"]
    m["do1_ratio"] = m["do1_1"] / m["id1"]
    m["phys_ratio"] = m["phys1"] / m["id1"]
    if "graph_gate" in r:
        g = r["graph_gate"]
        m.update(gate_auroc=g["auroc"], gate_auprc=g["auprc"], gate_shd=g["shd_thr0.5"], gate_edges=g["n_edges_thr0.5"],
                 gate_action_ok=float(g["action_correct_thr0.5"]))
    if "plan" in r:
        m.update(plan_id=r["plan"]["id"]["success"], plan_dist=r["plan"]["distractor"]["success"],
                 plan_id15=r["plan"]["id"]["success15"], plan_dist_fd=r["plan"]["distractor"]["final_dist"],
                 plan_id_fd=r["plan"]["id"]["final_dist"])
    return m


def group(rows):
    G = defaultdict(dict)       # (regime, frozen) -> variant -> seed -> metrics
    for r in rows:
        key = (r["regime"] + ("_idm" if r.get("idm") else ""), r["frozen"])
        G[key].setdefault(r["variant"], {})[r["seed"]] = (metrics(r), r)
    return G


def table(G, key, cols, title):
    lines = [f"### {title}", "", "| variant | n | " + " | ".join(c for c, _ in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for v in ORDER:
        if v not in G[key]:
            continue
        runs = G[key][v]
        cells = [ms([runs[s][0].get(c, np.nan) for s in runs]) for c, _ in cols]
        lines.append(f"| {v} | {len(runs)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Columns: " + "; ".join(f"`{c}` = {d}" for c, d in cols))
    lines.append("")
    return lines


def paired_table(G, key, cols, base="dense"):
    if base not in G[key]:
        return []
    lines = [f"Paired difference vs `{base}` (same seed), mean [95% CI]:", "",
             "| variant | " + " | ".join(c for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for v in ORDER:
        if v == base or v not in G[key]:
            continue
        seeds = sorted(set(G[key][v]) & set(G[key][base]))
        cells = []
        for c in cols:
            a = [G[key][v][s][0].get(c, np.nan) for s in seeds]
            b = [G[key][base][s][0].get(c, np.nan) for s in seeds]
            cells.append(paired(a, b))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def fig_gates(G, key, tag):
    variants = [v for v in ["soft", "hard", "mult"] if v in G[key]]
    if not variants:
        return
    seeds = sorted(next(iter(G[key].values())).keys())
    n = len(seeds)
    fig, axes = plt.subplots(len(variants) + 1, n, figsize=(2.3 * n, 2.3 * (len(variants) + 1)), squeeze=False)
    truth = np.zeros((5, 6))
    truth[:, :5] = E.TRUE_PARENTS
    truth[:, 5] = E.ACTION_PARENT
    truth[np.arange(5), np.arange(5)] = 1
    labels = ["0", "1", "2", "3", "D", "a"]
    for j in range(n):
        ax = axes[0, j]
        if j == 0:
            ax.imshow(truth, vmin=0, vmax=1, cmap="Greys")
            ax.set_title("true graph", fontsize=9)
        ax.axis("off")
    for i, v in enumerate(variants):
        for j, s in enumerate(seeds):
            ax = axes[i + 1, j]
            r = G[key][v].get(s)
            if r is None:
                ax.axis("off")
                continue
            M = np.array(r[1]["graph_gate"]["matrix"])
            ax.imshow(M, vmin=0, vmax=1, cmap="Greys")
            ax.set_title(f"{v} s{s} AUROC {r[0]['gate_auroc']:.2f}", fontsize=8)
            ax.set_xticks(range(6), labels, fontsize=7)
            ax.set_yticks(range(5), labels[:5], fontsize=7)
    fig.suptitle(f"Learned gates σ(A) ({tag}); rows = child, cols = parent", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"gates_{tag}.png"), dpi=130)
    plt.close(fig)


def fig_bars(G, key, cols, tag, ylabel):
    vs = [v for v in ORDER if v in G[key]]
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(vs), 3.2))
    w = 0.8 / len(cols)
    for k, (c, lab) in enumerate(cols):
        means, errs, pts = [], [], []
        for v in vs:
            x = np.array([G[key][v][s][0].get(c, np.nan) for s in G[key][v]], float)
            means.append(np.nanmean(x))
            errs.append(np.nanstd(x, ddof=1) if len(x) > 1 else 0)
            pts.append(x)
        xs = np.arange(len(vs)) + (k - (len(cols) - 1) / 2) * w
        ax.bar(xs, means, w, yerr=errs, label=lab, capsize=2, alpha=0.85)
        for xi, p in zip(xs, pts):
            ax.scatter(np.full(len(p), xi), p, s=6, color="k", zorder=3)
    ax.set_xticks(range(len(vs)), vs, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{tag}.png"), dpi=130)
    plt.close(fig)


def fig_alignment(G, key, tag):
    vs = [v for v in ORDER if v in G[key]]
    seeds = sorted({s for v in vs for s in G[key][v]})
    fig, axes = plt.subplots(len(vs), len(seeds), figsize=(2.2 * len(seeds), 2.2 * len(vs)), squeeze=False)
    for i, v in enumerate(vs):
        for j, s in enumerate(seeds):
            ax = axes[i, j]
            r = G[key][v].get(s)
            if r is None:
                ax.axis("off")
                continue
            R = np.array(r[1]["alignment"]["R"])
            ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"{v} s{s} min {r[0]['min_r2']:.2f}", fontsize=8)
            ax.set_xlabel("object", fontsize=7)
            ax.set_ylabel("factor", fontsize=7)
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Factor→object position R² ({tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"alignment_{tag}.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load_all()
    G = group(rows)
    out = ["# Results summary (auto-generated by aggregate.py)", ""]
    pred_cols = [("id1", "in-distribution h=1 decoded error / copy"), ("id5", "same, h=5"),
                 ("dist_ratio", "distractor-shift / ID error (h=1)"), ("do1_ratio", "do(1) / ID error (h=1, objs 0,2,3)"),
                 ("phys_ratio", "physics-shift / ID error (h=1)"), ("do1_5", "do(1) h=5 decoded error / copy")]
    graph_cols = [("gate_auroc", "AUROC of σ(A)"), ("gate_shd", "SHD at σ>0.5"), ("gate_edges", "#edges σ>0.5 (true 3)"),
                  ("gate_action_ok", "action column correct"), ("jac_auroc", "AUROC of Jacobian dependence"),
                  ("jac_shdk", "Jacobian SHD top-3"), ("attn_auroc", "AUROC of mean attention"),
                  ("jac20", "normalized ∂ẑ2/∂z0 (grandparent)"), ("jac24", "normalized ∂ẑ2/∂z_distractor"),
                  ("jac_nonedge", "mean normalized Jacobian on non-edges")]
    plan_cols = [("plan_id", "success r<0.1, ID"), ("plan_id15", "success r<0.15, ID"),
                 ("plan_dist", "success r<0.1, distractor shift"), ("plan_id_fd", "final distance ID")]
    align_cols = [("min_r2", "min matched factor→object R²"), ("mean_r2", "mean matched R²")]
    for key in sorted(G):
        regime, frozen = key
        tag = f"{regime}{'_frozen' if frozen else ''}"
        out += [f"## Regime: {tag}", ""]
        out += table(G, key, align_cols, "Alignment")
        out += table(G, key, pred_cols, "Prediction error")
        out += paired_table(G, key, ["id1", "dist_ratio", "do1_ratio", "phys_ratio"])
        out += table(G, key, graph_cols, "Graph recovery")
        if any("plan_id" in G[key][v][s][0] for v in G[key] for s in G[key][v]):
            out += table(G, key, plan_cols, "Planning")
            out += paired_table(G, key, ["plan_id", "plan_dist"])
        fig_gates(G, key, tag)
        fig_bars(G, key, [("id1", "ID"), ("dist1", "distractor ×3"), ("do1_1", "do(1)"), ("phys1", "physics")],
                 f"pred_error_{tag}", "decoded error / copy baseline (h=1)")
        if any("plan_id" in G[key][v][s][0] for v in G[key] for s in G[key][v]):
            fig_bars(G, key, [("plan_id", "ID"), ("plan_dist", "distractor ×3")], f"planning_{tag}", "success rate")
        fig_bars(G, key, [("jac_auroc", "Jacobian"), ("attn_auroc", "attention"), ("gate_auroc", "gate")],
                 f"graph_auroc_{tag}", "graph-recovery AUROC")
        if regime.startswith("pixel"):
            fig_alignment(G, key, tag)
    open(os.path.join(ROOT, "results", "summary.md"), "w").write("\n".join(out))
    json.dump({f"{k[0]}{'_frozen' if k[1] else ''}": {v: {s: G[k][v][s][0] for s in G[k][v]} for v in G[k]} for k in G},
              open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
