"""Aggregate results/runs.jsonl -> results/summary.csv, results/exponents.json, figures/*.png."""
import json
import math
import os
import statistics
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {"brute": "brute force", "dpll": "DPLL #SAT (components+cache)", "tn": "tensor network (opt_einsum)",
         "openai": "OpenAI scheme (r=2,Q=2,R=1)", "openai_R2": "OpenAI scheme (r=2,Q=3,R=2)",
         "anthropic": "Anthropic, 1 level (2x2 grid)", "anthropic_l0": "Anthropic, level 0 only"}
ORDER = ["brute", "dpll", "tn", "openai", "openai_R2", "anthropic", "anthropic_l0"]
STY = {"brute": ("k", "o"), "dpll": ("C0", "s"), "tn": ("C2", "^"), "openai": ("C3", "D"),
       "openai_R2": ("C1", "d"), "anthropic": ("C4", "v"), "anthropic_l0": ("C5", "x")}
FAM_TITLE = {"rand2.0": "random 3-SAT, m/n = 2", "rand4.26": "random 3-SAT, m/n = 4.26 (threshold)",
             "rand6.0": "random 3-SAT, m/n = 6", "planted": "planted 3-SAT, m/n = 4.26",
             "php": "pigeonhole PHP(h+1, h) (unsat)", "gridis": "independent sets of 3 x w grid"}


def load():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "runs.jsonl"))]
    return rows


def nval(r):
    if r.get("n") is not None:
        return r["n"]
    fam, p = r["family"], r["param"]
    return p * (p + 1) if fam == "php" else p


def main():
    rows = load()
    # --- cross-check counts ---
    counts = defaultdict(dict)
    for r in rows:
        if r["status"] == "ok":
            counts[(r["family"], r["param"], r["seed"])][r["method"]] = r["count"]
    disagree = [(k, v) for k, v in counts.items() if len(set(v.values())) > 1]
    n_checked = sum(1 for v in counts.values() if len(v) > 1)
    # --- per (method, family, n) aggregate ---
    agg = defaultdict(list)
    for r in rows:
        agg[(r["method"], r["family"], r["param"])].append(r)
    summary = []
    for (meth, fam, p), rs in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        ok = [r for r in rs if r["status"] == "ok"]
        times = [r["time"] if r["status"] == "ok" else float("inf") for r in rs]
        summary.append(dict(method=meth, family=fam, param=p, n=nval(rs[0]), reps=len(rs), ok=len(ok),
                            timeouts=sum(r["status"] == "timeout" for r in rs),
                            oom=sum(r["status"] == "oom" for r in rs),
                            errors=sum(r["status"] == "error" for r in rs),
                            median_time=statistics.median(times),
                            median_rss=statistics.median([r["peak_rss_mb"] for r in ok]) if ok else None,
                            base_rss=statistics.median([r["base_rss_mb"] for r in ok]) if ok else None,
                            stats=ok[0]["stats"] if ok else {}))
    with open(os.path.join(ROOT, "results", "summary.csv"), "w") as fh:
        fh.write("family,method,n,reps,ok,timeouts,oom,errors,median_time_s,median_peak_rss_mb\n")
        for s in summary:
            fh.write(f"{s['family']},{s['method']},{s['n']},{s['reps']},{s['ok']},{s['timeouts']},{s['oom']},"
                     f"{s['errors']},{s['median_time']:.4g},{s['median_rss'] if s['median_rss'] else ''}\n")
    # --- exponent fits: log2(median time) = a + b n on complete points with median > 0.05 s ---
    fits = {}
    for fam in FAM_TITLE:
        for meth in ORDER:
            pts = [(s["n"], s["median_time"]) for s in summary
                   if s["family"] == fam and s["method"] == meth and s["ok"] == s["reps"] and s["median_time"] > 0.05]
            if len(pts) >= 3:
                x = np.array([p[0] for p in pts], float)
                y = np.log2([p[1] for p in pts])
                b, a = np.polyfit(x, y, 1)
                fits[f"{fam}|{meth}"] = dict(slope=float(b), base=float(2 ** b), n_range=[int(x.min()), int(x.max())],
                                             points=len(pts))
            maxn = [s["n"] for s in summary if s["family"] == fam and s["method"] == meth and s["ok"] == s["reps"]]
            if maxn:
                fits.setdefault(f"{fam}|{meth}", {})["max_n_all_solved"] = int(max(maxn))
    json.dump({"fits": fits, "count_disagreements": [str(d) for d in disagree], "instances_cross_checked": n_checked},
              open(os.path.join(ROOT, "results", "exponents.json"), "w"), indent=1)
    print("cross-checked instances:", n_checked, "disagreements:", len(disagree))
    for k, v in fits.items():
        if "slope" in v:
            print(f"{k:28s} base={v['base']:.3f}  n in {v['n_range']}  max_n={v.get('max_n_all_solved')}")

    def series(fam, meth, key="median_time"):
        ss = [s for s in summary if s["family"] == fam and s["method"] == meth]
        xs = [s["n"] for s in ss if s["ok"] == s["reps"]]
        ys = [s[key] for s in ss if s["ok"] == s["reps"]]
        return xs, ys

    # --- runtime grid ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, fam in zip(axes.flat, FAM_TITLE):
        for meth in ORDER:
            xs, ys = series(fam, meth)
            if xs:
                c, mk = STY[meth]
                ax.semilogy(xs, ys, marker=mk, color=c, ms=4, lw=1.2, label=LABEL[meth])
        ax.axhline(60, color="gray", ls=":", lw=1)
        ax.set_title(FAM_TITLE[fam], fontsize=10)
        ax.set_xlabel("n (variables)")
        ax.set_ylabel("median wall time (s)")
        ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Median runtime per instance (only points where every instance finished within 60 s)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "runtime_all_families.png"), dpi=130)
    # --- headline figure: threshold family ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fam = "rand4.26"
    for meth in ORDER:
        xs, ys = series(fam, meth)
        if xs:
            c, mk = STY[meth]
            f = fits.get(f"{fam}|{meth}", {})
            lab = LABEL[meth] + (f"  (~{f['base']:.2f}^n)" if "base" in f else "")
            ax.semilogy(xs, ys, marker=mk, color=c, ms=4, lw=1.2, label=lab)
    xx = np.arange(4, 41)
    ax.semilogy(xx, 2.0 ** xx * 2e-8, "k:", lw=.8, label="2^n reference")
    ax.axhline(60, color="gray", ls=":", lw=1)
    ax.set_ylim(1e-4, 200)
    ax.set_xlim(0, 100)
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("median wall time per instance (s)")
    ax.set_title("Random 3-SAT at m/n = 4.26: who scales how")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "runtime_threshold.png"), dpi=150)
    # --- memory ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for meth in ORDER:
        xs, ys = series(fam, meth, "median_rss")
        if xs:
            c, mk = STY[meth]
            ax.semilogy(xs, ys, marker=mk, color=c, ms=4, lw=1.2, label=LABEL[meth])
    base = [s["base_rss"] for s in summary if s["base_rss"]]
    if base:
        ax.axhline(statistics.median(base), color="gray", ls=":", lw=1, label="interpreter + imports")
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("peak RSS of the worker process (MB)")
    ax.set_title("Peak memory, random 3-SAT at m/n = 4.26")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "memory_threshold.png"), dpi=150)
    # --- internals: OpenAI tuples, Anthropic bits ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for fam_, c in (("rand2.0", "C0"), ("rand4.26", "C3"), ("rand6.0", "C2")):
        ok = [r for r in rows if r["method"] == "openai" and r["family"] == fam_ and r["status"] == "ok"]
        xs = [r["stats"]["n_occ"] for r in ok if "n_occ" in r["stats"]]
        ys = [r["stats"]["n_measurement_tuples"] for r in ok if "n_occ" in r["stats"]]
        ax.semilogy(xs, ys, "o", color=c, ms=3, alpha=.6, label=f"m/n = {fam_[4:]}")
    xx = np.arange(1, 21)
    ax.semilogy(xx, 2.0 ** xx, "k--", lw=1, label="2^(occurring variables)")
    ax.set_xlabel("occurring variables")
    ax.set_ylabel("measurement tuples enumerated (per restricted sum)")
    ax.set_title("OpenAI scheme: the measurement enumeration is exhaustive search")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax = axes[1]
    ok = [r for r in rows if r["method"] == "anthropic" and r["status"] == "ok" and "U_before" in r["stats"]]
    for fam_, c in (("rand2.0", "C0"), ("rand4.26", "C3"), ("rand6.0", "C2"), ("planted", "C1")):
        sub = [r for r in ok if r["family"] == fam_]
        if sub:
            ax.plot([r["stats"]["U_before"] / 2 for r in sub], [r["stats"]["U_after_mean"] / 2 for r in sub],
                    "o", color=c, ms=4, alpha=.6, label=f"{fam_}")
    lim = np.array([0, 80])
    ax.plot(lim, lim, "k:", lw=1, label="no saving")
    ax.plot(lim, lim * (2 / 3), "k--", lw=.8, label="2/3")
    ax.set_xlabel("clique edge bits before the level (K(K-1)/2 · L)")
    ax.set_ylabel("edge bits between merged rows/cols after (mean over ι)")
    ax.set_title("Anthropic level 1 on the 2x2 grid: bits cut")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "internals.png"), dpi=140)
    # --- exponents bar chart ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fams = [f for f in FAM_TITLE if f not in ("php",)]
    w = 0.12
    for j, meth in enumerate(ORDER):
        vals = [fits.get(f"{f}|{meth}", {}).get("slope", np.nan) for f in fams]
        ax.bar(np.arange(len(fams)) + (j - 3) * w, vals, w, color=STY[meth][0], label=LABEL[meth])
    ax.axhline(1, color="k", lw=1, ls=":")
    ax.set_xticks(range(len(fams)), [FAM_TITLE[f] for f in fams], fontsize=8)
    ax.set_ylabel("fitted d log2(time) / dn")
    ax.set_title("Empirical growth exponents (1.0 = doubling per variable, like brute force)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "exponents.png"), dpi=140)


if __name__ == "__main__":
    main()
