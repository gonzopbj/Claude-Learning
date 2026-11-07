"""Paper cost model for the Anthropic construction on real formulas, far beyond executable sizes.

For random 3-SAT at m/n = 4.26 and K = g^2 in {4, 9, 16}: build the §6 network and the Lemma 1
grouping (as in src/anthropic_alg.py), then count, without contracting anything:
  S0   = C(K,2)·L            bits on the clique edges (level 0 = brute force over these)
  S1   = mean over ι of the smaller merged total (eq. 35, each bit once)   (level 1 instance)
  T    = (t+1)^{|R|}         number of signed terms at level 1
  Gmax = largest number of distinct variables inside one clause group (cost of one table query)
log2 cost(level 0) ≈ S0 + Gmax ;  log2 cost(level 1) ≈ log2 T + S1 + Gmax  (paper's accounting,
polynomial factors dropped). ι is sampled (2000 draws) when (t+1)^{|R|} is large.
"""
import itertools
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.anthropic_alg import _group_clauses, c_col, c_row, n_blocks  # noqa: E402
from src.cnf import normalize, random_kcnf_ratio  # noqa: E402


def model(clauses, K, seed=0, samples=2000):
    g = math.isqrt(K)
    grp = _group_clauses(clauses, K, seed, iters=1500)
    byvar = {}
    for ci, c in enumerate(clauses):
        for l in c:
            byvar.setdefault(abs(l), set()).add(int(grp[ci]))
    pair = {}
    for gs in byvar.values():
        gs = sorted(gs)
        for a, b in zip(gs, gs[1:]):
            pair[(a, b)] = pair.get((a, b), 0) + 1
    L = max([1] + list(pair.values()))
    gvars = [len({abs(l) for ci, c in enumerate(clauses) if grp[ci] == k for l in c}) for k in range(K)]
    t = n_blocks(L)
    rects = [(a, a2, j, j2) for a, a2 in itertools.combinations(range(g), 2)
             for j, j2 in itertools.combinations(range(g), 2)]
    rng = np.random.default_rng(seed)
    n_terms = (t + 1) ** len(rects)
    iotas = (itertools.product(range(t + 1), repeat=len(rects)) if n_terms <= samples
             else (tuple(rng.integers(0, t + 1, len(rects))) for _ in range(samples)))
    S1 = []
    for iota in iotas:
        pr, pc = {}, {}
        for (a, a2, j, j2), i in zip(rects, iota):
            pr[(a, a2)] = pr.get((a, a2), g * L) + c_row(i, L)
            pc[(j, j2)] = pc.get((j, j2), g * L) + c_col(i, L)
        S1.append(min(sum(pr.values()), sum(pc.values())))
    S0 = K * (K - 1) // 2 * L
    return dict(L=L, t=t, S0=S0, S1=float(np.mean(S1)), log2_terms=len(rects) * math.log2(t + 1),
                Gmax=max(gvars), edge_bits=sum(pair.values()))


rows = []
for n in [10, 20, 30, 40, 60, 80, 100, 150, 200, 300]:
    for K in (4, 9, 16):
        for seed in range(3):
            cl, _ = normalize(random_kcnf_ratio(n, 4.26, 3, 7000 + 10 * n + seed))
            r = model(cl, K, seed)
            r.update(n=n, K=K, seed=seed,
                     log2_cost_l0=r["S0"] + r["Gmax"],
                     log2_cost_l1=r["log2_terms"] + r["S1"] + r["Gmax"])
            rows.append(r)
    agg = {K: np.mean([r["log2_cost_l1"] for r in rows if r["n"] == n and r["K"] == K]) for K in (4, 9, 16)}
    agg0 = {K: np.mean([r["log2_cost_l0"] for r in rows if r["n"] == n and r["K"] == K]) for K in (4, 9, 16)}
    print(n, {K: (round(agg0[K], 1), round(agg[K], 1)) for K in agg}, flush=True)
json.dump(rows, open(os.path.join(ROOT, "results", "anthropic_cost_model.json"), "w"), indent=1)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7.5, 5))
ns = sorted({r["n"] for r in rows})
for K, c in ((4, "C0"), (9, "C1"), (16, "C2")):
    for key, ls, lab in (("log2_cost_l0", ":", "level 0 (no rewrite)"), ("log2_cost_l1", "-", "one rewrite level")):
        ys = [np.mean([r[key] / r["n"] for r in rows if r["n"] == n and r["K"] == K]) for n in ns]
        ax.plot(ns, ys, ls, color=c, marker="o", ms=3, label=f"K={K}, {lab}")
ax.axhline(1.0, color="k", lw=1, label="brute force 2^n")
ax.axhline(math.log2(1.6423), color="gray", lw=1, ls="--", label="1.6423^n (best #3-SAT)")
ax.set_xscale("log")
ax.set_xlabel("n (random 3-SAT, m/n = 4.26)")
ax.set_ylabel("log2(cost) / n   (paper cost model, poly factors dropped)")
ax.set_title("Anthropic construction: cost exponent per variable")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "anthropic_cost_model.png"), dpi=150)
