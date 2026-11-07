"""Lemma 2 check: measured matrix ranks of the pieces A_i vs the paper's bit bounds.

Writes results/lemma2_ranks.json and figures/lemma2_ranks.png.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.anthropic_alg import c_col, c_row, n_blocks, pieces_dense  # noqa: E402


def rank(M):
    return int(np.linalg.matrix_rank(M.astype(np.float64)))


rows = []
for lam in range(1, 7):
    A = pieces_dense(lam)
    N = 1 << lam
    t = n_blocks(lam)
    for i in range(t + 1):
        rr = rank(A[i].reshape(N * N, N * N))
        rc = rank(A[i].transpose(0, 2, 1, 3).reshape(N * N, N * N))
        rows.append(dict(lam=lam, i=i, t=t, rank_row_bits=math.log2(rr), rank_col_bits=math.log2(rc),
                         c_row=c_row(i, lam), c_col=c_col(i, lam)))
        print(rows[-1], flush=True)
    del A
json.dump(rows, open(os.path.join(ROOT, "results", "lemma2_ranks.json"), "w"), indent=1)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(6.4, 4.2))
lams = sorted({r["lam"] for r in rows})
best_meas, best_bound = [], []
for lam in lams:
    rs = [r for r in rows if r["lam"] == lam]
    # per piece: cost of the cheaper direction; the paper sums row+col (11/12 argument)
    best_meas.append(max(r["rank_row_bits"] + r["rank_col_bits"] for r in rs) / (4 * lam))
    best_bound.append(max(r["c_row"] + r["c_col"] for r in rs) / (4 * lam))
ax.plot(lams, best_bound, "o-", label="paper bound  max_i (c_row+c_col) / 4λ")
ax.plot(lams, best_meas, "s-", label="measured  max_i (log2 rank_row + log2 rank_col) / 4λ")
ax.axhline(11 / 12, ls="--", c="gray", lw=1, label="ρ = 11/12 (asymptotic)")
ax.axhline(1.0, ls=":", c="k", lw=1, label="no saving (two equality tests)")
ax.set_xlabel("string length λ (bits)")
ax.set_ylabel("bits after rewrite / bits before")
ax.set_title("Lemma 2: the crossing rewrite, bound vs. actual rank")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "lemma2_ranks.png"), dpi=150)
