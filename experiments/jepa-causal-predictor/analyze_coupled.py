"""Exploratory spurious-edge arm (SPEC addendum 2): runs trained in the `coupled` world.

Reports, per variant (mean ± sd over seeds):
  jac24    normalized ∂ẑ2/∂z4  (distractor -> object 2; the non-causal but predictive dependence)
  gate24   σ(A) for 4 -> 2 (hard gate only)
  o2_id    object-2 decoded error / copy, h=1, coupled test world
  o2_dec   same, decoupled test world
  ratio    o2_dec / o2_id   (per seed, then averaged)
Writes results/coupled.md.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "*_cpl_*", "eval.json"))):
    r = json.load(open(f))
    J = np.array(r["graph_jac"]["matrix"])
    m = dict(jac24=J[2, 4], jac21=J[2, 1], jac23=J[2, 3],
             o2_id=r["pred"]["id"]["h1"]["dec_nrmse_obj"][2], o2_dec=r["pred"]["decoupled"]["h1"]["dec_nrmse_obj"][2],
             all_id=r["pred"]["id"]["h1"]["dec_nrmse"], all_dec=r["pred"]["decoupled"]["h1"]["dec_nrmse"])
    m["ratio"] = m["o2_dec"] / m["o2_id"]
    if "graph_gate" in r:
        m["gate24"] = np.array(r["graph_gate"]["matrix"])[2, 4]
        m["gate_edges"] = r["graph_gate"]["n_edges_thr0.5"]
    rows.setdefault(r["variant"], []).append(m)

cols = ["jac24", "jac21", "jac23", "gate24", "gate_edges", "o2_id", "o2_dec", "ratio", "all_id", "all_dec"]
out = ["# Spurious-edge arm (coupled world), state regime, h = 1", "",
       "| variant | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for v in ["dense", "hard", "oracle"]:
    if v not in rows:
        continue
    cells = []
    for c in cols:
        x = np.array([m[c] for m in rows[v] if c in m], float)
        cells.append("–" if len(x) == 0 else f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}")
    out.append(f"| {v} | {len(rows[v])} | " + " | ".join(cells) + " |")
out += ["", "jac2j = normalized Jacobian ∂ẑ2/∂z_j (true parents of 2: objects 1 and 3; 4 is the spurious predictor).",
        "o2_* = object-2 decoded error / copy baseline; all_* = mean over objects 0–3. ratio = o2_dec / o2_id."]
open(os.path.join(ROOT, "results", "coupled.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
