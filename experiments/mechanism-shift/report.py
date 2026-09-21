"""Report builder: results dirs -> metrics.json, figures/, REPORT.md (SPEC.md "Analysis plan",
"Pre-registered predictions", "Deliverables"; INTERFACES.md section 7).

    python report.py results/main results/sweep_B10 results/sweep_B25 results/sweep_B100 \\
                     results/hidden results/knob_* --out results/report

Each results directory is classified from its config.json: `main` (B = 50, large shifts,
gamma_h = 0, visible), `sweep_B<B>`, `hidden`, or `knob_<shift_type>_h<gamma_h>`.  Raw files
are loaded one seed at a time (all agents of that seed together, because every primary is a
within-seed comparison), summarized, and dropped, so a full set of runs fits in memory.

Layout
    1. adapter `M`          every metrics.py name used here, in one place (rename = one line)
    2. discovery            classify results dirs, index raw files
    3. per-seed summaries   one dict per (label, agent, seed); cross-agent secondaries per seed
    4. aggregation          means and bootstrap CIs over seeds, cluster bootstrap for events
    5. verdicts             P1-P9: primary scalar, CI, one-sided p, Holm, HELD / NOT HELD /
                            NOT TESTABLE (with the reason)
    6. figures              matplotlib (Agg), PNG
    7. REPORT.md            one table per prediction, exploratory tables clearly labelled

Everything that is not a pre-registered primary is labelled exploratory (SPEC "Analysis
plan").  Decisions on ambiguous readings are marked "Decision:" in comments.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import traceback
import warnings
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import metrics as _metrics  # noqa: E402

# =========================================================================== 1. adapter

# Every metrics.py symbol used by this module.  If metrics renames something, change it here.
M = SimpleNamespace(
    load_raw=_metrics.load_raw,
    scalar=_metrics.scalar,
    dims=_metrics.dims,
    episode_kinds=_metrics.episode_kinds,
    qualifying_shifts=_metrics.qualifying_shifts,
    shift_log=_metrics.shift_log,
    episode_prediction_mask=_metrics.episode_prediction_mask,
    j_dependent_mask_run=_metrics.j_dependent_mask_run,
    pair_mask=_metrics.pair_mask,
    nmse_series=_metrics.nmse_series,
    nmse_summary=_metrics.nmse_summary,
    regret_per_shift=_metrics.regret_per_shift,
    recovery_per_shift=_metrics.recovery_per_shift,
    recovery_summary=_metrics.recovery_summary,
    pooled_recovery=_metrics.pooled_recovery,
    forgetting_per_shift=_metrics.forgetting_per_shift,
    coverage=_metrics.coverage,
    calibration_table=_metrics.calibration_table,
    selective_risk=_metrics.selective_risk,
    shd_series=_metrics.shd_series,
    credit_assignment=_metrics.credit_assignment,
    internal_objectives=_metrics.internal_objectives,
    bootstrap_ci=_metrics.bootstrap_ci,
    cluster_bootstrap=_metrics.cluster_bootstrap,
    holm=_metrics.holm,
    PRIMARIES=_metrics.PRIMARIES,
    evaluate_primary=_metrics.evaluate_primary,
    COVERAGE_GRID=_metrics.COVERAGE_GRID,
    N_BOOT=_metrics.N_BOOT,
)

ALPHA = 0.05
# Which results-dir kind feeds each primary (PrimarySpec.run in metrics.py names the same).
PRIMARY_KIND = {"P1": "main", "P2": "main", "P3": "main", "P4": "main", "P7": "main", "P8": "main",
                "P5": "sweep_B10", "P6": "hidden", "P9": "knob_noise-only_h0"}
PRIMARY_CLAIM = {
    "P1": "see is not do: obs-window's nMSE sits at the population floor F_obs (inside "
          "+/-(0.1 F_obs + 0.01)) in >= 80 % of seeds",
    "P2": "recovery: regret(mech-full) - 2 regret(mech-oracle-detect) <= 0 (both vs oracle-structure)",
    "P3": "calibration: mech-full 90 % coverage in stable, SHD = 0 episodes within +/-5 points of nominal",
    "P4": "memory: forgetting bump(mech-reset-all) - bump(mech-full) >= 0.04",
    "P5": "structure: nMSE(int-pairwise) - nMSE(mech-full) > 0 at B = 10, episodes 20-60",
    "P6": "hidden confounder: mech-full 90 % coverage on do(A) -> B (episodes 40-60) has CI upper bound < 0.5",
    "P7": "credit assignment: mech-full delay-0 attribution accuracy has CI lower bound >= 0.8",
    "P8": "objectives: self_score(mech-full) - self_score(mech-overconfident) > 0 (episodes 10-60)",
    "P9": "noise-only shifts: mech-full's regret vs oracle-structure >= 0.02 per shift",
}

# Fixed per-agent colors (never cycled) from the reference palette; line style separates
# variants of one family.  Text never wears series color.
_C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
          green="#008300", violet="#4a3aa7", red="#e34948", gray="#52514e", ink="#0b0b0b")
AGENT_STYLE = {
    "marginal-mean": (_C["gray"], ":", 1.6),
    "obs-window": (_C["orange"], "-", 1.6),
    "obs-cumulative": (_C["orange"], "--", 1.6),
    "int-pairwise": (_C["yellow"], "-", 1.6),
    "mech-full": (_C["blue"], "-", 2.6),
    "mech-random": (_C["blue"], "--", 1.6),
    "mech-full-nofloor": (_C["blue"], ":", 1.6),
    "mech-reset-all": (_C["red"], "-", 1.6),
    "mech-no-detect": (_C["magenta"], "-", 1.6),
    "mech-oracle-detect": (_C["aqua"], "-", 1.6),
    "mech-overconfident": (_C["violet"], "-", 1.6),
    "oracle-structure": (_C["green"], "-", 1.6),
    "oracle": (_C["ink"], "--", 1.2),
    "mech-full-diag": (_C["aqua"], "--", 1.6),
}
AGENT_ORDER = list(AGENT_STYLE)


def _style(agent):
    return AGENT_STYLE.get(agent, (_C["gray"], "-", 1.4))


def _sorted_agents(names):
    return sorted(names, key=lambda a: (AGENT_ORDER.index(a) if a in AGENT_ORDER else 99, a))


# =========================================================================== 2. discovery

class RunDir:
    """One results directory: its config, label and the raw files indexed by (agent, seed)."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, "config.json")) as f:
            self.config = json.load(f)
        self.files: dict[tuple[str, int], str] = {}
        raw_dir = os.path.join(self.path, "raw")
        pat = re.compile(r"^(?P<agent>.+)__seed(?P<seed>\d+)\.npz$")
        for name in sorted(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else []:
            m = pat.match(name)
            if m:
                self.files[(m["agent"], int(m["seed"]))] = os.path.join(raw_dir, name)
        self.agents = _sorted_agents({a for a, _ in self.files})
        self.seeds = sorted({s for _, s in self.files})
        self.kind, self.label = classify(self.config)

    def constants_W(self, agent="obs-window", default=3):
        consts = self.config.get("constants") or {}
        return int((consts.get("agents", {}).get(agent, {}) or {}).get("W", default))


def classify(config: dict) -> tuple[str, str]:
    """(kind, label) of a results dir from its config.json."""
    B = int(config.get("budget", 50))
    st, h = config.get("shift_type", "large"), int(config.get("hetero", 0))
    if not config.get("shifts_enabled", True):
        return "noshift", f"noshift_B{B}"
    if config.get("hidden"):
        return "hidden", "hidden"
    if st != "large" or h != 0:
        return "knob", f"knob_{st}_h{h}"
    if B != 50:
        return "sweep", f"sweep_B{B}"
    return "main", "main"


def discover(paths: list[str]) -> list[RunDir]:
    dirs = [RunDir(p) for p in paths]
    seen: dict[str, int] = {}
    for rd in dirs:                       # two dirs of the same kind: keep both, suffix labels
        n = seen.get(rd.label, 0)
        seen[rd.label] = n + 1
        if n:
            rd.label = f"{rd.label}#{n + 1}"
    return dirs


# =========================================================================== 3. per-seed summaries

def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size and np.isfinite(x).any() else math.nan


def _ep_mask(T, lo, hi):
    t = np.arange(1, T + 1)
    return (t >= lo) & (t <= hi)


def summarize_agent(raw: dict, ref_raw: dict | None) -> dict:
    """Everything the report needs from one (agent, seed) raw file, as small arrays and scalars.
    `ref_raw` is oracle-structure's raw for the same seed (regret reference), or None."""
    T, Q, d = M.dims(raw)
    kinds = M.episode_kinds(raw)
    series = M.nmse_series(raw)
    S: dict = {"T": T, "wall_time_s": float(M.scalar(raw, "wall_time_s")) if "wall_time_s" in raw else math.nan}
    S["nmse_series"] = series
    S["abstain_series"] = np.asarray(raw["abstain"], dtype=np.float64).mean(axis=(1, 2))
    S["cov90_series"] = np.asarray(raw["cov90"], dtype=np.float64).mean(axis=(1, 2))
    S["width90_series"] = np.asarray(raw["width90"], dtype=np.float64).mean(axis=(1, 2))
    S.update(M.nmse_summary(raw))                       # nmse, descendant split, abstention
    for k in ("stable", "shift", "post_shift"):
        S[f"nmse_{k}"] = _nanmean(series[kinds[k]])
    S["nmse_20_60"] = _nanmean(series[_ep_mask(T, 20, 60)])
    S["nmse_last"] = float(series[-1])
    S["nmse_ep5"] = float(series[4]) if T >= 5 else math.nan
    S["calibration"] = M.calibration_table(raw)
    sel = M.selective_risk(raw)
    S["selective"] = {k: sel.get(k, math.nan) for k in
                      ("aurc", "aurc_oracle", "aurc_random", "risk_at", "mistake_recall",
                       "mistake_precision", "mistake_rate")}
    S["risk_curve"] = np.asarray(sel.get("risk_curve", np.full(len(M.COVERAGE_GRID), np.nan)))
    if bool(np.any(raw["has_learned_adj"])):
        shd = M.shd_series(raw)
        S["shd_series"] = shd
        S["shd_10_60"] = _nanmean(shd[_ep_mask(T, 10, 60)])
        S["shd_zero_at_10"] = float(shd[9] == 0) if T >= 10 else math.nan
    has_det = bool(M.scalar(raw, "has_detector")) if "has_detector" in raw else False
    if has_det or np.asarray(raw["reset_events"]).size:
        ca = M.credit_assignment(raw)
        S["credit"] = {k: v for k, v in ca.items() if k != "per_shift"}
    obj = M.internal_objectives(raw)
    ss, dec = np.asarray(obj["self_score"]), np.asarray(obj["var_obj_decrease"])
    S["self_score_10_60"] = _nanmean(ss[_ep_mask(T, 10, 60)])
    S["var_obj_decrease_1_10"] = _nanmean(dec[_ep_mask(T, 1, 10)])
    S["self_score_series"], S["var_obj_decrease_series"] = ss, dec
    shifts, excluded = M.qualifying_shifts(raw)
    S["n_shifts_qualifying"], S["n_shifts_excluded"] = len(shifts), excluded
    S["shift_ts"] = [s.t for s in shifts]
    if ref_raw is not None:
        S["regret"] = [r["regret"] for r in M.regret_per_shift(raw, ref_raw, shifts)]
        S["regret_mean"] = _nanmean(S["regret"]) if S["regret"] else math.nan
    S["recovery"] = [{"time": r["time"], "recovered": r["recovered"]}
                     for r in M.recovery_per_shift(raw, shifts)]
    vis, _ = M.qualifying_shifts(raw, visible_only=True)
    S["forgetting"] = [b["bump"] for b in M.forgetting_per_shift(raw, vis)]
    S["forgetting_mean"] = _nanmean(S["forgetting"]) if S["forgetting"] else math.nan
    return S


def undetected_shift_stats(raw: dict, n_after: int = 0) -> dict:
    """For qualifying visible shifts where the detector did NOT reset the shifted mechanism at
    step 4 of the shift episode: 90 % coverage and abstention on j-dependent queries in the
    shift episode (and the `n_after` following), against the stable-episode abstention rate
    (P3(b), P9 'small').  Decision: "did not fire on the shifted mechanism at step 4" is read
    from reset_events as "no (t, j) row", whatever happened to other mechanisms."""
    T = M.dims(raw)[0]
    shifts, _ = M.qualifying_shifts(raw, visible_only=True)
    resets = {(int(t), int(j)) for t, j in np.asarray(raw["reset_events"]).reshape(-1, 2)}
    stable = M.episode_kinds(raw)["stable"]
    abst = np.asarray(raw["abstain"], dtype=np.float64)
    cov = np.asarray(raw["cov90"], dtype=np.float64)
    stable_abst = _nanmean(abst[stable]) if stable.any() else math.nan
    per = []
    for s in shifts:
        if (s.t, s.j) in resets:
            continue
        dep = M.j_dependent_mask_run(raw, s.j)
        covs, absts = [], []
        for u in range(s.t, min(s.t + n_after, T) + 1):
            m = dep[u - 1]
            if m.any():
                covs.append(float(cov[u - 1][m].mean()))
                absts.append(float(abst[u - 1][m].mean()))
        if covs:
            per.append({"t": s.t, "j": s.j, "cov90": covs, "abstain": absts})
    first_cov = [p["cov90"][0] for p in per]
    two_max = [max(p["cov90"][:2]) for p in per if len(p["cov90"]) >= 2]
    return {"n_shifts": len(shifts), "n_undetected": len(per),
            "cov90_first": _nanmean(first_cov) if first_cov else math.nan,
            "cov90_max_first_two": _nanmean(two_max) if two_max else math.nan,
            "abstain_excess": (_nanmean([p["abstain"][0] for p in per]) - stable_abst) if per else math.nan,
            "abstain_stable": stable_abst}


def secondaries_per_seed(kind: str, runs: dict, rd: RunDir) -> dict:
    """Cross-agent, per-seed numbers behind the secondary claims (all exploratory)."""
    out: dict = {}
    mf = runs.get("mech-full")
    if kind == "main":                   # P1 / P3(b) secondaries
        ow = runs.get("obs-window")
        if mf is not None and ow is not None and M.dims(mf)[0] >= 5:
            out["p1_mech_below_obs_ep5"] = float(M.nmse_series(mf)[4] < M.nmse_series(ow)[4])
        if mf is not None:
            out["p3b"] = undetected_shift_stats(mf, n_after=0)
    if kind in ("sweep", "main"):        # P5's paired differences, at every budget available
        ip, mr, nf = runs.get("int-pairwise"), runs.get("mech-random"), runs.get("mech-full-nofloor")
        T = M.dims(mf)[0] if mf is not None else None

        def m2060(r):
            return _nanmean(M.nmse_series(r)[_ep_mask(T, 20, 60)])

        def shd1060(r):
            return _nanmean(M.shd_series(r)[_ep_mask(T, 10, 60)])
        if mf is not None and ip is not None:
            out["p5a_int_minus_full"] = m2060(ip) - m2060(mf)
            out["p3c_ratio_ep_last"] = float(M.nmse_series(ip)[-1] / max(M.nmse_series(mf)[-1], 1e-12))
        if mf is not None and mr is not None:
            out["p5b_full_minus_random"] = m2060(mf) - m2060(mr)
            out["p5c_shd_full_minus_random"] = shd1060(mf) - shd1060(mr)
        if nf is not None and mr is not None:
            out["p5c_shd_nofloor_minus_random"] = shd1060(nf) - shd1060(mr)
    elif kind == "hidden":
        if mf is not None:
            A, B = int(M.scalar(mf, "hidden_A")), int(M.scalar(mf, "hidden_B"))
            T, Q, d = M.dims(mf)
            c = float(np.asarray(mf["conf_c"])[0])
            out["conf_c"], out["conf_F"] = c, float(np.asarray(mf["conf_F"])[0])
            out["p6i_shd_zero_at_10"] = float(M.shd_series(mf)[9] == 0) if T >= 10 else math.nan
            late = M.episode_prediction_mask(mf, _ep_mask(T, 40, 60))
            pair = M.pair_mask(mf, A, B)
            # P6(ii): the signed unnormalized error err * sd_ref(X_B) on do(A) -> B queries is
            # ~ (dilution * c) * v.  Decision: "at v = 2" is read as the slope error / v over
            # queries with |v| >= 1 (queries draw v ~ U(-2, 2), so exactly v = 2 never occurs),
            # divided by c: the claim is ratio in [0.3, 1.0].
            v = np.broadcast_to(np.asarray(mf["query_v"], dtype=np.float64)[:, :, None], pair.shape)
            sd_B = np.broadcast_to(np.asarray(mf["sd_ref"], dtype=np.float64)[:, B][:, None, None], pair.shape)
            sel = pair & late & (np.abs(v) >= 1.0)
            if sel.any() and abs(c) > 1e-9:
                err_u = np.asarray(mf["err"], dtype=np.float64)[sel] * sd_B[sel]
                out["p6ii_ratio"] = float(np.mean(err_u / v[sel]) / c)
            out["p6iii_cov90_pair"], out["p6_pair_cov90_series"] = {}, {}
            for a, r in runs.items():
                pm = M.pair_mask(r, A, B)
                out["p6iii_cov90_pair"][a] = M.coverage(r, pm & late)["cov90"]
                cov = np.asarray(r["cov90"], dtype=np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["p6_pair_cov90_series"][a] = np.where(pm.sum(axis=(1, 2)) > 0,
                                                              (cov * pm).sum(axis=(1, 2)) / np.maximum(pm.sum(axis=(1, 2)), 1),
                                                              np.nan)
            abst = np.asarray(mf["abstain"], dtype=np.float64)
            out["p6iii_abstain_excess_pair"] = float(abst[pair & late].mean() - abst.mean()) if (pair & late).any() else math.nan
            h = [s for s in M.shift_log(mf) if s.j >= d]
            if h:
                resets = {(int(t), int(j)) for t, j in np.asarray(mf["reset_events"]).reshape(-1, 2)}
                out["p6iv_both_fire"] = float((h[0].t, A) in resets and (h[0].t, B) in resets)
                out["p6iv_any_fire"] = float(any((h[0].t, j) in resets for j in range(d)))
    elif kind == "knob":
        if mf is not None and rd.config.get("shift_type") == "small":
            out["p9_small"] = undetected_shift_stats(mf, n_after=1)
    return out


# =========================================================================== 4. aggregation

SCALAR_KEYS = ("nmse", "nmse_descendant", "nmse_non_descendant", "nmse_non_abstained",
               "abstain_rate", "nmse_stable", "nmse_shift", "nmse_post_shift", "nmse_20_60",
               "nmse_last", "nmse_ep5", "shd_10_60", "shd_zero_at_10", "self_score_10_60",
               "var_obj_decrease_1_10", "regret_mean", "forgetting_mean", "wall_time_s")
SERIES_KEYS = ("nmse_series", "abstain_series", "cov90_series", "width90_series", "shd_series",
               "risk_curve", "self_score_series", "var_obj_decrease_series")
CREDIT_RATE_KEYS = ("accuracy_delay0", "recall_window", "recall_any", "misattribution_rate",
                    "attribution_precision", "false_reset_rate")


def _ci(values, n_boot):
    return M.bootstrap_ci([v for v in values if v is not None], n_boot=n_boot)


def _mean_series(list_of_arrays):
    arrs = [np.asarray(a, dtype=np.float64) for a in list_of_arrays if a is not None]
    if not arrs:
        return None
    L = max(len(a) for a in arrs)
    mat = np.full((len(arrs), L), np.nan)
    for k, a in enumerate(arrs):
        mat[k, :len(a)] = a
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns are legitimate
        return np.nanmean(mat, axis=0)


def aggregate_agent(per_seed: list[dict], n_boot: int) -> dict:
    """Means with percentile-bootstrap CIs over seeds; pooled event statistics with a
    seed-level cluster bootstrap (SPEC "Analysis plan")."""
    A: dict = {"n_seeds": len(per_seed)}
    for k in SCALAR_KEYS:
        vals = [s.get(k, math.nan) for s in per_seed]
        if any(v is not None and math.isfinite(v) for v in vals):
            A[k] = _ci(vals, n_boot)
    A["series"] = {k: _mean_series([s.get(k) for s in per_seed]) for k in SERIES_KEYS}
    A["series"] = {k: v for k, v in A["series"].items() if v is not None}
    # calibration strata: mean over seeds of each entry, CI for cov90
    strata = sorted({k for s in per_seed for k in s["calibration"]})
    A["calibration"] = {}
    for st in strata:
        rows = [s["calibration"][st] for s in per_seed if st in s["calibration"]]
        entry = {k: _nanmean([r.get(k, math.nan) for r in rows])
                 for k in ("cov50", "cov90", "cov99", "width50", "width90", "width99", "abstain_rate")}
        entry["cov90_ci"] = _ci([r.get("cov90", math.nan) for r in rows], n_boot)
        entry["n"] = int(sum(r.get("n", 0) for r in rows))
        A["calibration"][st] = entry
    sel = [s["selective"] for s in per_seed]
    A["selective"] = {k: _ci([x.get(k, math.nan) for x in sel], n_boot)
                      for k in ("aurc", "aurc_oracle", "aurc_random", "mistake_recall",
                                "mistake_precision", "mistake_rate")}
    A["selective"]["risk_at"] = {c: _nanmean([x["risk_at"].get(c, math.nan) for x in sel if isinstance(x.get("risk_at"), dict)])
                                 for c in (0.5, 0.8, 0.9, 1.0)}
    credits = [s["credit"] for s in per_seed if "credit" in s]
    if credits:
        C = {"confusion": {k: int(sum(c["confusion"].get(k, 0) for c in credits))
                           for k in ("correct", "wrong-mechanism", "miss", "false-alarm")},
             "n_shifts": int(sum(c["n_shifts"] for c in credits)),
             "n_shifts_excluded": int(sum(c["n_shifts_excluded"] for c in credits)),
             "n_resets": int(sum(c["n_resets"] for c in credits))}
        for k in CREDIT_RATE_KEYS:
            C[k] = _ci([c.get(k, math.nan) for c in credits], n_boot)
        delays = [dl for c in credits for dl, det in zip(c["delays"], c["detected"]) if det]
        C["delay_detected_mean"] = _nanmean(delays) if delays else math.nan
        A["credit"] = C
    recs = [s["recovery"] for s in per_seed if s.get("recovery")]
    if recs:
        pooled = M.pooled_recovery(recs)
        R = {"n_events": pooled["n"], "n_censored": pooled["n_censored"],
             "censored_fraction": pooled["n_censored"] / pooled["n"] if pooled["n"] else math.nan}
        R["median"] = M.cluster_bootstrap(recs, lambda cl: M.pooled_recovery(cl)["median"], n_boot=n_boot)
        for k in (1, 2, 3):
            R[f"within_{k}"] = M.cluster_bootstrap(recs, lambda cl, k=k: M.pooled_recovery(cl)[f"within_{k}"], n_boot=n_boot)
        A["recovery"] = R
    A["n_shifts_qualifying"] = int(sum(s.get("n_shifts_qualifying", 0) for s in per_seed))
    A["n_shifts_excluded"] = int(sum(s.get("n_shifts_excluded", 0) for s in per_seed))
    return A


def aggregate_secondaries(kind: str, per_seed: list[dict], n_boot: int) -> dict:
    """Bootstrap CIs (over seeds) of the per-seed secondary numbers; nested dicts (per agent,
    or the undetected-shift blocks) are averaged field by field."""
    out: dict = {}
    keys = sorted({k for s in per_seed for k in s})
    for k in keys:
        vals = [s[k] for s in per_seed if k in s]
        if not vals:
            continue
        if isinstance(vals[0], dict) and k in ("p3b", "p9_small"):
            out[k] = {f: _ci([v.get(f, math.nan) for v in vals], n_boot)
                      for f in ("cov90_first", "cov90_max_first_two", "abstain_excess", "abstain_stable")}
            out[k]["n_shifts"] = int(sum(v["n_shifts"] for v in vals))
            out[k]["n_undetected"] = int(sum(v["n_undetected"] for v in vals))
        elif isinstance(vals[0], dict) and k == "p6iii_cov90_pair":
            out[k] = {a: _ci([v.get(a, math.nan) for v in vals], n_boot)
                      for a in sorted({a for v in vals for a in v})}
        elif isinstance(vals[0], dict) and k == "p6_pair_cov90_series":
            out[k] = {a: _mean_series([v.get(a) for v in vals]) for a in sorted({a for v in vals for a in v})}
        elif isinstance(vals[0], (int, float)):
            out[k] = _ci(vals, n_boot)
            if k in ("p6ii_ratio",):
                inside = [float(0.3 <= v <= 1.0) for v in vals if math.isfinite(v)]
                out[k]["fraction_in_band"] = _nanmean(inside) if inside else math.nan
    return out


# =========================================================================== 5. build

class Report:
    """Holds everything computed from the results dirs; `build()` fills it."""

    def __init__(self, dirs: list[RunDir], n_boot: int = M.N_BOOT, log=print):
        self.dirs = dirs
        self.n_boot = n_boot
        self.log = log
        self.by_label: dict[str, RunDir] = {rd.label: rd for rd in dirs}
        self.per_seed: dict[str, dict[str, list[dict]]] = {}   # label -> agent -> [S per seed]
        self.sec_per_seed: dict[str, list[dict]] = {}          # label -> [secondaries per seed]
        self.shift_ts: dict[str, dict[int, list[int]]] = {}    # label -> seed -> shift episodes
        self.primary_values: dict[str, dict] = {}              # P -> {"values", "reasons", "label"}
        self.agg: dict[str, dict] = {}
        self.sec: dict[str, dict] = {}
        self.primaries: dict[str, dict] = {}
        self.problems: list[str] = []

    # ---- loading and per-seed work
    def build(self):
        for rd in self.dirs:
            self.log(f"report: {rd.label} <- {rd.path} ({len(rd.agents)} agents, {len(rd.seeds)} seeds)")
            self._process_dir(rd)
        self.log("report: aggregating")
        for label, agents in self.per_seed.items():
            self.agg[label] = {a: aggregate_agent(lst, self.n_boot) for a, lst in agents.items()}
            self.sec[label] = aggregate_secondaries(self.by_label[label].kind, self.sec_per_seed.get(label, []), self.n_boot)
        self._evaluate_primaries()
        return self

    def _label_for_primary(self, name: str) -> str | None:
        want = PRIMARY_KIND[name]
        if want in self.by_label:
            return want
        # P9 accepts the noise-only knob at gamma_h = 1 if that is the only one available.
        if name == "P9":
            for lab, rd in self.by_label.items():
                if rd.kind == "knob" and rd.config.get("shift_type") == "noise-only":
                    return lab
        return None

    def _process_dir(self, rd: RunDir):
        per_seed: dict[str, list[dict]] = {a: [] for a in rd.agents}
        secs: list[dict] = []
        self.shift_ts[rd.label] = {}
        prim_names = [p for p in PRIMARY_KIND if self._label_for_primary(p) == rd.label]
        prim_vals = {p: {"values": [], "reasons": {}, "label": rd.label, "seeds": []} for p in prim_names}
        for seed in rd.seeds:
            runs: dict[str, dict] = {}
            for a in rd.agents:
                path = rd.files.get((a, seed))
                if path is None:
                    self.problems.append(f"{rd.label}: missing raw file for {a} seed {seed}")
                    continue
                runs[a] = M.load_raw(path)
            ref = runs.get("oracle-structure")
            for a, raw in runs.items():
                try:
                    per_seed[a].append(summarize_agent(raw, ref))
                except Exception:               # noqa: BLE001 - recorded, report continues
                    self.problems.append(f"{rd.label}: summarize {a} seed {seed} failed:\n{traceback.format_exc()}")
            any_raw = next(iter(runs.values()), None)
            if any_raw is not None:
                self.shift_ts[rd.label][seed] = [s.t for s in M.qualifying_shifts(any_raw)[0]]
            try:
                secs.append(secondaries_per_seed(rd.kind, runs, rd))
            except Exception:                   # noqa: BLE001
                self.problems.append(f"{rd.label}: secondaries seed {seed} failed:\n{traceback.format_exc()}")
            for p in prim_names:
                kwargs = {"W": rd.constants_W()} if p == "P1" else {}
                try:
                    v, why = M.PRIMARIES[p].fn(runs, **kwargs)
                except Exception:               # noqa: BLE001
                    v, why = math.nan, f"exception: {traceback.format_exc().splitlines()[-1]}"
                prim_vals[p]["values"].append(v)
                prim_vals[p]["seeds"].append(seed)
                if why:
                    prim_vals[p]["reasons"][seed] = why
        self.per_seed[rd.label] = {a: lst for a, lst in per_seed.items() if lst}
        self.sec_per_seed[rd.label] = secs
        self.primary_values.update(prim_vals)

    # ---- verdicts
    def _evaluate_primaries(self):
        """One-sided bootstrap p per primary, Holm across the testable ones, verdict text."""
        results = {}
        for name in PRIMARY_KIND:
            spec = M.PRIMARIES[name]
            pv = self.primary_values.get(name)
            entry = {"name": name, "claim": PRIMARY_CLAIM[name], "kind": spec.kind,
                     "direction": spec.direction, "threshold": spec.threshold,
                     "description": spec.description, "run": None, "values": [], "reasons": {}}
            if pv is None:
                entry["not_testable_reason"] = f"no results directory of kind '{PRIMARY_KIND[name]}' was given"
                entry.update({"p": math.nan, "ci": M.bootstrap_ci([]), "n_seeds": 0})
            else:
                entry.update({"run": pv["label"], "values": pv["values"], "seeds": pv["seeds"],
                              "reasons": {int(k): v for k, v in pv["reasons"].items()}})
                entry.update(M.evaluate_primary(spec, pv["values"], n_boot=self.n_boot))
                if entry["n_seeds"] == 0:
                    reasons = sorted(set(pv["reasons"].values()))
                    entry["not_testable_reason"] = "; ".join(reasons) or "no finite per-seed value"
            results[name] = entry
        corrected = M.holm({n: r["p"] for n, r in results.items()}, ALPHA)
        for n, r in results.items():
            r.update(corrected[n])
            if not math.isfinite(r["p"]):
                r["verdict"] = "NOT TESTABLE"
            else:
                r["verdict"] = "HELD" if r["reject"] else "NOT HELD"
            r["held"] = r["verdict"] == "HELD"
        self.primaries = results
        self.holm_family = [n for n, r in results.items() if math.isfinite(r["p"])]

    # ---- output
    def metrics_json(self) -> dict:
        return {
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alpha": ALPHA, "n_boot": self.n_boot, "holm_family": self.holm_family,
            "results_dirs": {lab: {"path": rd.path, "kind": rd.kind, "agents": rd.agents,
                                   "seeds": rd.seeds,
                                   "config": {k: v for k, v in rd.config.items()
                                              if k not in ("wall_time_s", "job_time_s", "constants")},
                                   "constants_sha256": rd.config.get("constants_sha256")}
                             for lab, rd in self.by_label.items()},
            "primaries": self.primaries,
            "agents": {lab: {a: {k: v for k, v in A.items() if k != "series"} for a, A in agg.items()}
                       for lab, agg in self.agg.items()},
            "secondaries": {lab: {k: v for k, v in s.items() if k != "p6_pair_cov90_series"}
                            for lab, s in self.sec.items()},
            "problems": self.problems,
        }


# =========================================================================== 6. figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                         "legend.frameon": False, "font.size": 9, "axes.titlesize": 10})
    return plt


def _line(ax, x, y, agent, **kw):
    c, ls, lw = _style(agent)
    ax.plot(x, y, color=c, linestyle=ls, linewidth=lw, label=agent, **kw)


def fig_nmse(rep: Report, label: str, path: str):
    plt = _plt()
    agg, per_seed = rep.agg[label], rep.per_seed[label]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})
    T = None
    for a in _sorted_agents(agg):
        s = agg[a]["series"].get("nmse_series")
        if s is None:
            continue
        T = len(s)
        _line(a1, np.arange(1, T + 1), s, a)
    if T:
        # shift markers: one tick per (seed, qualifying shift) along the top of the axis
        ts = [t for lst in rep.shift_ts[label].values() for t in lst]
        if ts:
            a1.plot(ts, np.full(len(ts), 1.0), transform=a1.get_xaxis_transform(), marker="|",
                    linestyle="none", color=_C["gray"], alpha=0.35, markersize=8, clip_on=False)
    a1.set_yscale("log")
    a1.set_xlabel("episode t")
    a1.set_ylabel("nMSE (mean over seeds)")
    a1.set_title(f"{label}: nMSE over episodes (ticks at top = shift episodes, all seeds)")
    a1.legend(ncol=1, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    # shift-aligned panel
    offs = np.arange(-3, 7)
    for a in _sorted_agents(per_seed):
        rows = []
        for S in per_seed[a]:
            ser = S["nmse_series"]
            for t in S["shift_ts"]:
                idx = t - 1 + offs
                ok = (idx >= 0) & (idx < len(ser))
                row = np.full(len(offs), np.nan)
                row[ok] = ser[idx[ok]]
                rows.append(row)
        if rows:
            with np.errstate(all="ignore"):
                _line(a2, offs, np.nanmean(np.vstack(rows), axis=0), a)
    a2.axvline(0, color=_C["gray"], linewidth=0.8)
    a2.set_yscale("log")
    a2.set_xlabel("episodes since shift (m >= 0.05)")
    a2.set_title("shift-aligned mean nMSE (exploratory)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_coverage_vs_width(rep: Report, label: str, path: str):
    plt = _plt()
    agg = rep.agg[label]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, lvl, nominal in zip(axes, ("50", "90", "99"), (0.5, 0.9, 0.99)):
        for a in _sorted_agents(agg):
            st = agg[a]["calibration"].get("stable")
            if not st:
                continue
            c, _, _ = _style(a)
            x, y = st.get(f"width{lvl}", math.nan), st.get(f"cov{lvl}", math.nan)
            if lvl == "90":
                ci = st["cov90_ci"]
                ax.errorbar([x], [y], yerr=[[y - ci["lo"]], [ci["hi"] - y]], color=c, capsize=2, linewidth=1)
            ax.plot([x], [y], "o", color=c, markersize=7, markeredgecolor="white", label=a)
        ax.axhline(nominal, color=_C["gray"], linewidth=0.8, linestyle="--")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel(f"mean normalized width{lvl}")
        ax.set_ylabel(f"coverage of the {lvl} % interval")
        ax.set_title(f"stable episodes, {lvl} % level")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle(f"{label}: coverage vs width (all predictions, stable episodes)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_risk_coverage(rep: Report, label: str, path: str):
    plt = _plt()
    agg = rep.agg[label]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for a in _sorted_agents(agg):
        if a in ("marginal-mean", "oracle"):
            continue                                # never abstain / zero width: no ordering
        rc = agg[a]["series"].get("risk_curve")
        if rc is not None:
            _line(ax, M.COVERAGE_GRID, rc, a)
    ax.set_yscale("log")
    ax.set_xlabel("coverage c (fraction of predictions kept, lowest width90 first)")
    ax.set_ylabel("selective risk = nMSE of kept predictions")
    ax.set_title(f"{label}: risk-coverage curves (mean over seeds)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shd(rep: Report, label: str, path: str) -> bool:
    plt = _plt()
    agg = rep.agg[label]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    drawn = False
    for a in _sorted_agents(agg):
        s = agg[a]["series"].get("shd_series")
        if s is not None:
            _line(ax, np.arange(1, len(s) + 1), s, a)
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ts = [t for lst in rep.shift_ts[label].values() for t in lst]
    if ts:
        ax.plot(ts, np.full(len(ts), 1.0), transform=ax.get_xaxis_transform(), marker="|",
                linestyle="none", color=_C["gray"], alpha=0.35, markersize=8, clip_on=False)
    ax.set_xlabel("episode t")
    ax.set_ylabel("SHD to the true visible DAG")
    ax.set_title(f"{label}: structure recovery (mean over seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_budget_sweep(rep: Report, path: str) -> bool:
    plt = _plt()
    points: dict[str, dict[int, dict]] = {}          # agent -> B -> agg
    for label, rd in rep.by_label.items():
        if rd.kind in ("sweep", "main") and label in rep.agg:
            B = int(rd.config.get("budget", 50))
            for a, A in rep.agg[label].items():
                if a in ("mech-full", "mech-random", "mech-full-nofloor", "int-pairwise"):
                    points.setdefault(a, {})[B] = A
    if not points or all(len(v) < 2 for v in points.values()):
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for a in _sorted_agents(points):
        Bs = sorted(points[a])
        c, ls, lw = _style(a)
        for ax, key in ((a1, "nmse_20_60"), (a2, "shd_10_60")):
            xs = [B for B in Bs if key in points[a][B]]
            if not xs:
                continue
            ys = [points[a][B][key]["mean"] for B in xs]
            lo = [points[a][B][key]["mean"] - points[a][B][key]["lo"] for B in xs]
            hi = [points[a][B][key]["hi"] - points[a][B][key]["mean"] for B in xs]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=c, linestyle=ls, linewidth=lw, marker="o",
                        markersize=5, capsize=2, label=a)
    a1.set_xscale("log")
    a1.set_xticks([10, 25, 50, 100])
    a1.set_xticklabels(["10", "25", "50", "100"])
    a1.set_yscale("log")
    a1.set_xlabel("budget B")
    a1.set_ylabel("mean nMSE, episodes 20-60")
    a1.set_title("value of interventions: nMSE vs B (95 % CI over seeds)")
    if a1.get_legend_handles_labels()[0]:
        a1.legend(fontsize=7)
    a2.set_xscale("log")
    a2.set_xticks([10, 25, 50, 100])
    a2.set_xticklabels(["10", "25", "50", "100"])
    a2.set_xlabel("budget B")
    a2.set_ylabel("mean SHD, episodes 10-60")
    a2.set_title("structure vs B")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_credit_confusion(rep: Report, path: str) -> bool:
    plt = _plt()
    rows = []
    for label, agg in rep.agg.items():
        for a in _sorted_agents(agg):
            C = agg[a].get("credit")
            if C and C["n_shifts"]:
                rows.append((f"{label} / {a}", C["confusion"]))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    parts = [("correct", _C["aqua"]), ("wrong-mechanism", _C["orange"]), ("miss", _C["red"])]
    for key, color in parts:
        vals = np.array([r[1].get(key, 0) for r in rows], dtype=float)
        ax.barh(y, vals, left=left, color=color, label=key, height=0.6, edgecolor="white", linewidth=1)
        left += vals
    for k, (name, conf) in enumerate(rows):
        ax.text(left[k] + 0.3, k, f"false alarms: {conf.get('false-alarm', 0)}", va="center",
                fontsize=7.5, color=_C["gray"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("qualifying shifts (m >= 0.05), pooled over seeds")
    ax.set_title("credit assignment per shift: reset of the shifted mechanism in {t, t+1}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_hidden_pair(rep: Report, label: str, path: str) -> bool:
    plt = _plt()
    sec = rep.sec.get(label, {})
    cov = sec.get("p6iii_cov90_pair")
    if not cov:
        return False
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.9))
    agents = _sorted_agents(cov)
    for k, a in enumerate(agents):
        c, _, _ = _style(a)
        ci = cov[a]
        a1.bar(k, ci["mean"], color=c, width=0.7, edgecolor="white")
        a1.errorbar(k, ci["mean"], yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]], color=_C["ink"], capsize=2, linewidth=1)
    a1.axhline(0.9, color=_C["gray"], linestyle="--", linewidth=0.8)
    a1.axhline(0.5, color=_C["red"], linestyle=":", linewidth=0.8)
    a1.set_xticks(range(len(agents)))
    a1.set_xticklabels(agents, rotation=35, ha="right", fontsize=7.5)
    a1.set_ylabel("90 % coverage on do(A) -> B, episodes 40-60")
    a1.set_title("(A, B): coverage per agent (P6 iii)")
    vals = [s.get("p6ii_ratio") for s in rep.sec_per_seed.get(label, [])]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if vals:
        a2.hist(vals, bins=15, color=_C["blue"], edgecolor="white")
    a2.axvspan(0.3, 1.0, color=_C["aqua"], alpha=0.15, label="predicted band [0.3, 1.0]")
    a2.set_xlabel("mech-full slope error on do(A) -> B divided by c (per seed)")
    a2.set_ylabel("seeds")
    a2.set_title("(A, B): diluted omitted-variable bias (P6 ii)")
    a2.legend(fontsize=7)
    series = sec.get("p6_pair_cov90_series", {})
    for a in _sorted_agents(series):
        if a in ("mech-full", "mech-full-diag", "oracle-structure", "int-pairwise", "obs-cumulative"):
            s = series[a]
            _line(a3, np.arange(1, len(s) + 1), s, a)
    a3.axhline(0.9, color=_C["gray"], linestyle="--", linewidth=0.8)
    a3.set_xlabel("episode t")
    a3.set_ylabel("90 % coverage on do(A) -> B")
    a3.set_title("(A, B): coverage per episode (diag: B fit on do(A) rows)")
    a3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_objectives(rep: Report, label: str, path: str) -> bool:
    plt = _plt()
    agg = rep.agg[label]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    drawn = False
    for a in _sorted_agents(agg):
        ss = agg[a]["series"].get("self_score_series")
        dec = agg[a]["series"].get("var_obj_decrease_series")
        if ss is None or not np.isfinite(ss).any():
            continue
        drawn = True
        _line(a1, np.arange(1, len(ss) + 1), ss, a)
        if dec is not None:
            _line(a2, np.arange(1, len(dec) + 1), dec, a)
    if not drawn:
        plt.close(fig)
        return False
    a1.set_xlabel("episode t")
    a1.set_ylabel("self log-score (mean over B samples)")
    a1.set_title("internal objective 1: proper score on the agent's own samples")
    a2.set_xlabel("episode t")
    a2.set_ylabel("var_obj before - after")
    a2.set_yscale("symlog", linthresh=0.01)
    a2.set_title("internal objective 2: posterior-variance reduction")
    a1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_detection_roc(constants: dict, path: str) -> bool:
    roc = (constants or {}).get("lambda_details", {}).get("roc")
    if not roc:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = [r["false_reset_rate"] for r in roc]
    y = [r["recall_window"] for r in roc]
    ax.plot(x, y, color=_C["blue"], marker="o", linewidth=2)
    for r, xx, yy in zip(roc, x, y):
        ax.annotate(f"{r['multiplier']} x lambda*", (xx, yy), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("false-reset rate per mechanism-episode (stable periods)")
    ax.set_ylabel("recall on shifts with m >= 0.05 (reset of j in {t, t+1})")
    ax.set_title("detection ROC on validation seeds (exploratory; from constants.json)")
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def make_figures(rep: Report, fig_dir: str, constants: dict | None) -> dict[str, str]:
    """Write every figure that has data; returns {figure key: relative path}."""
    os.makedirs(fig_dir, exist_ok=True)
    made: dict[str, str] = {}

    def attempt(key, fn, *args):
        p = os.path.join(fig_dir, f"{key}.png")
        try:
            ok = fn(*args, p)
            if ok is not False:
                made[key] = os.path.relpath(p, os.path.dirname(fig_dir))
        except Exception:                           # noqa: BLE001
            rep.problems.append(f"figure {key} failed:\n{traceback.format_exc()}")

    for label in rep.agg:
        attempt(f"nmse_{label}", fig_nmse, rep, label)
        attempt(f"coverage_vs_width_{label}", fig_coverage_vs_width, rep, label)
        attempt(f"risk_coverage_{label}", fig_risk_coverage, rep, label)
        attempt(f"shd_{label}", fig_shd, rep, label)
        attempt(f"objectives_{label}", fig_objectives, rep, label)
        if rep.by_label[label].kind == "hidden":
            attempt(f"hidden_pair_{label}", fig_hidden_pair, rep, label)
    attempt("budget_sweep", fig_budget_sweep, rep)
    attempt("credit_confusion", fig_credit_confusion, rep)
    if constants:
        attempt("detection_roc", fig_detection_roc, constants)
    return made


# =========================================================================== 7. REPORT.md

def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def ci_str(ci, nd=3):
    if not ci or ci.get("n", 0) == 0 or not math.isfinite(ci.get("mean", math.nan)):
        return "—"
    return f"{fmt(ci['mean'], nd)} [{fmt(ci['lo'], nd)}, {fmt(ci['hi'], nd)}]"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _agent_rows(agg: dict, keys: list[tuple[str, str]], nd=3, sel=None) -> list[list]:
    rows = []
    for a in _sorted_agents(agg):
        A = agg[a]
        row = [f"`{a}`"]
        for key, kind in keys:
            if kind == "ci":
                row.append(ci_str(A.get(key), nd))
            elif kind == "sel_ci":
                row.append(ci_str(A["selective"].get(key), nd))
            elif kind == "cal":
                st, f = key.split(":")
                row.append(fmt(A["calibration"].get(st, {}).get(f), nd))
            elif kind == "credit_ci":
                row.append(ci_str(A.get("credit", {}).get(key), nd))
        rows.append(row)
    return rows


def write_report_md(rep: Report, figs: dict[str, str], out_path: str, constants: dict | None):
    L: list[str] = []
    P = rep.primaries
    L.append("# Mechanism-Shift Testbed — report\n")
    L.append(f"Generated {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
             f"Bootstrap resamples: {rep.n_boot}; alpha = {ALPHA} (Holm across the "
             f"{len(rep.holm_family)} testable primaries: {', '.join(rep.holm_family) or 'none'}).\n")
    L.append("## Runs read\n")
    rows = []
    for lab, rd in rep.by_label.items():
        c = rd.config
        rows.append([f"`{lab}`", rd.kind, c.get("budget"), c.get("shift_type"), c.get("hetero"),
                     "yes" if c.get("hidden") else "no", f"{len(rd.seeds)}", f"{len(rd.agents)}",
                     (c.get("git_hash") or "—")[:10], (c.get("constants_sha256") or "—")[:12],
                     fmt(c.get("total_wall_s"), 0)])
    L.append(table(["label", "kind", "B", "shift type", "γ_h", "hidden", "seeds", "agents",
                    "git", "constants sha256", "wall s"], rows))
    kinds_present = {rd.kind for rd in rep.dirs}
    for want in ("main", "sweep", "hidden", "knob"):
        if want not in kinds_present:
            L.append(f"- No `{want}` results directory was given; predictions that need it are NOT TESTABLE.\n")

    # ---- summary of verdicts
    L.append("\n## Pre-registered predictions — verdicts (Holm-adjusted, alpha = 0.05)\n")
    rows = []
    for name in PRIMARY_KIND:
        r = P[name]
        rows.append([f"**{name}**", PRIMARY_CLAIM[name], f"`{r['run'] or '—'}`", r["n_seeds"],
                     ci_str(r.get("ci")) + (f" (fraction inside band {fmt(r['fraction'], 2)})" if "fraction" in r else ""),
                     fmt(r.get("p"), 4), fmt(r.get("p_adj"), 4),
                     f"**{r['verdict']}**" + (f" — {r['not_testable_reason']}" if r["verdict"] == "NOT TESTABLE" else "")])
    L.append(table(["P", "claim", "run", "seeds", "primary scalar: mean [95 % CI]", "p (one-sided)",
                    "Holm p", "verdict"], rows))
    L.append("A prediction is HELD iff its Holm-adjusted one-sided bootstrap test rejects in the "
             "predicted direction (SPEC 'Analysis plan'). CIs are unadjusted percentile-bootstrap "
             "intervals over per-seed scalars; differences between agents are paired by seed.\n")

    def sec(label):
        return rep.sec.get(label, {})

    def agg(label):
        return rep.agg.get(label, {})

    main = rep._label_for_primary("P1")
    sweep10 = "sweep_B10" if "sweep_B10" in rep.agg else None
    sweep100 = "sweep_B100" if "sweep_B100" in rep.agg else None
    sweep25 = "sweep_B25" if "sweep_B25" in rep.agg else None
    hidden = "hidden" if "hidden" in rep.agg else None
    knob_h1 = "knob_large_h1" if "knob_large_h1" in rep.agg else None
    knob_small = "knob_small_h0" if "knob_small_h0" in rep.agg else None
    knob_noise = rep._label_for_primary("P9")

    def primary_block(name):
        r = P[name]
        L.append(f"\n### {name} — {PRIMARY_CLAIM[name]}\n")
        if r["kind"] == "interval":
            test = f"mean inside [{fmt(-r['threshold'])}, {fmt(r['threshold'])}]"
        elif r["kind"] == "fraction":
            test = f"fraction of seeds inside the band > {fmt(r['threshold'])}"
        else:
            test = f"mean {r['direction']} {fmt(r['threshold'])}"
        L.append(table(["primary scalar", "run", "seeds", "mean [95 % CI]", "test", "p", "Holm p", "verdict"],
                       [[r["description"], f"`{r['run'] or '—'}`", r["n_seeds"], ci_str(r.get("ci")), test, fmt(r.get("p"), 4),
                         fmt(r.get("p_adj"), 4), f"**{r['verdict']}**"]]))
        if r["reasons"]:
            L.append("Per-seed values that could not be computed: " +
                     "; ".join(f"seed {s}: {why}" for s, why in list(r["reasons"].items())[:6]) +
                     (" …" if len(r["reasons"]) > 6 else "") + "\n")
        if r["verdict"] == "NOT TESTABLE":
            L.append(f"NOT TESTABLE because: {r.get('not_testable_reason', 'no finite per-seed value')}.\n")

    # ---- P1
    primary_block("P1")
    if main:
        A, S = agg(main), sec(main)
        rows = [["mech-full nMSE below obs-window at episode 5 (fraction of seeds; predicted >= 0.9)",
                 ci_str(S.get("p1_mech_below_obs_ep5"), 2)]]
        for a in ("obs-window", "obs-cumulative", "int-pairwise", "mech-full", "marginal-mean"):
            if a in A:
                rows.append([f"`{a}` nMSE on non-descendant pairs (false effects; obs-* predicted bounded away from 0)",
                             ci_str(A[a].get("nmse_non_descendant"))])
                rows.append([f"`{a}` nMSE on descendant pairs", ci_str(A[a].get("nmse_descendant"))])
        L.append("Exploratory (P1 secondaries):\n\n" + table(["quantity", "mean [95 % CI] over seeds"], rows))
    # ---- P2
    primary_block("P2")
    if main:
        A = agg(main)
        rows = []
        for a in ("mech-full", "mech-oracle-detect", "mech-random", "mech-reset-all", "mech-no-detect",
                  "obs-window", "obs-cumulative", "int-pairwise", "oracle-structure"):
            if a in A and "recovery" in A[a]:
                R = A[a]["recovery"]
                rows.append([f"`{a}`", ci_str(A[a].get("regret_mean")), ci_str(R["median"] | {"mean": R["median"]["point"], "n": R["n_events"]}, 1),
                             ci_str(R["within_1"] | {"mean": R["within_1"]["point"], "n": R["n_events"]}, 2),
                             ci_str(R["within_2"] | {"mean": R["within_2"]["point"], "n": R["n_events"]}, 2),
                             ci_str(R["within_3"] | {"mean": R["within_3"]["point"], "n": R["n_events"]}, 2),
                             fmt(R["censored_fraction"], 2), R["n_events"]])
        L.append("Exploratory (P2 secondaries; regret is vs `oracle-structure` over the shift episode "
                 "and the three following; recovery = Kaplan–Meier on `m >= 0.05` shifts with a "
                 "seed-level cluster bootstrap; 'within k' counts the shift episode as the first):\n\n" +
                 table(["agent", "regret per shift", "KM median recovery (episodes)", "recovered within 1",
                        "within 2 (mech-full predicted >= 0.9)", "within 3", "censored fraction (obs-cumulative predicted >= 0.5)", "events"], rows))
    # ---- P3
    primary_block("P3")
    if main:
        A, S = agg(main), sec(main)
        rows = []
        for a in _sorted_agents(A):
            C = A[a]["calibration"]
            rows.append([f"`{a}`"] + [fmt(C.get(st, {}).get("cov90"), 3) + " / " + fmt(C.get(st, {}).get("width90"), 2)
                                      for st in ("stable", "shift", "post_shift", "shd_zero", "shd_positive", "path_equal", "path_unequal")]
                        + [fmt(A[a].get("abstain_rate", {}).get("mean"), 3)])
        L.append("Exploratory (Metric 4; each cell is 90 % coverage / mean normalized width90, on ALL "
                 "predictions of the stratum; SHD and path strata exist for learned-structure agents only):\n\n" +
                 table(["agent", "stable", "shift", "post-shift", "SHD = 0", "SHD > 0", "paths equal", "paths differ", "abstain rate"], rows))
        p3b = S.get("p3b")
        if p3b:
            L.append("P3(b) (exploratory): shift episodes with `m >= 0.05` where the detector did NOT reset the "
                     "shifted mechanism at step 4 — mech-full on j-dependent queries:\n\n" +
                     table(["quantity", "value"],
                           [["undetected / qualifying shifts (pooled)", f"{p3b['n_undetected']} / {p3b['n_shifts']}"],
                            ["90 % coverage on j-dependent queries in the shift episode (predicted < 0.5)", ci_str(p3b["cov90_first"])],
                            ["abstention excess over the stable rate, points (predicted <= 0.02)", ci_str(p3b["abstain_excess"])]]))
        if "int-pairwise" in A:
            st = A["int-pairwise"]["calibration"].get("stable", {})
            L.append(f"P3(c) (exploratory): `int-pairwise` stable-episode 90 % coverage = {ci_str(st.get('cov90_ci'))} (predicted within ±0.05 of 0.9). ")
        if sweep10 and "p3c_ratio_ep_last" in sec(sweep10):
            L.append(f"At B = 10, nMSE(int-pairwise)/nMSE(mech-full) at the last episode = {ci_str(sec(sweep10)['p3c_ratio_ep_last'], 2)} (predicted >= 3).\n")
        else:
            L.append("\n")
    # ---- P4
    primary_block("P4")
    if main:
        A = agg(main)
        rows = [[f"`{a}`", ci_str(A[a].get("forgetting_mean")), ci_str(A[a].get("regret_mean"))]
                for a in ("mech-reset-all", "mech-full", "obs-window", "int-pairwise", "mech-no-detect",
                          "mech-random", "mech-oracle-detect", "oracle-structure") if a in A]
        L.append("Exploratory (Metric 3; forgetting bump = nMSE on j-independent queries in the shift "
                 "episode minus its mean over the two previous episodes; predicted: mech-reset-all >= 0.05, "
                 "mech-full < 0.01, windowed baselines show a bump, mech-no-detect none):\n\n" +
                 table(["agent", "forgetting bump per shift", "regret per shift"], rows))
    # ---- P5
    primary_block("P5")
    rows = []
    for lab in (sweep10, sweep25, main, sweep100):
        if lab and lab in rep.sec:
            S = sec(lab)
            B = rep.by_label[lab].config.get("budget")
            rows.append([f"B = {B} (`{lab}`)", ci_str(S.get("p5a_int_minus_full")), ci_str(S.get("p5b_full_minus_random")),
                         ci_str(S.get("p5c_shd_full_minus_random"), 2), ci_str(S.get("p5c_shd_nofloor_minus_random"), 2)])
    if rows:
        L.append("Exploratory (P5 secondaries; paired per-seed differences; nMSE over episodes 20-60, SHD over 10-60):\n\n" +
                 table(["budget", "(a) int-pairwise − mech-full nMSE (> 0 at 10; within ±0.02 at 100)",
                        "(b) mech-full − mech-random nMSE (< 0 at 10, 25; closes at 100)",
                        "(c) SHD mech-full − mech-random (predicted <= 0)", "(c) SHD nofloor − mech-random (> 0 at 10)"], rows))
    if "budget_sweep" in figs:
        L.append(f"![budget sweep]({figs['budget_sweep']})\n")
    # ---- P6
    primary_block("P6")
    if hidden:
        S, A = sec(hidden), agg(hidden)
        rows = [["(i) mech-full SHD = 0 at episode 10 (fraction of seeds; predicted >= 0.8)", ci_str(S.get("p6i_shd_zero_at_10"), 2)],
                ["(ii) slope error on do(A) → B divided by c, mech-full, episodes 40-60, |v| >= 1", ci_str(S.get("p6ii_ratio"), 2)],
                ["(ii) fraction of seeds with the ratio in [0.3, 1.0] (predicted >= 0.8)", fmt(S.get("p6ii_ratio", {}).get("fraction_in_band"), 2)],
                ["(iii) mech-full abstention on (A, B) minus its global rate, points (predicted <= 0.02)", ci_str(S.get("p6iii_abstain_excess_pair"))],
                ["(iv) H shift resets both A and B in the same episode (fraction of seeds; predicted >= 0.5)", ci_str(S.get("p6iv_both_fire"), 2)],
                ["(iv) H shift resets at least one visible mechanism (fraction of seeds)", ci_str(S.get("p6iv_any_fire"), 2)],
                ["population confounding bias c (mean over seeds)", ci_str(S.get("conf_c"))],
                ["implied nMSE floor F_conf on do(A) → B", ci_str(S.get("conf_F"))]]
        L.append("Exploratory (P6 secondaries):\n\n" + table(["claim", "value"], rows))
        cov = S.get("p6iii_cov90_pair", {})
        if cov:
            L.append("90 % coverage on do(A) → B queries, episodes 40-60, per agent (diagnostic `mech-full-diag` "
                     "predicted within ±0.10 of 0.9; `obs-*` and `oracle-structure` predicted biased):\n\n" +
                     table(["agent", "coverage on (A, B)", "overall nMSE", "SHD 10-60"],
                           [[f"`{a}`", ci_str(cov[a]), ci_str(A.get(a, {}).get("nmse")), ci_str(A.get(a, {}).get("shd_10_60"), 2)] for a in _sorted_agents(cov)]))
        if f"hidden_pair_{hidden}" in figs:
            L.append(f"![hidden pair]({figs[f'hidden_pair_{hidden}']})\n")
    # ---- P7
    primary_block("P7")
    rows = []
    for lab in [main, knob_h1, knob_small, knob_noise] + [l for l in rep.agg if rep.by_label[l].kind == "knob" and l not in (knob_h1, knob_small, knob_noise)]:
        if not lab:
            continue
        A = agg(lab)
        for a in ("mech-full", "mech-oracle-detect"):
            if a in A and "credit" in A[a]:
                C = A[a]["credit"]
                rows.append([f"`{lab}`", f"`{a}`", C["n_shifts"], C["n_shifts_excluded"], C["n_resets"],
                             ci_str(C["accuracy_delay0"], 2), ci_str(C["recall_window"], 2), ci_str(C["misattribution_rate"], 2),
                             ci_str(C["attribution_precision"], 2), ci_str(C["false_reset_rate"], 4),
                             f"{C['confusion']['correct']} / {C['confusion']['wrong-mechanism']} / {C['confusion']['miss']} / {C['confusion']['false-alarm']}",
                             ci_str(A[a].get("nmse"))])
    if rows:
        L.append("Exploratory (Metric 8 across configurations; predicted: main false-reset rate <= 0.02; under γ_h = 1 "
                 "mech-full's accuracy drops >= 15 points and misattribution rises while mech-oracle-detect's nMSE "
                 "moves < 0.02):\n\n" +
                 table(["run", "agent", "shifts", "excluded (m < 0.05)", "resets", "accuracy (delay 0)", "recall in {t, t+1}",
                        "misattribution rate", "attribution precision", "false-reset rate / mech-episode",
                        "correct / wrong / miss / false alarms", "nMSE"], rows))
    if "credit_confusion" in figs:
        L.append(f"![credit assignment]({figs['credit_confusion']})\n")
    # ---- P8
    primary_block("P8")
    if main:
        A = agg(main)
        rows = []
        for a in ("mech-full", "mech-overconfident", "mech-random", "oracle-structure"):
            if a in A:
                rows.append([f"`{a}`", ci_str(A[a].get("self_score_10_60"), 2), ci_str(A[a].get("var_obj_decrease_1_10"), 3),
                             ci_str(A[a].get("abstain_rate"), 3), fmt(A[a]["calibration"].get("stable", {}).get("cov90"), 3),
                             ci_str(A[a]["selective"].get("aurc")), ci_str(A[a]["selective"].get("aurc_oracle")),
                             ci_str(A[a]["selective"].get("aurc_random")), ci_str(A[a].get("nmse"))])
        L.append("Exploratory (P8 secondaries; predicted: mech-overconfident has the larger var_obj decrease in "
                 "episodes 1-10, abstains < 2 %, and has 90 % coverage >= 20 points worse and AURC worse than mech-full):\n\n" +
                 table(["agent", "self log-score (10-60)", "var_obj decrease (1-10)", "abstain rate", "cov90 stable",
                        "AURC", "AURC oracle order", "AURC random order", "nMSE"], rows))
        if f"objectives_{main}" in figs:
            L.append(f"![internal objectives]({figs[f'objectives_{main}']})\n")
    # ---- P9
    primary_block("P9")
    if knob_noise:
        A = agg(knob_noise)
        rows = [[f"`{a}`", ci_str(A[a].get("regret_mean")), ci_str(A[a].get("nmse")),
                 ci_str(A[a].get("credit", {}).get("false_reset_rate"), 4) if "credit" in A[a] else "—"]
                for a in ("mech-full", "mech-oracle-detect", "int-pairwise", "oracle-structure") if a in A]
        L.append("Exploratory (noise-only; regret vs oracle-structure; `mech-oracle-detect` is predicted to pay the same price):\n\n" +
                 table(["agent", "regret per noise-only shift (δσ >= 0.05)", "nMSE", "false-reset rate"], rows))
    if knob_small:
        A, S = agg(knob_small), sec(knob_small)
        C = A.get("mech-full", {}).get("credit")
        p9 = S.get("p9_small")
        rows = []
        if C:
            rows.append(["detection recall on m >= 0.05 shifts, reset in {t, t+1} (predicted < 0.6)", ci_str(C["recall_window"], 2)])
            rows.append(["detection recall, any later episode before the next shift", ci_str(C["recall_any"], 2)])
        if p9:
            rows.append(["undetected / qualifying shifts", f"{p9['n_undetected']} / {p9['n_shifts']}"])
            rows.append(["undetected shifts: max 90 % coverage on j-dependent queries over {t, t+1} (predicted < 0.7)", ci_str(p9["cov90_max_first_two"])])
            rows.append(["undetected shifts: abstention excess over the stable rate (predicted no rise)", ci_str(p9["abstain_excess"])])
        if rows:
            L.append("Exploratory (`small` shifts):\n\n" + table(["quantity", "value"], rows))

    # ---- exploratory per-run tables
    L.append("\n## Exploratory tables per run\n")
    for lab in rep.agg:
        A = agg(lab)
        L.append(f"\n### `{lab}` — nMSE (Metric 1) and selective risk (Metric 5)\n")
        L.append(table(["agent", "nMSE", "stable", "shift", "post-shift", "descendant", "non-descendant",
                        "non-abstained", "abstain rate", "AURC", "mistake recall", "mistake precision", "SHD 10-60", "wall s"],
                       _agent_rows(A, [("nmse", "ci"), ("nmse_stable", "ci"), ("nmse_shift", "ci"), ("nmse_post_shift", "ci"),
                                       ("nmse_descendant", "ci"), ("nmse_non_descendant", "ci"), ("nmse_non_abstained", "ci"),
                                       ("abstain_rate", "ci"), ("aurc", "sel_ci"), ("mistake_recall", "sel_ci"),
                                       ("mistake_precision", "sel_ci"), ("shd_10_60", "ci"), ("wall_time_s", "ci")])))
        if "marginal-mean" in A:
            L.append(f"`marginal-mean` floor: nMSE {ci_str(A['marginal-mean'].get('nmse'))}. ")
        L.append(f"Qualifying shifts pooled over seeds: {next(iter(A.values()))['n_shifts_qualifying']}; "
                 f"excluded by the m >= 0.05 filter: {next(iter(A.values()))['n_shifts_excluded']}.\n")
        for key in (f"nmse_{lab}", f"coverage_vs_width_{lab}", f"risk_coverage_{lab}", f"shd_{lab}"):
            if key in figs:
                L.append(f"![{key}]({figs[key]})\n")

    if constants:
        L.append("\n## Calibration (constants.json, validation seeds; exploratory)\n")
        ld = constants.get("lambda_details", {})
        L.append(f"lambda* = {fmt(constants.get('lambda'), 3)}; no-shift pooled exceedance rate "
                 f"{fmt(ld.get('pooled_exceedance_rate_at_lambda'), 4)}, offline Page–Hinkley re-simulation "
                 f"{fmt(ld.get('resimulated_false_reset_rate_at_lambda'), 4)}, empirical false-reset rate with shifts "
                 f"at lambda* {fmt(ld.get('false_reset_rate_at_lambda'), 4)} (per mechanism-episode).\n")
        if ld.get("roc"):
            L.append(table(["multiplier", "lambda", "recall in {t, t+1}", "accuracy (delay 0)", "false-reset rate", "misattribution rate"],
                           [[r["multiplier"], fmt(r["lambda"], 2), fmt(r["recall_window"], 2), fmt(r["accuracy_delay0"], 2),
                             fmt(r["false_reset_rate"], 4), fmt(r["misattribution_rate"], 2)] for r in ld["roc"]]))
        if "detection_roc" in figs:
            L.append(f"![detection ROC]({figs['detection_roc']})\n")
        L.append(table(["agent", "W", "gamma", "tau"],
                       [[f"`{a}`", fmt(e.get("W"), 0), fmt(e.get("gamma"), 2), fmt(e.get("tau"), 3)]
                        for a, e in constants.get("agents", {}).items()]))

    L.append("\n## Reading notes\n")
    L.append("- Per-seed scalars first, CIs over seeds (percentile bootstrap); event-pooled statistics use a seed-level cluster bootstrap.\n"
             "- Shift-conditional numbers use shifts with `m >= 0.05` (`delta_sigma` for noise-only); the excluded count is stated per run.\n"
             "- Coverage is always on all predictions, never the non-abstained subset. Abstention is the agent's own raw width90 against its tuned tau.\n"
             "- P6(ii) reads 'at v = 2' as the slope error per unit v over queries with |v| >= 1 (queries draw v ~ U(-2, 2)).\n"
             "- 'Detected at step 4' means a `reset_events` row (t, j) for the shifted mechanism j in the shift episode t.\n")
    if rep.problems:
        L.append("\n## Problems encountered while building this report\n")
        for p in rep.problems:
            L.append("```\n" + p.strip() + "\n```\n")
    with open(out_path, "w") as f:
        f.write("\n".join(L))


# =========================================================================== 8. entry point

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def build_report(result_dirs: list[str], out_dir: str, constants_path: str | None = None,
                 n_boot: int = M.N_BOOT, log=print) -> Report:
    dirs = discover(result_dirs)
    constants = None
    if constants_path is None:                  # fall back to the constants recorded in a config
        for rd in dirs:
            if rd.config.get("constants"):
                constants = rd.config["constants"]
                break
    else:
        with open(constants_path) as f:
            constants = json.load(f)
    rep = Report(dirs, n_boot=n_boot, log=log).build()
    os.makedirs(out_dir, exist_ok=True)
    figs = make_figures(rep, os.path.join(out_dir, "figures"), constants)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(_jsonable(rep.metrics_json()), f, indent=1)
    write_report_md(rep, figs, os.path.join(out_dir, "REPORT.md"), constants)
    log(f"report: wrote {out_dir}/metrics.json, {len(figs)} figures, REPORT.md; "
        f"verdicts: " + ", ".join(f"{n}={r['verdict']}" for n, r in rep.primaries.items()))
    if rep.problems:
        log(f"report: {len(rep.problems)} problem(s) recorded in REPORT.md")
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", nargs="+", help="results directories (each with config.json and raw/)")
    p.add_argument("--out", default=None, help="output dir (default: the first results dir)")
    p.add_argument("--constants", default=None, help="constants.json (for the calibration section)")
    p.add_argument("--n-boot", type=int, default=M.N_BOOT)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, **k, flush=True))
    build_report(args.results, args.out or args.results[0], args.constants, args.n_boot, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
