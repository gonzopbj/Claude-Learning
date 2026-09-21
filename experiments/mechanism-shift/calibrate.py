"""Tuning on validation seeds 1000-1009 -> constants.json (SPEC.md "Tuning and calibration";
INTERFACES.md section 5 for the file's schema).

    python calibrate.py --out constants.json [--seeds 1000-1009 --workers 4]
    python calibrate.py --out /tmp/c.json --smoke        # 1 seed, short runs, same code path

What is tuned, and how (each step is one function below):

  W, gamma   For obs-window (W in {1, 2, 3, 5}), int-pairwise (W in {3, 5, 8, 12}) and
             mech-no-detect (gamma in {0.5, 0.7, 0.9}): the grid value minimizing mean nMSE
             (SPEC Metric 1) over all validation episodes and seeds.  These agents have no
             detector, so the choice does not depend on lambda.
  lambda     One value for every detector agent.  mech-full runs on the validation seeds with
             the shift schedule DISABLED (World(..., shifts_enabled=False)) and lambda = inf,
             so the raw file holds the Page-Hinkley statistic g_t and this episode's GLR_t for
             every tested mechanism-episode without any reset ever happening.  lambda* is the
             smallest value giving <= 1 false reset per 100 mechanism-episodes under BOTH of
             two estimates of the false-reset rate (see `choose_lambda`): the pooled
             exceedance rate mean(g > lambda) of INTERFACES section 7, and an offline
             re-simulation of Page-Hinkley from the stored GLR_t that zeroes g after each
             simulated fire.  Neither estimate is exact (a real fire also truncates the buffer,
             which changes later GLR values), so the report's true check is the empirical
             false-reset rate of the with-shifts run at lambda* (`false_reset_rate_at_lambda`).
  tau        Per agent: the 90th percentile of the raw width90 (q0.95 - q0.05, the agent's
             own uncertainty score) over that agent's predictions in validation STABLE
             episodes, with shifts on and the tuned W / gamma / lambda in force.  Target
             abstention rate 10 % per agent, matched by construction.  marginal-mean and
             oracle never abstain (tau = null); mech-full-diag is hidden-only and copies
             mech-full's tau (recorded under "notes").
  ROC        Exploratory: mech-full with shifts at lambda in {0.5, 1, 2, 4, 8} x lambda*;
             recall on m >= 0.05 shifts vs false-reset rate (metrics.credit_assignment).

Every run goes through `run.run_job`, i.e. the same World / build_agent / protocol path as an
evaluation run; per-agent constants are built here as dicts (e.g. {"W": 8}) and never by
editing constants.json.  Seeds below 1000 are refused: no tuning on evaluation seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import metrics  # noqa: E402
import run as runner  # noqa: E402
from agents import MAIN_AGENTS, REGISTRY  # noqa: E402

SPEC_VERSION = "v2"
DEFAULT_VALIDATION_SEEDS = "1000-1009"
# SPEC "Tuning and calibration": the grids.
GRIDS = {
    "obs-window": ("W", (1, 2, 3, 5)),
    "int-pairwise": ("W", (3, 5, 8, 12)),
    "mech-no-detect": ("gamma", (0.5, 0.7, 0.9)),
}
LAMBDA_MULTIPLIERS = (0.5, 1, 2, 4, 8)
FALSE_RESET_TARGET = 0.01           # <= 1 false reset per 100 mechanism-episodes
TAU_PERCENTILE = 90.0               # 10 % abstention in stable episodes
NEVER_ABSTAIN = ("marginal-mean", "oracle")
DETECTOR_AGENTS = tuple(n for n in MAIN_AGENTS
                        if REGISTRY[n].kwargs.get("reset_rule") in ("per-mechanism", "all"))


# =========================================================================== worker

def _relevant_constants(agent: str, constants: dict) -> dict:
    """Only the keys the agent actually reads (INTERFACES section 2), so identical runs are
    recognised and shared between the tuning steps."""
    keep = {}
    if agent in GRIDS and GRIDS[agent][0] in constants:
        keep[GRIDS[agent][0]] = constants[GRIDS[agent][0]]
    if agent in DETECTOR_AGENTS and "lambda" in constants:
        keep["lambda"] = constants["lambda"]
    return keep


def summarize_raw(raw: dict) -> dict:
    """Reduce one raw result to what calibration needs (runs in the worker, so only a few
    hundred kB cross the process boundary instead of the whole raw file)."""
    stable = metrics.episode_kinds(raw)["stable"]
    out = {
        "nmse_mean": float(np.mean(np.asarray(raw["err"], dtype=np.float64) ** 2)),
        "nmse_series": metrics.nmse_series(raw).astype(np.float32),
        "width90_raw_stable": np.asarray(raw["width90_raw"])[stable].ravel().astype(np.float32),
        "g_stat": np.asarray(raw["g_stat"], dtype=np.float64),
        "glr_stat": np.asarray(raw["glr_stat"], dtype=np.float64),
        "learned_adj": np.asarray(raw["learned_adj"], dtype=bool),
        "struct_refit_events": np.asarray(raw["struct_refit_events"]).reshape(-1, 2),
        "reset_events": np.asarray(raw["reset_events"]).reshape(-1, 2),
        "n_shifts": int(np.sum(np.asarray(raw["shift_j"]) >= 0)),
        "wall_time_s": float(raw["wall_time_s"]),
    }
    if bool(raw["has_detector"]) or out["reset_events"].shape[0]:
        ca = metrics.credit_assignment(raw)
        out["credit"] = {k: ca[k] for k in ("n_shifts", "n_shifts_excluded", "n_resets",
                                            "recall_window", "recall_any", "accuracy_delay0",
                                            "false_reset_rate", "misattribution_rate",
                                            "attribution_precision", "confusion")}
    return out


def calib_job(spec: runner.JobSpec) -> runner.JobResult:
    """run.run_job followed by summarize_raw, inside the worker."""
    res = runner.run_job(spec, return_raw=True)
    if res.raw is not None:
        res.extra = summarize_raw(res.raw)
        res.raw = None
    return res


class _Runner:
    """Runs (agent, constants, shifts_enabled) x seeds through run.run_jobs and caches the
    summaries, so a configuration shared by two steps is simulated once."""

    def __init__(self, seeds, episodes, budget, n_obs, n_queries, workers, log):
        self.seeds = list(seeds)
        self.base = dict(episodes=episodes, budget=budget, n_obs=n_obs, n_queries=n_queries)
        self.workers = workers
        self.log = log
        self.cache: dict = {}            # (agent, frozen constants, shifts) -> {seed: summary}
        self.total_wall = 0.0

    @staticmethod
    def key(agent, constants, shifts_enabled):
        rel = _relevant_constants(agent, constants)
        return (agent, tuple(sorted(rel.items())), bool(shifts_enabled))

    def run(self, requests: list[tuple[str, dict, bool]]) -> None:
        """requests: (agent, constants, shifts_enabled).  Fills the cache."""
        specs, keys = [], []
        for agent, constants, shifts in requests:
            k = self.key(agent, constants, shifts)
            if k in self.cache or k in keys:
                continue
            keys.append(k)
            for s in self.seeds:
                specs.append(runner.JobSpec(agent=agent, seed=s, shifts_enabled=shifts,
                                            constants=dict(_relevant_constants(agent, constants)),
                                            out_path=None, **self.base))
        if not specs:
            return
        self.log(f"calibrate: {len(specs)} runs ({len(keys)} configurations x "
                 f"{len(self.seeds)} seeds) on {self.workers} workers")
        t0 = time.perf_counter()
        results = runner.run_jobs(specs, workers=self.workers, worker_fn=calib_job)
        for spec, res in zip(specs, results):
            if res.error:
                raise RuntimeError(f"validation run {spec.agent} seed {spec.seed} failed:\n{res.error}")
            k = self.key(spec.agent, spec.constants, spec.shifts_enabled)
            self.cache.setdefault(k, {})[spec.seed] = res.extra
            self.total_wall += res.wall_time_s
        self.log(f"calibrate: done in {time.perf_counter() - t0:.1f}s")

    def get(self, agent, constants, shifts_enabled=True) -> dict:
        return self.cache[self.key(agent, constants, shifts_enabled)]


# =========================================================================== lambda

def resimulate_page_hinkley(glr: np.ndarray, drift: np.ndarray, zero_before: np.ndarray,
                            lam: float) -> tuple[np.ndarray, int]:
    """Offline Page-Hinkley (SPEC "Shift detection") from stored per-episode GLR values.

    glr, drift: [T, d] (NaN in glr = N_min guard skipped the mechanism, which holds g at 0);
    zero_before[r, j]: g_j was zeroed at step 7 of the previous episode (parent-set change);
    lam: threshold.  g_t = max(0, g_{t-1} + GLR_t - drift); fire when g_t > lam, then g := 0.
    Returns (g after the update and before any fire-zeroing, number of fires).
    """
    T, d = glr.shape
    g = np.zeros(d)
    G = np.full((T, d), np.nan)
    fires = 0
    for r in range(T):
        g = np.where(zero_before[r], 0.0, g)
        guarded = ~np.isfinite(glr[r])
        g = np.where(guarded, 0.0, np.maximum(0.0, g + np.nan_to_num(glr[r]) - drift[r]))
        G[r] = np.where(guarded, np.nan, g)
        fired = (~guarded) & (g > lam)
        fires += int(fired.sum())
        g = np.where(fired, 0.0, g)
    return G, fires


def _drift_and_zeroing(summary: dict) -> tuple[np.ndarray, np.ndarray]:
    """drift[r, j] = p + 2 with p = |pa(j)| + 1 under the parent set in force at step 4 of
    episode t = r + 1, i.e. the adjacency after step 7 of episode t - 1 (learned_adj[r - 1];
    no parents before episode 1).  zero_before[r, j]: a structure-induced refit of j at
    episode t - 1 (struct_refit_events), which zeroes g_j (SPEC step 7)."""
    adj = summary["learned_adj"]
    T, d, _ = adj.shape
    n_par = np.zeros((T, d))
    n_par[1:] = adj[:-1].sum(axis=2)
    drift = (n_par + 1) + 2
    zero_before = np.zeros((T, d), dtype=bool)
    for t, j in summary["struct_refit_events"]:
        if 1 <= t < T:                  # refit at episode t affects step 4 of episode t + 1
            zero_before[int(t), int(j)] = True
    return drift, zero_before


def choose_lambda(summaries: dict, target: float = FALSE_RESET_TARGET) -> dict:
    """lambda* from mech-full's no-shift runs at lambda = inf (one summary per seed).

    Two estimates of the false-reset rate per mechanism-episode as a function of lambda:
      pooled:  mean over tested mechanism-episodes of 1[g_t > lambda] with the stored,
               never-reset g_t (INTERFACES section 7: "mean(g > lambda) <= 0.01");
      resim:   fires / tested mechanism-episodes when Page-Hinkley is re-run offline from
               GLR_t with g zeroed after every simulated fire.
    lambda* = the smallest candidate (a stored g value, so thresholds between consecutive g
    values behave identically) at which both are <= target.  The re-simulation at lambda =
    inf must reproduce the stored g (a self-check of the drift / zeroing reconstruction);
    its maximum discrepancy is recorded.
    """
    per_seed = []
    for seed in sorted(summaries):
        s = summaries[seed]
        drift, zero_before = _drift_and_zeroing(s)
        G_inf, _ = resimulate_page_hinkley(s["glr_stat"], drift, zero_before, math.inf)
        ok = np.isfinite(s["g_stat"]) & np.isfinite(G_inf)
        disc = float(np.max(np.abs(G_inf[ok] - s["g_stat"][ok]))) if ok.any() else math.nan
        per_seed.append((s, drift, zero_before, disc))
    pooled_g = np.concatenate([s["g_stat"][np.isfinite(s["g_stat"])] for s, *_ in per_seed])
    n_tested = int(pooled_g.size)
    if n_tested == 0:
        raise RuntimeError("no tested mechanism-episode: cannot calibrate lambda")
    max_disc = float(np.nanmax([d for *_, d in per_seed]))
    # The N_min guard, a truncated buffer or float32 storage all leave |discrepancy| small;
    # anything above 1e-2 means the reconstruction does not match mech.py and the resim
    # estimate is dropped (the pooled rule alone then decides).
    resim_ok = math.isfinite(max_disc) and max_disc < 1e-2

    candidates = np.unique(np.concatenate([[0.0], pooled_g]))
    chosen, table = None, []
    for lam in candidates:
        pooled_rate = float(np.mean(pooled_g > lam))
        resim_rate = math.nan
        if resim_ok:
            fires = sum(resimulate_page_hinkley(s["glr_stat"], dr, zb, lam)[1]
                        for s, dr, zb, _ in per_seed)
            resim_rate = fires / n_tested
        ok = pooled_rate <= target and (not resim_ok or resim_rate <= target)
        if ok:
            chosen = (float(lam), pooled_rate, resim_rate)
            break
    if chosen is None:                   # cannot happen: at lam = max(g) nothing exceeds
        lam = float(pooled_g.max())
        chosen = (lam, 0.0, 0.0)
    lam_star, pooled_rate, resim_rate = chosen
    # A coarse curve of both estimates for the record (exploratory).
    for q in (50, 90, 95, 99, 99.9):
        lam = float(np.percentile(pooled_g, q))
        row = {"quantile": q, "lambda": lam, "pooled_rate": float(np.mean(pooled_g > lam))}
        if resim_ok:
            row["resim_rate"] = sum(resimulate_page_hinkley(s["glr_stat"], dr, zb, lam)[1]
                                    for s, dr, zb, _ in per_seed) / n_tested
        table.append(row)
    return {
        "lambda": lam_star,
        "target_false_reset_rate": target,
        "pooled_exceedance_rate_at_lambda": pooled_rate,
        "resimulated_false_reset_rate_at_lambda": resim_rate,
        "resimulation_used": resim_ok,
        "resimulation_max_abs_discrepancy_vs_stored_g": max_disc,
        "n_tested_mechanism_episodes": n_tested,
        "n_seeds": len(per_seed),
        "rate_curve": table,
        "method": ("smallest stored g value at which both mean(g > lambda) over the never-reset "
                   "no-shift run and the offline Page-Hinkley re-simulation from GLR_t give "
                   "<= target false resets per tested mechanism-episode"),
    }


# =========================================================================== W, gamma, tau, ROC

def choose_grid(agent: str, R: _Runner) -> dict:
    """The grid value minimizing mean nMSE over validation seeds and episodes."""
    key, values = GRIDS[agent]
    table = []
    for v in values:
        summ = R.get(agent, {key: v})
        table.append({key: v, "nmse": float(np.mean([summ[s]["nmse_mean"] for s in summ]))})
    best = min(table, key=lambda row: row["nmse"])
    return {"key": key, "best": best[key], "grid": table}


def choose_tau(summaries: dict, pct: float = TAU_PERCENTILE) -> float:
    """90th percentile of raw width90 over all stable-episode predictions of all seeds."""
    w = np.concatenate([summaries[s]["width90_raw_stable"] for s in sorted(summaries)])
    w = w[np.isfinite(w)]
    return float(np.percentile(w, pct)) if w.size else math.inf


def detection_roc(R: _Runner, lam_star: float) -> list[dict]:
    """Recall on qualifying shifts vs false-reset rate at each multiplier of lambda*."""
    rows = []
    for mult in LAMBDA_MULTIPLIERS:
        summ = R.get("mech-full", {"lambda": mult * lam_star})
        cr = [summ[s]["credit"] for s in sorted(summ) if "credit" in summ[s]]
        def _mean(key):
            vals = [c[key] for c in cr if c[key] is not None and math.isfinite(c[key])]
            return float(np.mean(vals)) if vals else math.nan
        rows.append({"multiplier": mult, "lambda": mult * lam_star,
                     "recall_window": _mean("recall_window"), "recall_any": _mean("recall_any"),
                     "accuracy_delay0": _mean("accuracy_delay0"),
                     "false_reset_rate": _mean("false_reset_rate"),
                     "misattribution_rate": _mean("misattribution_rate"),
                     "n_shifts": int(sum(c["n_shifts"] for c in cr)),
                     "n_resets": int(sum(c["n_resets"] for c in cr))})
    return rows


# =========================================================================== main

def calibrate(seeds, out_path, episodes=60, budget=50, n_obs=200, n_queries=200, workers=4,
              log=print, fixed_lambda=None) -> dict:
    """Tune every agent's constants on validation seeds at the given budget.

    `fixed_lambda`: the SPEC wants ONE detector threshold for every detector agent, chosen at
    B = 50.  Per-budget calibrations (needed so the baselines' W and every agent's tau are
    tuned at the budget they are evaluated at; leakage audit finding) pass the B = 50 value
    here and skip the no-shift lambda run."""
    seeds = list(seeds)
    bad = [s for s in seeds if s < runner.MIN_VALIDATION_SEED]
    if bad:
        raise ValueError(f"calibrate.py runs validation seeds (>= {runner.MIN_VALIDATION_SEED}) "
                         f"only; got {bad}")
    R = _Runner(seeds, episodes, budget, n_obs, n_queries, workers, log)
    t_start = time.perf_counter()

    # ---- batch 1: the W / gamma grids and the no-shift lambda run (independent of each other)
    requests = [(agent, {key: v}, True) for agent, (key, vals) in GRIDS.items() for v in vals]
    if fixed_lambda is None:
        requests.append(("mech-full", {"lambda": math.inf}, False))
    R.run(requests)
    tuned = {agent: choose_grid(agent, R) for agent in GRIDS}
    if fixed_lambda is None:
        lam_info = choose_lambda(R.get("mech-full", {"lambda": math.inf}, shifts_enabled=False))
    else:
        lam_info = {"lambda": float(fixed_lambda),
                    "method": "fixed: reused from the B = 50 calibration (SPEC: one lambda "
                              "for every detector agent)",
                    "pooled_exceedance_rate_at_lambda": math.nan,
                    "resimulated_false_reset_rate_at_lambda": math.nan}
    lam_star = lam_info["lambda"]
    log(f"calibrate: lambda* = {lam_star:.4g} "
        f"(pooled {lam_info['pooled_exceedance_rate_at_lambda']:.4f}, "
        f"resim {lam_info['resimulated_false_reset_rate_at_lambda']:.4f}); "
        + ", ".join(f"{a} {t['key']}={t['best']}" for a, t in tuned.items()))

    # ---- batch 2: every agent at its tuned constants (for tau) and the lambda ROC
    def final_constants(agent):
        c = {}
        if agent in tuned:
            c[tuned[agent]["key"]] = tuned[agent]["best"]
        if agent in DETECTOR_AGENTS:
            c["lambda"] = lam_star
        return c
    tau_agents = [a for a in MAIN_AGENTS if a not in NEVER_ABSTAIN]
    requests = [(a, final_constants(a), True) for a in tau_agents]
    requests += [("mech-full", {"lambda": m * lam_star}, True) for m in LAMBDA_MULTIPLIERS]
    R.run(requests)

    agents_out = {}
    for a in MAIN_AGENTS:
        entry = {}
        if a in tuned:
            entry[tuned[a]["key"]] = tuned[a]["best"]
        entry["tau"] = None if a in NEVER_ABSTAIN else choose_tau(R.get(a, final_constants(a)))
        agents_out[a] = entry
    agents_out["mech-full-diag"] = {"tau": agents_out["mech-full"]["tau"]}

    roc = detection_roc(R, lam_star)
    at_star = next(r for r in roc if r["multiplier"] == 1)
    lam_info.update({"false_reset_rate_at_lambda": at_star["false_reset_rate"],
                     "recall_at_lambda": at_star["recall_window"],
                     "grid_multipliers": list(LAMBDA_MULTIPLIERS), "roc": roc})

    constants = {
        "spec_version": SPEC_VERSION,
        "validation_seeds": seeds,
        "created_utc": runner.utc_now(),
        "lambda": lam_star,
        "lambda_details": lam_info,
        "agents": agents_out,
        "tuning": {a: t for a, t in tuned.items()},
        "tau_percentile": TAU_PERCENTILE,
        "run_parameters": {"episodes": episodes, "budget": budget, "n_obs": n_obs,
                           "n_queries": n_queries, "shift_type": "large", "hetero": 0, "hidden": False},
        "notes": ["mech-full-diag is hidden-only and was not run on validation; its tau is "
                  "copied from mech-full.",
                  "marginal-mean and oracle never abstain (tau = null).",
                  "lambda chosen from a no-shift mech-full run at lambda = inf; see "
                  "lambda_details.method.  The offline Page-Hinkley re-simulation ignores that "
                  "a real fire also truncates the buffer, so false_reset_rate_at_lambda (from "
                  "the with-shifts run at lambda*) is the empirical check."],
        "git_hash": runner.git_hash(),
        "total_core_seconds": R.total_wall,
        "wall_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_jsonable(constants), f, indent=2)
    log(f"calibrate: wrote {out_path} ({constants['wall_seconds']:.1f}s wall, "
        f"{R.total_wall:.0f} core-seconds)")
    return constants


def _jsonable(x):
    """numpy -> plain Python; NaN / inf -> None (strict JSON)."""
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
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="constants.json")
    p.add_argument("--seeds", default=DEFAULT_VALIDATION_SEEDS)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fixed-lambda", type=float, default=None,
                   help="reuse this detector threshold instead of calibrating one (per-budget "
                        "calibrations pass the B = 50 value)")
    p.add_argument("--smoke", action="store_true",
                   help="first seed only and 10 episodes: exercises the whole code path fast")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = runner.parse_seeds(args.seeds)
    episodes = args.episodes
    if args.smoke:
        seeds, episodes = seeds[:1], min(episodes, 10)
    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, **k, flush=True))
    try:
        calibrate(seeds, args.out, episodes=episodes, budget=args.budget, n_obs=args.n_obs,
                  n_queries=args.n_queries, workers=args.workers, log=log,
                  fixed_lambda=args.fixed_lambda)
    except ValueError as e:
        print(f"calibrate.py: refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
