"""Command-line runner: (agent, seed) jobs in a process pool, raw files plus config.json
(SPEC.md "Deliverables", "Compute budget", "Tuning and calibration"; INTERFACES.md section 7).

    python run.py --agents main --seeds 0-19 --constants constants.json --out results/main
    python run.py --agents sweep --budget 10 --seeds 0-19 --constants constants.json --out results/sweep_B10
    python run.py --agents hidden --hidden --seeds 0-39 --constants constants.json --out results/hidden
    python run.py --agents knobs --shift-type small --hetero 1 --seeds 0-19 --constants ... --out ...
    python run.py --agents mech-full,obs-window --smoke --out /tmp/smoke      # 1 seed x 3 episodes

One job = one `(agent, seed)` pair: a fresh `World`, a fresh agent, `protocol.run_agent`, and
`protocol.save_raw` into `<out>/raw/<agent>__seed<s>.npz`.  Nothing is shared between jobs, so
the pool cannot break the paired design: every agent sees byte-identical draws because the
world's streams are keyed by `(seed, t)`, not by who asks (SPEC "RNG streams").

Refusal rule (SPEC "Tuning and calibration"; INTERFACES section 5): evaluation seeds (`< 1000`)
never run without a parsed, schema-valid `constants.json`; validation seeds (`>= 1000`) may,
which is how `calibrate.py` produces that file.  `config.json` records every parameter, the git
hash, the SHA-256 of `constants.json`, and the wall time of every job.

`run_job(JobSpec)` is the unit of work; `calibrate.py` calls it directly with `return_raw=True`.
"""

from __future__ import annotations

import os

# The matrices in this testbed are 6 x 6; BLAS threads only fight the process pool.  Set before
# numpy is imported so the setting takes effect in the parent and in forked workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import datetime as _dt
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import protocol  # noqa: E402
from agents import (HIDDEN_AGENTS, KNOB_AGENTS, MAIN_AGENTS, REGISTRY, SWEEP_AGENTS,  # noqa: E402
                    SWEEP_NOFLOOR_BUDGETS, agent_constants, build_agent)
from world import World  # noqa: E402

# Seeds below this are evaluation seeds and need constants.json (SPEC "Tuning and calibration").
MIN_VALIDATION_SEED = 1000
# --smoke: one seed, this many episodes (SPEC test 12).
SMOKE_EPISODES = 3
SMOKE_DEFAULT_SEED = 1000
# World refuses T below the end of the shift window; shorter runs truncate a full world (below).
MIN_WORLD_T = 55
SHIFT_TYPES = ("large", "small", "noise-only")
AGENT_GROUPS = ("all", "main", "sweep", "hidden", "knobs")


# =========================================================================== argument helpers

def parse_seeds(text: str) -> list[int]:
    """'0-19' -> [0..19]; '0,3,5' -> [0, 3, 5]; '0-2,1000-1001' -> [0, 1, 2, 1000, 1001].
    Sorted, duplicates removed; raises ValueError on anything else."""
    seeds: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi or lo < 0:
                raise ValueError(f"bad seed range {part!r}")
            seeds.update(range(lo, hi + 1))
        else:
            s = int(part)
            if s < 0:
                raise ValueError(f"negative seed {s}")
            seeds.add(s)
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    return sorted(seeds)


def resolve_agents(spec: str, hidden: bool = False, budget: int = 50) -> list[str]:
    """Agent group or comma list -> list of registry names (SPEC "Compute budget" table).

    all    -> MAIN_AGENTS, or HIDDEN_AGENTS when --hidden
    main   -> MAIN_AGENTS (13 agents)
    sweep  -> SWEEP_AGENTS plus mech-full-nofloor when budget in SWEEP_NOFLOOR_BUDGETS
    hidden -> HIDDEN_AGENTS
    knobs  -> KNOB_AGENTS
    Hidden-only agents (mech-full-diag) are refused outside the hidden variant.
    """
    spec = spec.strip()
    if spec == "all":
        names = list(HIDDEN_AGENTS if hidden else MAIN_AGENTS)
    elif spec == "main":
        names = list(MAIN_AGENTS)
    elif spec == "sweep":
        names = list(SWEEP_AGENTS)
        if budget in SWEEP_NOFLOOR_BUDGETS:
            names.append("mech-full-nofloor")
    elif spec == "hidden":
        names = list(HIDDEN_AGENTS)
    elif spec == "knobs":
        names = list(KNOB_AGENTS)
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
        if not names:
            raise ValueError(f"no agents in {spec!r}")
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown agent(s) {unknown}; known: {sorted(REGISTRY)}")
    if not hidden:
        bad = [n for n in names if REGISTRY[n].hidden_only]
        if bad:
            raise ValueError(f"{bad} only run in the hidden variant (--hidden)")
    seen: list[str] = []
    for n in names:                      # keep order, drop duplicates
        if n not in seen:
            seen.append(n)
    return seen


# =========================================================================== constants.json

def load_constants(path: str) -> tuple[dict, str]:
    """Parse constants.json; returns (dict, sha256 hex of the file bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def check_constants(consts: dict, agents: list[str]) -> None:
    """Schema check of INTERFACES section 5: a top-level numeric `lambda` and an `agents`
    object with a `tau` entry (number or null) for every requested agent.  Raises ValueError."""
    if not isinstance(consts, dict):
        raise ValueError("constants.json must be a JSON object")
    lam = consts.get("lambda")
    if not isinstance(lam, (int, float)) or isinstance(lam, bool) or math.isnan(lam):
        raise ValueError("constants.json needs a numeric top-level 'lambda'")
    per_agent = consts.get("agents")
    if not isinstance(per_agent, dict):
        raise ValueError("constants.json needs an 'agents' object")
    for name in agents:
        entry = per_agent.get(name)
        if not isinstance(entry, dict) or "tau" not in entry:
            raise ValueError(f"constants.json has no 'tau' for agent {name!r}")
        tau = entry["tau"]
        if tau is not None and (isinstance(tau, bool) or not isinstance(tau, (int, float))):
            raise ValueError(f"constants.json: 'tau' for {name!r} must be a number or null")


# =========================================================================== jobs

@dataclass
class JobSpec:
    """Everything one (agent, seed) job needs; plain data so it pickles into the pool."""
    agent: str
    seed: int
    episodes: int = 60
    budget: int = 50
    n_obs: int = 200
    n_queries: int = 200
    shift_type: str = "large"
    hetero: int = 0
    hidden: bool = False
    shifts_enabled: bool = True
    constants: dict | None = None        # the per-agent dict (agent_constants output), or None
    out_path: str | None = None          # where to save the raw .npz; None = do not save
    d: int = 6


@dataclass
class JobResult:
    agent: str
    seed: int
    wall_time_s: float = math.nan       # run_agent's own wall time (from raw["wall_time_s"])
    total_time_s: float = math.nan      # world + agent construction + run + save
    path: str | None = None
    error: str | None = None            # traceback text when the job failed
    raw: dict | None = None             # only when run_job(..., return_raw=True)
    extra: dict = field(default_factory=dict)


def make_world(spec: JobSpec) -> World:
    """A fresh World for the job.  World refuses T < 55 (its shift window ends at 55), so a
    shorter run (--smoke, tests) builds the full-length world and truncates the loop:
    every per-episode stream is a pure function of (seed, t) and the schedule is pre-drawn,
    so the episodes that do run are identical to the first `episodes` of the full run."""
    T_world = max(int(spec.episodes), MIN_WORLD_T)
    world = World(int(spec.seed), d=spec.d, T=T_world, B=int(spec.budget), n_obs=int(spec.n_obs),
                  shift_type=spec.shift_type, hetero=int(spec.hetero), hidden=bool(spec.hidden),
                  n_queries=int(spec.n_queries), shifts_enabled=bool(spec.shifts_enabled))
    if int(spec.episodes) < T_world:
        world.T = int(spec.episodes)     # truncation: protocol.run_agent loops to world.T
    return world


def run_job(spec: JobSpec, return_raw: bool = False) -> JobResult:
    """World -> agent -> protocol.run_agent -> save_raw.  Never raises: failures are returned
    in `error` so one bad job does not take the whole pool down."""
    t0 = time.perf_counter()
    res = JobResult(agent=spec.agent, seed=int(spec.seed), path=spec.out_path)
    try:
        world = make_world(spec)
        agent = build_agent(spec.agent, world, int(spec.seed), spec.constants)
        raw = protocol.run_agent(world, agent)
        if spec.out_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(spec.out_path)), exist_ok=True)
            protocol.save_raw(spec.out_path, raw)
        res.wall_time_s = float(raw["wall_time_s"])
        if return_raw:
            res.raw = raw
    except Exception:                    # noqa: BLE001 - reported, not swallowed
        res.error = traceback.format_exc()
    res.total_time_s = time.perf_counter() - t0
    return res


def _run_job_raw(spec: JobSpec) -> JobResult:
    return run_job(spec, return_raw=True)


def run_jobs(specs: list[JobSpec], workers: int = 4, return_raw: bool = False,
             worker_fn=None, progress=None) -> list[JobResult]:
    """Run the jobs in a multiprocessing.Pool(workers) (in-process when workers <= 1).

    `worker_fn(spec) -> result` replaces run_job (calibrate.py uses it to reduce each raw
    inside the worker); `progress(done, total, result)` is called after every job.  Results
    come back in the order of `specs`.
    """
    fn = worker_fn or (_run_job_raw if return_raw else run_job)
    results: list = [None] * len(specs)
    total = len(specs)

    def _record(idx, r):
        results[idx] = r
        if progress is not None:
            progress(sum(x is not None for x in results), total, r)

    if workers <= 1 or total <= 1:
        for idx, spec in enumerate(specs):
            _record(idx, fn(spec))
        return results
    with multiprocessing.Pool(processes=int(workers)) as pool:
        for idx, r in pool.imap_unordered(_indexed(fn), list(enumerate(specs))):
            _record(idx, r)
    return results


class _indexed:
    """Picklable wrapper that carries the job index through imap_unordered."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        idx, spec = item
        return idx, self.fn(spec)


# =========================================================================== config.json

def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:                    # noqa: BLE001 - git may be absent; tolerated
        return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================================================================== CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", default="main",
                   help="all | main | sweep | hidden | knobs | comma list of registry names")
    p.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 0,3,5 or 0-2,1000-1009")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--budget", type=int, default=50, help="interventional budget B per episode")
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--shift-type", default="large", choices=SHIFT_TYPES)
    p.add_argument("--hetero", type=int, default=0, choices=(0, 1), help="gamma_h knob")
    p.add_argument("--hidden", action="store_true", help="hidden-confounder variant")
    p.add_argument("--no-shifts", action="store_true",
                   help="disable the shift schedule (lambda calibration runs only)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--constants", default=None, help="constants.json from calibrate.py")
    p.add_argument("--out", required=True, help="results directory, e.g. results/main")
    p.add_argument("--smoke", action="store_true",
                   help=f"1 seed x {SMOKE_EPISODES} episodes (first requested seed, or "
                        f"{SMOKE_DEFAULT_SEED} when --seeds is not given)")
    p.add_argument("--quiet", action="store_true")
    return p


def plan_from_args(args) -> tuple[list[JobSpec], dict]:
    """Turn parsed CLI args into job specs plus the config dict.  Applies the refusal rule
    BEFORE any world is constructed.  Raises ValueError / FileNotFoundError on refusal."""
    seeds = parse_seeds(args.seeds)
    episodes = int(args.episodes)
    if args.smoke:
        seeds = [seeds[0]] if args.seeds_given else [SMOKE_DEFAULT_SEED]
        episodes = SMOKE_EPISODES
    agents = resolve_agents(args.agents, hidden=args.hidden, budget=int(args.budget))

    consts, sha = None, None
    if args.constants is not None:
        consts, sha = load_constants(args.constants)
        check_constants(consts, agents)
    eval_seeds = [s for s in seeds if s < MIN_VALIDATION_SEED]
    if eval_seeds and consts is None:
        raise ValueError(f"seeds {eval_seeds[:5]}{'...' if len(eval_seeds) > 5 else ''} are "
                         f"evaluation seeds (< {MIN_VALIDATION_SEED}); pass --constants "
                         "constants.json (SPEC 'Tuning and calibration')")

    raw_dir = os.path.join(args.out, "raw")
    specs = [JobSpec(agent=a, seed=s, episodes=episodes, budget=int(args.budget),
                     n_obs=int(args.n_obs), n_queries=int(args.n_queries),
                     shift_type=args.shift_type, hetero=int(args.hetero), hidden=bool(args.hidden),
                     shifts_enabled=not args.no_shifts,
                     constants=agent_constants(consts, a),
                     out_path=os.path.join(raw_dir, protocol.raw_filename(a, s)))
             for a in agents for s in seeds]
    config = {
        "run_name": os.path.basename(os.path.normpath(args.out)),
        "agents": agents, "seeds": seeds, "episodes": episodes, "budget": int(args.budget),
        "n_obs": int(args.n_obs), "n_queries": int(args.n_queries), "shift_type": args.shift_type,
        "hetero": int(args.hetero), "hidden": bool(args.hidden),
        "shifts_enabled": not args.no_shifts, "workers": int(args.workers), "smoke": bool(args.smoke),
        "d": 6,
        "constants_path": args.constants, "constants_sha256": sha, "constants": consts,
        "git_hash": git_hash(), "python": platform.python_version(), "numpy": np.__version__,
        "n_jobs": len(specs),
    }
    return specs, config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.seeds_given = any(a == "--seeds" or a.startswith("--seeds=")
                           for a in (argv if argv is not None else sys.argv[1:]))
    try:
        specs, config = plan_from_args(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"run.py: refused: {e}", file=sys.stderr)
        return 2

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    config["started_utc"] = utc_now()
    t0 = time.perf_counter()
    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, **k, flush=True))
    log(f"run.py: {len(specs)} jobs ({len(config['agents'])} agents x {len(config['seeds'])} seeds, "
        f"T={config['episodes']}, B={config['budget']}) on {args.workers} workers -> {args.out}")

    def progress(done, total, r: JobResult):
        status = "FAILED" if r.error else f"{r.wall_time_s:6.2f}s"
        log(f"  [{done:4d}/{total}] {r.agent:>20s} seed {r.seed:<5d} {status}")

    results = run_jobs(specs, workers=int(args.workers), progress=progress)

    config["finished_utc"] = utc_now()
    config["total_wall_s"] = time.perf_counter() - t0
    config["wall_time_s"] = {f"{r.agent}__seed{r.seed}": (None if r.error else r.wall_time_s)
                             for r in results}
    config["job_time_s"] = {f"{r.agent}__seed{r.seed}": r.total_time_s for r in results}
    errors = {f"{r.agent}__seed{r.seed}": r.error for r in results if r.error}
    config["errors"] = errors
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    ok = sum(1 for r in results if not r.error)
    log(f"run.py: {ok}/{len(results)} jobs ok in {config['total_wall_s']:.1f}s; "
        f"config -> {os.path.join(args.out, 'config.json')}")
    if errors:
        for k, tb in errors.items():
            print(f"--- {k} failed:\n{tb}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
