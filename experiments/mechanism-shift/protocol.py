"""The episode loop, implemented once for every agent (SPEC.md "Episode protocol" steps 1-8;
INTERFACES.md section 4).

`run_agent(world, agent)` runs the T episodes and returns the raw result dictionary whose
exact schema is INTERFACES.md section 6; `save_raw` writes it as `.npz`.

Responsibilities that live HERE and nowhere else:
  * the order of the eight steps and of the agent hooks inside them,
  * the round-robin floor (shared `ValueCycle` on the agent) and the warm-up rule,
  * the self-log-score-before-reveal ordering of every interventional sample,
  * the evaluator: ground truth, normalized error, widths, coverage and abstention.

Agents never see truth, Var_ref, the schedule or the permutation; this module only reads
those from the World and writes them to the raw file.  This module does not import `world`.
"""

from __future__ import annotations

import math
import time

import numpy as np

from agents.base import ALLOWED_VALUES, QUANTILE_LEVELS, WARMUP_EPISODES

# Codes for the raw file's shift_type_code field (-1 = no shift).
SHIFT_TYPE_CODES = {"large": 0, "small": 1, "noise-only": 2}


# --------------------------------------------------------------------------- small helpers

def floor_size(B: int, d: int) -> tuple[int, int, int]:
    """(F, n_floor, n_free) with F = ceil(B / (2d)) samples per variable (SPEC step 6).

    B = 10 -> (1, 6, 4); 25 -> (3, 18, 7); 50 -> (5, 30, 20); 100 -> (9, 54, 46) at d = 6.
    """
    if B < d:
        raise ValueError(f"budget B={B} smaller than d={d}: the floor cannot be satisfied")
    F = math.ceil(B / (2 * d))
    n_floor = d * F
    return F, n_floor, B - n_floor


def raw_filename(agent_name: str, seed: int) -> str:
    return f"{agent_name}__seed{int(seed)}.npz"


def save_raw(path, raw: dict) -> None:
    np.savez_compressed(path, **raw)


def _check_indices(items, d: int, hook: str) -> list[int]:
    out = []
    for j in items:
        if int(j) != j or not (0 <= int(j) < d):
            raise ValueError(f"{hook} returned invalid mechanism index {j!r}")
        out.append(int(j))
    return out


def _check_plan(plan, n_free: int, d: int) -> list[tuple[int, float]]:
    plan = list(plan)
    if len(plan) != n_free:
        raise ValueError(f"choose_free_interventions returned {len(plan)} pairs, "
                         f"expected {n_free}")
    out = []
    for i, v in plan:
        if int(i) != i or not (0 <= int(i) < d):
            raise ValueError(f"invalid intervention target {i!r}")
        if float(v) not in ALLOWED_VALUES:
            raise ValueError(f"invalid intervention value {v!r}; allowed {sorted(ALLOWED_VALUES)}")
        out.append((int(i), float(v)))
    return out


def select_off_target(arr: np.ndarray, query_i: np.ndarray) -> np.ndarray:
    """Drop the j == i entry of every query: (Q, d[, ...]) -> (Q, d-1[, ...]).

    Column c of the result is variable j = c if c < i else c + 1 (ascending j; see
    INTERFACES.md section 0).  Boolean indexing flattens in row-major order, so each row keeps
    its d-1 surviving columns in ascending order.
    """
    Q, d = arr.shape[0], arr.shape[1]
    mask = np.ones((Q, d), dtype=bool)
    mask[np.arange(Q), query_i] = False
    return arr[mask].reshape((Q, d - 1) + arr.shape[2:])


# --------------------------------------------------------------------------- evaluator

def evaluate_answer(ans: dict, query_i: np.ndarray, truth: np.ndarray, sd_ref: np.ndarray,
                    tau: float) -> dict:
    """Score one episode's answers against exact truth (INTERFACES.md section 4.3).

    ans["point"]: (Q, d); ans["quantiles"]: (Q, d, 6) at QUANTILE_LEVELS.  truth: (Q, d) exact
    interventional means; sd_ref: (d,) sqrt(Var_ref).  Returns (Q, d-1) arrays.
    """
    Q, d = truth.shape
    point = np.asarray(ans["point"], dtype=np.float64)
    quant = np.asarray(ans["quantiles"], dtype=np.float64)
    if point.shape != (Q, d):
        raise ValueError(f"answer['point'] has shape {point.shape}, expected {(Q, d)}")
    if quant.shape != (Q, d, len(QUANTILE_LEVELS)):
        raise ValueError(f"answer['quantiles'] has shape {quant.shape}, "
                         f"expected {(Q, d, len(QUANTILE_LEVELS))}")

    p = select_off_target(point, query_i)              # (Q, d-1)
    q = select_off_target(quant, query_i)              # (Q, d-1, 6)
    y = select_off_target(truth, query_i)              # (Q, d-1)
    s = select_off_target(np.broadcast_to(sd_ref, (Q, d)), query_i)

    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("answer contains non-finite values at some j != i")
    if np.any(q[..., 1:] < q[..., :-1] - 1e-9):
        raise ValueError("answer quantiles are not non-decreasing along the last axis")

    width90_raw = q[..., 4] - q[..., 1]                # q0.95 - q0.05: the agent's own score
    return {
        "err": (p - y) / s,                            # signed normalized error; nMSE = err²
        "width50": (q[..., 3] - q[..., 2]) / s,
        "width90": width90_raw / s,
        "width99": (q[..., 5] - q[..., 0]) / s,
        "width90_raw": width90_raw,
        "cov50": (q[..., 2] <= y) & (y <= q[..., 3]),
        "cov90": (q[..., 1] <= y) & (y <= q[..., 4]),
        "cov99": (q[..., 0] <= y) & (y <= q[..., 5]),
        "abstain": width90_raw > tau,                  # SPEC: abstain = (q0.95 - q0.05 > tau)
    }


# --------------------------------------------------------------------------- the loop

def run_agent(world, agent) -> dict:
    """Run `agent` through all T episodes of `world`; return the raw result dictionary."""
    d, T, B, Q = int(world.d), int(world.T), int(world.B), int(world.n_queries)
    if agent.d != d:
        raise ValueError(f"agent.d={agent.d} != world.d={d}")
    if agent.intervenes:
        F, n_floor, _ = floor_size(B, d)

    nan = np.nan
    f32 = np.float32
    R = {
        # per-episode world state
        "episode_t": np.arange(1, T + 1, dtype=np.int32),
        "shift_j": np.full(T, -1, dtype=np.int32),
        "shift_type_code": np.full(T, -1, dtype=np.int8),
        "shift_m": np.full(T, nan, dtype=f32),
        "shift_delta": np.full(T, nan, dtype=f32),
        "shift_delta_sigma": np.full(T, nan, dtype=f32),
        "true_adj": np.zeros((T, d, d), dtype=bool),
        "true_W": np.zeros((T, d, d), dtype=f32),
        "true_b": np.zeros((T, d), dtype=f32),
        "true_sigma": np.zeros((T, d), dtype=f32),
        "sd_ref": np.zeros((T, d), dtype=f32),
        "F_obs": np.full(T, nan, dtype=f32),
        "conf_c": np.full(T, nan, dtype=f32),
        "conf_F": np.full(T, nan, dtype=f32),
        # per-prediction
        "err": np.zeros((T, Q, d - 1), dtype=f32),
        "width50": np.zeros((T, Q, d - 1), dtype=f32),
        "width90": np.zeros((T, Q, d - 1), dtype=f32),
        "width99": np.zeros((T, Q, d - 1), dtype=f32),
        "width90_raw": np.zeros((T, Q, d - 1), dtype=f32),
        "cov50": np.zeros((T, Q, d - 1), dtype=bool),
        "cov90": np.zeros((T, Q, d - 1), dtype=bool),
        "cov99": np.zeros((T, Q, d - 1), dtype=bool),
        "abstain": np.zeros((T, Q, d - 1), dtype=bool),
        # per-query
        "query_i": np.zeros((T, Q), dtype=np.int32),
        "query_v": np.zeros((T, Q), dtype=f32),
        # per-sample interventions
        "intervention_i": np.full((T, B), -1, dtype=np.int32),
        "intervention_v": np.full((T, B), nan, dtype=f32),
        "sample_score": np.full((T, B), nan, dtype=f32),
        # per-episode agent state
        "learned_adj": np.zeros((T, d, d), dtype=bool),
        "has_learned_adj": np.zeros(T, dtype=bool),
        "g_stat": np.full((T, d), nan, dtype=f32),
        "glr_stat": np.full((T, d), nan, dtype=f32),
        "self_score": np.full(T, nan, dtype=f32),
        "var_obj_before": np.full(T, nan, dtype=f32),
        "var_obj_after": np.full(T, nan, dtype=f32),
    }
    reset_events: list[tuple[int, int]] = []
    struct_refit_events: list[tuple[int, int]] = []

    t_start = time.perf_counter()
    for t in range(1, T + 1):
        r = t - 1

        # ---- step 1: the world applies the pre-drawn shift; the SCM is fixed for the episode
        rec = world.apply_shift(t)
        if rec is not None:
            R["shift_j"][r] = int(rec.j)
            R["shift_type_code"][r] = SHIFT_TYPE_CODES[rec.shift_type]
            R["shift_m"][r] = rec.m
            R["shift_delta"][r] = rec.delta
            R["shift_delta_sigma"][r] = rec.delta_sigma
        R["true_adj"][r] = world.true_adjacency()
        R["true_W"][r] = world.true_W()
        R["true_b"][r] = world.true_b()
        R["true_sigma"][r] = world.true_sigma()
        sd_ref = np.asarray(world.sigma_ref(), dtype=np.float64)
        R["sd_ref"][r] = sd_ref
        R["F_obs"][r] = world.F_obs()
        conf = world.confounding()
        if conf is not None:
            R["conf_c"][r] = conf["c"]
            R["conf_F"][r] = conf["F_conf"]

        # ---- step 2: keyed draws (identical for every agent)
        X_obs = world.observational_batch(t)
        query_i, query_v = world.query_set(t)
        query_i = np.asarray(query_i, dtype=np.int64)
        query_v = np.asarray(query_v, dtype=np.float64)
        R["query_i"][r] = query_i
        R["query_v"][r] = query_v

        # ---- step 3: the agent receives the observational batch
        agent.observe(X_obs, t)

        # ---- step 4: detection and the reset rule
        for j in _check_indices(agent.detect_and_reset(), d, "detect_and_reset"):
            reset_events.append((t, j))
        stats = agent.detector_stats()
        if stats is not None:
            R["g_stat"][r] = np.asarray(stats["g"], dtype=np.float64)
            R["glr_stat"][r] = np.asarray(stats["glr"], dtype=np.float64)

        # ---- step 5: append the batch to every buffer and refit all posteriors
        agent.refit()

        # ---- step 6: interventions, round 1 (floor) then round 2 (free budget)
        if agent.intervenes:
            k = 0

            def sample(i: int, v: float) -> None:
                nonlocal k
                row = world.intervene(t, k, i, v)          # noise row k whatever (i, v) is
                s = agent.score_before_reveal(i, v, row)   # self log-score BEFORE reveal
                agent.receive_intervention(i, v, row)
                R["intervention_i"][r, k] = i
                R["intervention_v"][r, k] = v
                if s is not None:
                    R["sample_score"][r, k] = float(s)
                k += 1

            if t <= WARMUP_EPISODES:
                n_rr = B                   # warm-up: the whole budget is round-robin
            elif agent.uses_floor:
                n_rr = n_floor             # F samples per variable
            else:
                n_rr = 0                   # mech-full-nofloor
            for _ in range(n_rr):
                i, v = agent.cycle.round_robin()
                sample(i, v)
            n_free = B - n_rr
            if n_free > 0:
                plan = _check_plan(agent.choose_free_interventions(n_free, t), n_free, d)
                for i, v in plan:
                    sample(i, v)
            assert k == B

        # ---- step 7: structure re-estimation
        for j in _check_indices(agent.update_structure(t), d, "update_structure"):
            struct_refit_events.append((t, j))

        # ---- step 8: answer the queries; the evaluator scores them against exact truth
        # Truth is computed from the protocol's own copies BEFORE the agent sees the queries,
        # and the agent receives read-only copies: an agent cannot change what it is graded on
        # (leakage audit finding).
        truth = np.stack([np.asarray(world.truth(int(i), float(v)), dtype=np.float64)
                          for i, v in zip(query_i, query_v)])
        qi_agent = query_i.copy()
        qv_agent = query_v.copy()
        qi_agent.setflags(write=False)
        qv_agent.setflags(write=False)
        ans = agent.answer(qi_agent, qv_agent)
        ev = evaluate_answer(ans, query_i, truth, sd_ref, agent.tau)
        for key, val in ev.items():
            R[key][r] = val
        adj = agent.learned_adjacency()
        if adj is not None:
            R["learned_adj"][r] = np.asarray(adj, dtype=bool)
            R["has_learned_adj"][r] = True
        obj = agent.internal_objectives()
        if obj is not None:
            R["self_score"][r] = obj.get("self_score", nan)
            R["var_obj_before"][r] = obj.get("var_obj_before", nan)
            R["var_obj_after"][r] = obj.get("var_obj_after", nan)

    wall = time.perf_counter() - t_start

    R["reset_events"] = np.asarray(reset_events, dtype=np.int32).reshape(-1, 2)
    R["struct_refit_events"] = np.asarray(struct_refit_events, dtype=np.int32).reshape(-1, 2)

    pair = world.hidden_pair()
    R.update({
        "agent_name": np.str_(agent.name),
        "agent_id": np.int32(agent.agent_id),
        "seed": np.int32(world.seed),
        "d": np.int32(d), "T": np.int32(T), "B": np.int32(B),
        "n_obs": np.int32(world.n_obs), "Q": np.int32(Q),
        "shift_type": np.str_(world.shift_type),
        "hetero": np.int32(world.hetero),
        "hidden": np.bool_(world.hidden),
        "shifts_enabled": np.bool_(getattr(world, "shifts_enabled", True)),
        "learns_structure": np.bool_(agent.learns_structure),
        "has_detector": np.bool_(agent.has_detector),
        "intervenes": np.bool_(agent.intervenes),
        "uses_floor": np.bool_(agent.uses_floor),
        "is_oracle": np.bool_(agent.is_oracle),
        "tau": np.float32(agent.tau),
        "wall_time_s": np.float32(wall),
        "hidden_A": np.int32(-1 if pair is None else pair[0]),
        "hidden_B": np.int32(-1 if pair is None else pair[1]),
    })
    return R
