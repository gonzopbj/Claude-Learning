"""Integration tests that need the real World AND the real agents together
(SPEC.md "Tests" 3 and 12).

  3.  Leakage, third clause: running `mech-full` on a relabelled copy of the same world yields
      identical predictions up to relabelling.  Clauses one and two (WorldView exposes nothing,
      agents/ never imports world) live in tests/test_world.py; the agent-level version of the
      third clause on a fake world lives in tests/test_mech.py.  This file runs the real thing
      through `protocol.run_agent` on `World` versus `RelabelledWorld`.
  12. Smoke: one `mech-full` seed for 3 episodes completes in under 2 s; the full 13-agent set
      on one seed for 3 episodes raises no exception at B = 10 (small-n guards).

Two facts found while writing test 3 are recorded here because they matter for reading it:

* The protocol's round-robin floor (SPEC step 6, round 1) walks variables in LABEL order and
  pairs noise row k with sample k whatever (i, v) is.  On a relabelled world, sample k therefore
  clamps a different underlying variable with the same noise row, so the interventional data
  of the two runs are genuinely different and nothing downstream can be identical.  That is a
  property of the protocol, not a leak in the agent.  The test aligns the floor by giving the
  relabelled agent a cycle that walks perm[0], perm[1], ... (the same underlying variables in
  the same order); the agent's own choices (free budget, values, structure, detector) are then
  the object under test.
* `MechAgent.draw_joint` pushes label-ordered standard-normal draws through a per-mechanism
  Cholesky factor, and Cholesky is not permutation-equivariant, so the S = 1000 Monte Carlo
  answers of the two runs are equal in distribution but not bit-identical.  Everything analytic
  is asserted exactly; the Monte Carlo point predictions and widths are asserted to within
  Monte Carlo error (measured max |d err| about 0.01 sd_ref over 60 episodes on four seeds;
  the bounds below are three times that).
"""

from __future__ import annotations

import time

import numpy as np

import protocol
import run
from agents import MAIN_AGENTS, build_agent
from agents.base import ValueCycle
from world import RelabelledWorld, World


# --------------------------------------------------------------------------- helpers

class PermutedCycle(ValueCycle):
    """Round-robin in the order perm[0], perm[1], ...: on the relabelled world this clamps the
    same underlying variables, in the same order, as the plain cycle on the original world.
    Per-variable value positions are still indexed by (new) label, so values agree too."""

    def __init__(self, d: int, perm) -> None:
        super().__init__(d)
        self.perm = np.asarray(perm, dtype=np.int64)

    def round_robin(self):
        i = int(self.perm[self.next_var])
        self.next_var = (self.next_var + 1) % self.d
        return i, self.next_value(i)


def to_full(arr: np.ndarray, query_i: np.ndarray, d: int) -> np.ndarray:
    """Inverse of protocol.select_off_target: (T, Q, d-1) off-target values -> (T, Q, d) with
    NaN at the clamped variable, so the two runs can be compared column by column."""
    T, Q, _ = arr.shape
    mask = np.ones((T, Q, d), dtype=bool)
    np.put_along_axis(mask, np.asarray(query_i)[:, :, None], False, axis=2)
    out = np.full((T, Q, d), np.nan)
    out[mask] = np.asarray(arr, dtype=np.float64).ravel()
    return out


def paired_runs(seed: int, T: int, perm: np.ndarray, agent_name: str = "mech-full"):
    """The same world twice, plain and relabelled by `perm` (new label of old a is perm[a]),
    each run by a fresh agent built from the same seed.  T is truncated as run.make_world does
    (World refuses T < 55; the first T episodes of the truncated world are the first T of the
    full one because every stream is keyed by (seed, t))."""
    w1 = World(seed, B=50)
    w1.T = T
    w2 = RelabelledWorld(World(seed, B=50), perm)
    w2.T = T
    a1 = build_agent(agent_name, w1, seed)
    a2 = build_agent(agent_name, w2, seed)
    # The agent's private tie-break permutation must be the SAME order in new labels
    # (the stream would otherwise give a different order relative to the underlying variables).
    a2.pi = perm[a1.pi]
    a2.cycle = PermutedCycle(w1.d, perm)          # align the protocol's floor (see module doc)
    return protocol.run_agent(w1, a1), protocol.run_agent(w2, a2)


# --------------------------------------------------------------------------- SPEC test 3 (c)

def test_mech_full_on_relabelled_world_is_identical_up_to_relabelling():
    seed, T = 7, 12
    perm = np.random.default_rng(107).permutation(6)
    assert not np.array_equal(perm, np.arange(6))
    R1, R2 = paired_runs(seed, T, perm)
    inv = np.argsort(perm)                       # new -> old

    # the world side really is the same world (guards the test, not the agent)
    assert np.array_equal(R2["query_i"], perm[R1["query_i"]])
    assert np.array_equal(R2["query_v"], R1["query_v"])
    assert np.array_equal(R2["shift_j"], np.where(R1["shift_j"] >= 0, perm[R1["shift_j"]], -1))

    # --- analytic parts: exact up to relabelling
    assert np.array_equal(R2["intervention_i"], perm[R1["intervention_i"]])   # incl. free budget
    assert np.array_equal(R2["intervention_v"], R1["intervention_v"])
    np.testing.assert_allclose(R2["sample_score"], R1["sample_score"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(R2["self_score"], R1["self_score"], rtol=1e-6, atol=1e-6)
    for key in ("g_stat", "glr_stat"):
        np.testing.assert_allclose(R2[key][:, perm], R1[key], rtol=1e-6, atol=1e-6,
                                   equal_nan=True)
    for t in range(T):
        assert np.array_equal(R2["learned_adj"][t], R1["learned_adj"][t][np.ix_(inv, inv)])
    assert np.array_equal(R2["has_learned_adj"], R1["has_learned_adj"])
    assert R1["has_learned_adj"].all() and R1["learned_adj"][-1].any()   # something was learned
    for key in ("reset_events", "struct_refit_events"):
        ev1 = set(map(tuple, R1[key].tolist()))
        ev2 = {(t, int(inv[j])) for t, j in R2[key].tolist()}
        assert ev1 == ev2, key
    assert len(set(map(tuple, R1["struct_refit_events"].tolist()))) > 0

    # --- Monte Carlo parts: equal in distribution, within MC error in value
    d = 6
    err1 = to_full(R1["err"], R1["query_i"], d)
    err2 = to_full(R2["err"], R2["query_i"], d)[:, :, perm]     # column perm[k] <- old k
    d_err = np.abs(err2 - err1)
    assert np.nanmax(d_err) < 0.03 and np.nanmean(d_err) < 0.005
    w1 = to_full(R1["width90"], R1["query_i"], d)
    w2 = to_full(R2["width90"], R2["query_i"], d)[:, :, perm]
    assert np.nanmax(np.abs(w2 - w1)) < 0.15
    for key in ("cov50", "cov90", "cov99"):
        assert abs(R2[key].mean() - R1[key].mean()) < 0.03, key
    assert np.array_equal(R2["abstain"], R1["abstain"])          # tau = inf: nobody abstains


def test_unaligned_floor_is_the_only_source_of_disagreement():
    """Without the floor alignment the two runs diverge from sample 0 of episode 1: this pins
    the module-docstring claim that the divergence is the protocol's label-ordered floor."""
    seed, T = 7, 4
    perm = np.random.default_rng(107).permutation(6)
    w1 = World(seed, B=50)
    w1.T = T
    w2 = RelabelledWorld(World(seed, B=50), perm)
    w2.T = T
    a1, a2 = build_agent("mech-full", w1, seed), build_agent("mech-full", w2, seed)
    a2.pi = perm[a1.pi]                                          # aligned pi, plain cycle
    R1, R2 = protocol.run_agent(w1, a1), protocol.run_agent(w2, a2)
    assert R2["intervention_i"][0, 0] == 0 and perm[R1["intervention_i"][0, 0]] == perm[0]
    assert not np.array_equal(R2["intervention_i"], perm[R1["intervention_i"]])


# --------------------------------------------------------------------------- SPEC test 12

def test_mech_full_three_episodes_under_two_seconds():
    spec = run.JobSpec(agent="mech-full", seed=1000, episodes=3, budget=50)
    t0 = time.perf_counter()
    res = run.run_job(spec, return_raw=True)
    elapsed = time.perf_counter() - t0
    assert res.error is None, res.error
    assert res.raw["err"].shape == (3, 200, 5)
    assert elapsed < 2.0, f"{elapsed:.2f} s"


def test_all_main_agents_three_episodes_at_B10_raise_nothing():
    assert len(MAIN_AGENTS) == 13
    for name in MAIN_AGENTS:
        res = run.run_job(run.JobSpec(agent=name, seed=1000, episodes=3, budget=10),
                          return_raw=True)
        assert res.error is None, f"{name}:\n{res.error}"
        raw = res.raw
        assert raw["err"].shape == (3, 200, 5) and int(raw["B"]) == 10
        live = ~raw["abstain"]
        assert np.isfinite(raw["err"][live]).all(), name
        assert np.isfinite(raw["width90"][live]).all(), name
        if raw["intervenes"]:
            assert (raw["intervention_i"] >= 0).all(), name
