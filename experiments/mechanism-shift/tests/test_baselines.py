"""Tests for agents/baselines.py and agents/oracle.py (SPEC.md test 11, "Baseline details",
"What the agent is asked"; INTERFACES.md sections 2-4).

Covers
  * SPEC test 11a: `obs-window` prediction equals the Gaussian conditional
    mu_j + Sigma_ji / Sigma_ii (v - mu_i) fitted on the same rows (windowed and cumulative);
  * SPEC test 11b: `int-pairwise` recovers Theta[j, i] within 3 standard errors from 2000
    clamped rows, and its slope equals the closed-form OLS slope on those rows;
  * the guards: n >= 4 rows, >= 2 distinct v, regressor variance >= 1e-8, and the fallback to
    the marginal-mean rule (for int-pairwise: the mean of its observational pool);
  * the sliding windows (obs-window and int-pairwise), the uniform-random free-budget policy
    with the per-variable value cycle, the shared bootstrap resample;
  * `marginal-mean` never abstains; `oracle` returns exact points with all quantiles equal
    to the truth;
  * integration: marginal-mean, obs-window, obs-cumulative, int-pairwise and oracle each run
    3 episodes through `protocol.run_agent` on World(seed=1000) at B = 10 and B = 50 without
    exceptions and produce a raw dict satisfying the INTERFACES section 6 schema.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import protocol
from agents import AGENT_IDS, REGISTRY, build_agent
from agents.base import ALLOWED_VALUES, QUANTILE_LEVELS, VALUE_CYCLE
from agents.baselines import (N_BOOT, IntPairwiseAgent, MarginalMeanAgent, ObsPairwiseAgent,
                              bootstrap_weights)
from agents.oracle import OracleAgent
from world import World, WorldView

D = 6
VIEW = WorldView(d=D, B=50, T=60, n_obs=200, n_queries=200)


# --------------------------------------------------------------------------- helpers

def make(cls, name, seed=0, constants=None, access=VIEW, **kwargs):
    """Construct a baseline directly (bypassing build_agent) with its registry agent_id."""
    return cls(name, AGENT_IDS[name], D, seed, constants, access, **kwargs)


def gaussian_batch(rng, n=200):
    """A correlated (n, D) batch: a small linear SCM so the columns are genuinely dependent."""
    eps = rng.standard_normal((n, D))
    X = np.empty((n, D))
    X[:, 0] = 1.0 + eps[:, 0]
    X[:, 1] = -0.5 + 1.2 * X[:, 0] + 0.7 * eps[:, 1]
    X[:, 2] = 0.3 - 0.8 * X[:, 1] + eps[:, 2]
    X[:, 3] = 0.9 * X[:, 0] + 0.6 * X[:, 2] + 1.3 * eps[:, 3]
    X[:, 4] = 0.2 + eps[:, 4]
    X[:, 5] = -1.1 * X[:, 4] + 0.5 * X[:, 3] + 0.8 * eps[:, 5]
    return X


def gaussian_conditional(X, qi, qv):
    """SPEC test 11: mu_j + Sigma_ji / Sigma_ii (v - mu_i) fitted on the rows of X, (Q, D)."""
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False, bias=True)
    out = np.empty((len(qi), X.shape[1]))
    for q, (i, v) in enumerate(zip(qi, qv)):
        out[q] = mu + Sigma[:, i] / Sigma[i, i] * (v - mu[i])
    return out


def queries(rng, Q=40):
    return rng.integers(0, D, size=Q), rng.uniform(-2.0, 2.0, size=Q)


def off_target(arr, qi):
    """Entries at j != i only (the evaluator ignores j == i)."""
    return protocol.select_off_target(arr, qi)


def check_answer_shape(ans, qi):
    """The invariants protocol.evaluate_answer enforces on every answer."""
    Q = len(qi)
    assert ans["point"].shape == (Q, D)
    assert ans["quantiles"].shape == (Q, D, len(QUANTILE_LEVELS))
    p, q = off_target(ans["point"], qi), off_target(ans["quantiles"], qi)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(q))
    assert np.all(q[..., 1:] >= q[..., :-1] - 1e-9)


# --------------------------------------------------------------------------- bootstrap helper

def test_bootstrap_weights_is_one_shared_row_resample():
    rng = np.random.default_rng(0)
    n = 37
    w = bootstrap_weights(rng, n)
    assert w.shape == (1 + N_BOOT, n)
    assert np.all(w[0] == 1.0)                       # row 0: the full pool
    assert np.all(w[1:].sum(axis=1) == n)            # every resample draws exactly n rows
    assert np.all(w >= 0) and np.all(w == np.round(w))
    # drawn from the generator handed in: same seed -> same resample, different seed -> not
    assert np.array_equal(bootstrap_weights(np.random.default_rng(0), n), w)
    assert not np.array_equal(bootstrap_weights(np.random.default_rng(1), n), w)


# --------------------------------------------------------------------------- SPEC test 11a: obs-window

def test_obs_window_equals_gaussian_conditional_on_the_same_rows():
    rng = np.random.default_rng(11)
    agent = make(ObsPairwiseAgent, "obs-window", constants={"W": 1}, windowed=True)
    assert agent.W == 1 and not agent.intervenes
    X = gaussian_batch(rng)
    agent.observe(X, 1)
    agent.refit()
    qi, qv = queries(rng)
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    expected = gaussian_conditional(X, qi, qv)
    assert np.allclose(off_target(ans["point"], qi), off_target(expected, qi), atol=1e-9)
    # the bootstrap interval brackets the point in the vast majority of cells and is not
    # degenerate (a 200-row pool leaves real sampling uncertainty)
    q = off_target(ans["quantiles"], qi)
    p = off_target(ans["point"], qi)
    assert np.mean((q[..., 1] <= p) & (p <= q[..., 4])) > 0.95
    assert np.all(q[..., 4] - q[..., 1] > 0)


def test_obs_window_uses_exactly_the_last_W_batches_and_cumulative_uses_all():
    rng = np.random.default_rng(12)
    batches = [gaussian_batch(rng) + 3.0 * t for t in range(1, 5)]   # distinct means per episode
    win = make(ObsPairwiseAgent, "obs-window", constants={"W": 2}, windowed=True)
    cum = make(ObsPairwiseAgent, "obs-cumulative", constants={"W": 2}, windowed=False)
    assert win.W == 2
    qi, qv = queries(rng)
    for t, X in enumerate(batches, start=1):
        for a in (win, cum):
            a.observe(X, t)
            a.refit()
    # window: batches 3 and 4 (the current episode counts as one of the "last W")
    assert win.pool().shape[0] == 2 * 200
    exp_win = gaussian_conditional(np.concatenate(batches[2:]), qi, qv)
    assert np.allclose(off_target(win.answer(qi, qv)["point"], qi), off_target(exp_win, qi),
                       atol=1e-9)
    # cumulative: everything, and the W constant is ignored
    assert cum.pool().shape[0] == 4 * 200
    exp_cum = gaussian_conditional(np.concatenate(batches), qi, qv)
    assert np.allclose(off_target(cum.answer(qi, qv)["point"], qi), off_target(exp_cum, qi),
                       atol=1e-9)
    # default window when no constant is given
    assert make(ObsPairwiseAgent, "obs-window", windowed=True).W == 3


# --------------------------------------------------------------------------- SPEC test 11b: int-pairwise

def _run_clamped_rows(world, agent, i, n_rows, t=1):
    """Feed agent `n_rows` rows do(X_i = v) of episode t (v cycled through VALUE_CYCLE)."""
    vals = np.empty(n_rows)
    rows = np.empty((n_rows, D))
    for k in range(n_rows):
        v = VALUE_CYCLE[k % len(VALUE_CYCLE)]
        row = world.intervene(t, k, i, v)
        agent.receive_intervention(i, v, row)
        vals[k], rows[k] = v, row
    return vals, rows


@pytest.mark.parametrize("seed,i", [(1000, 0), (1001, 3), (1002, 5)])
def test_int_pairwise_recovers_total_effect_within_3_se_from_2000_rows(seed, i):
    world = World(seed, B=2000)             # B = 2000 pre-drawn noise rows in episode 1
    world.apply_shift(1)
    agent = make(IntPairwiseAgent, "int-pairwise", seed=seed, access=world.view())
    agent.observe(world.observational_batch(1), 1)
    agent.refit()
    vals, rows = _run_clamped_rows(world, agent, i, 2000)

    # the agent's line through two query values gives its intercept and slope for do(X_i)
    ans = agent.answer(np.array([i, i]), np.array([0.0, 1.0]))
    check_answer_shape(ans, np.array([i, i]))
    b0_hat = ans["point"][0]
    b1_hat = ans["point"][1] - ans["point"][0]

    # closed-form OLS of X_j on [1, v] over the same rows, with its standard error
    A = np.column_stack([np.ones(2000), vals])
    beta, *_ = np.linalg.lstsq(A, rows, rcond=None)              # (2, D)
    resid = rows - A @ beta
    s2 = (resid ** 2).sum(axis=0) / (2000 - 2)
    se_slope = np.sqrt(s2 / ((vals - vals.mean()) ** 2).sum())

    W = world.true_W()
    Theta = np.linalg.inv(np.eye(D) - W)                          # Theta[j, i] = total effect
    for j in range(D):
        if j == i:
            continue
        assert abs(b1_hat[j] - beta[1, j]) < 1e-8 and abs(b0_hat[j] - beta[0, j]) < 1e-8
        assert abs(b1_hat[j] - Theta[j, i]) <= 3 * se_slope[j], (j, b1_hat[j], Theta[j, i])
        # and the intercept is the exact c_j(i) = truth(i, j, 0) within 3 se of the intercept
        se_int = np.sqrt(s2[j] * (1 / 2000 + vals.mean() ** 2 / ((vals - vals.mean()) ** 2).sum()))
        assert abs(b0_hat[j] - world.truth(i, 0.0)[j]) <= 3 * se_int


# --------------------------------------------------------------------------- guards and fallbacks

def test_obs_guard_fewer_than_4_rows_falls_back_to_the_marginal_mean():
    rng = np.random.default_rng(20)
    agent = make(ObsPairwiseAgent, "obs-window", windowed=True)
    X = gaussian_batch(rng, n=3)                    # n = 3 < 4: no fit anywhere
    agent.observe(X, 1)
    agent.refit()
    qi, qv = queries(rng)
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    assert np.allclose(ans["point"], np.broadcast_to(X.mean(axis=0), (len(qi), D)))


def test_obs_guard_constant_regressor_falls_back_only_for_that_target():
    rng = np.random.default_rng(21)
    agent = make(ObsPairwiseAgent, "obs-cumulative", windowed=False)
    X = gaussian_batch(rng)
    X[:, 2] = 0.75                                  # var(X_2) = 0 < 1e-8
    agent.observe(X, 1)
    agent.refit()
    qi = np.array([2, 2, 0, 4])
    qv = np.array([-1.5, 2.0, 1.0, -0.3])
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    mu = X.mean(axis=0)
    assert np.allclose(ans["point"][:2], np.broadcast_to(mu, (2, D)))      # do(X_2): marginal mean
    expected = gaussian_conditional(np.delete(X, 2, axis=1), qi[2:] - (qi[2:] > 2), qv[2:])
    keep = [0, 1, 3, 4, 5]
    assert np.allclose(ans["point"][2:, keep], expected)                     # others: OLS as usual


def test_int_pairwise_guards_fall_back_to_its_observational_pool_mean():
    world = World(1003, B=50)
    world.apply_shift(1)
    agent = make(IntPairwiseAgent, "int-pairwise", seed=1003, access=world.view())
    X_obs = world.observational_batch(1)
    agent.observe(X_obs, 1)
    agent.refit()
    mu_obs = X_obs.mean(axis=0)
    rng = np.random.default_rng(22)
    qi, qv = queries(rng)

    # (a) no interventional rows at all: every pair falls back to the observational mean
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    assert np.allclose(ans["point"], np.broadcast_to(mu_obs, (len(qi), D)))
    assert np.all(off_target(ans["quantiles"], qi)[..., 4] >= off_target(ans["quantiles"], qi)[..., 1])

    # (b) 3 rows on target 0 (< 4) and 10 rows on target 1 all at v = 2 (1 distinct value):
    #     both targets still fall back; target 2 with 4 rows at two values is fitted
    k = 0
    for v in (2.0, -2.0, 1.0):
        agent.receive_intervention(0, v, world.intervene(1, k, 0, v)); k += 1
    for _ in range(10):
        agent.receive_intervention(1, 2.0, world.intervene(1, k, 1, 2.0)); k += 1
    rows2, vals2 = [], []
    for v in (2.0, -2.0, 1.0, -1.0):
        row = world.intervene(1, k, 2, v); k += 1
        agent.receive_intervention(2, v, row)
        rows2.append(row); vals2.append(v)
    qi = np.array([0, 1, 2, 2])
    qv = np.array([1.3, -0.4, 0.0, 1.0])
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    assert np.allclose(ans["point"][0], mu_obs) and np.allclose(ans["point"][1], mu_obs)
    A = np.column_stack([np.ones(4), vals2])
    beta, *_ = np.linalg.lstsq(A, np.stack(rows2), rcond=None)
    for j in range(D):
        if j != 2:
            assert abs(ans["point"][2, j] - beta[0, j]) < 1e-8
            assert abs(ans["point"][3, j] - beta[0, j] - beta[1, j]) < 1e-8
    # the fitted pair is NOT the marginal mean
    assert not np.allclose(np.delete(ans["point"][3], 2), np.delete(mu_obs, 2))


def test_int_pairwise_window_drops_old_rows_and_keeps_W_episodes():
    world = World(1004, B=10)
    agent = make(IntPairwiseAgent, "int-pairwise", seed=1004, constants={"W": 2},
                 access=world.view())
    assert agent.W == 2
    for t in range(1, 5):
        world.apply_shift(t)
        agent.observe(world.observational_batch(t), t)
        agent.refit()
        # rows of episodes t-1 and t survive after refit (episode t's own rows arrive below)
        tgt, val, X = agent.int_pool()
        assert len(tgt) == (10 if t >= 2 else 0)
        assert agent.obs_pool().shape[0] == 200 * t      # fallback pool is cumulative
        for k in range(10):
            i, v = agent.cycle.round_robin()
            agent.receive_intervention(i, v, world.intervene(t, k, i, v))
        tgt, val, X = agent.int_pool()
        assert len(tgt) == 10 * min(t, 2)
        assert X.shape == (len(tgt), D) and set(val).issubset(ALLOWED_VALUES)
    assert make(IntPairwiseAgent, "int-pairwise").W == 5          # documented default


def test_int_pairwise_free_policy_is_uniform_random_with_the_value_cycle():
    a = make(IntPairwiseAgent, "int-pairwise", seed=7)
    b = make(IntPairwiseAgent, "int-pairwise", seed=7)
    plan = a.choose_free_interventions(400, 4)
    assert len(plan) == 400
    targets = np.array([i for i, _ in plan])
    assert targets.min() >= 0 and targets.max() < D
    assert np.all(np.bincount(targets, minlength=D) > 30)       # every target gets used
    # values follow each variable's cycle [2, -2, 1, -1] from its current position
    pos = np.zeros(D, dtype=int)
    for i, v in plan:
        assert v == VALUE_CYCLE[pos[i] % 4] and v in ALLOWED_VALUES
        pos[i] += 1
    # the targets come from the agent's keyed stream: same seed and agent_id -> same plan
    assert b.choose_free_interventions(400, 4) == plan
    # ... and a different seed gives a different one
    c = make(IntPairwiseAgent, "int-pairwise", seed=8)
    assert c.choose_free_interventions(400, 4) != plan


def test_int_pairwise_bootstrap_interval_covers_truth_at_roughly_nominal_rate():
    """Sanity on the shared-row bootstrap: with 400 clamped rows per target the 90 % interval
    for E[X_j | do(X_i = v)] should cover the exact truth most of the time."""
    world = World(1005, B=2400)
    world.apply_shift(1)
    agent = make(IntPairwiseAgent, "int-pairwise", seed=1005, access=world.view())
    agent.observe(world.observational_batch(1), 1)
    agent.refit()
    k = 0
    for i in range(D):
        for _ in range(400):
            v = agent.cycle.next_value(i)
            agent.receive_intervention(i, v, world.intervene(1, k, i, v)); k += 1
    qi, qv = world.query_set(1)
    ans = agent.answer(qi, qv)
    check_answer_shape(ans, qi)
    truth = np.stack([world.truth(int(i), float(v)) for i, v in zip(qi, qv)])
    y, q = off_target(truth, qi), off_target(ans["quantiles"], qi)
    cov90 = np.mean((q[..., 1] <= y) & (y <= q[..., 4]))
    assert 0.8 <= cov90 <= 0.98, cov90


# --------------------------------------------------------------------------- marginal-mean

def test_marginal_mean_predicts_cumulative_mean_and_never_abstains():
    rng = np.random.default_rng(30)
    agent = make(MarginalMeanAgent, "marginal-mean", constants={"tau": 0.01})
    assert agent.tau == math.inf and agent.never_abstains and not agent.intervenes
    batches = []
    for t in range(1, 4):
        X = gaussian_batch(rng) + t
        batches.append(X)
        agent.observe(X, t)
        agent.refit()
        qi, qv = queries(rng)
        ans = agent.answer(qi, qv)
        check_answer_shape(ans, qi)
        mu = np.concatenate(batches).mean(axis=0)
        assert np.allclose(ans["point"], np.broadcast_to(mu, (len(qi), D)))
        # the prediction ignores (i, v): identical rows for every query
        assert np.all(ans["point"] == ans["point"][0])
        assert np.all(ans["quantiles"] == ans["quantiles"][0])
        # bootstrap of the mean: interval brackets the point and shrinks as the pool grows
        q = ans["quantiles"][0]
        assert np.all(q[:, 1] <= mu) and np.all(mu <= q[:, 4])
        width = q[:, 4] - q[:, 1]
        if t > 1:
            assert np.all(width < prev_width * 1.1)
        prev_width = width
    # through the evaluator: abstain is False everywhere because tau = inf
    ev = protocol.evaluate_answer(ans, qi, np.zeros((len(qi), D)), np.ones(D), agent.tau)
    assert not np.any(ev["abstain"])


# --------------------------------------------------------------------------- oracle

def test_oracle_returns_exact_truth_with_zero_width():
    world = World(1006)
    agent = build_agent("oracle", world, 1006, {"tau": 0.5})
    assert isinstance(agent, OracleAgent) and agent.is_oracle
    assert agent.tau == math.inf and not agent.intervenes and agent.never_abstains
    for t in range(1, 10):
        world.apply_shift(t)
        agent.observe(world.observational_batch(t), t)
        agent.refit()
        qi, qv = world.query_set(t)
        ans = agent.answer(qi, qv)
        check_answer_shape(ans, qi)
        truth = np.stack([world.truth(int(i), float(v)) for i, v in zip(qi, qv)])
        assert np.array_equal(ans["point"], truth)
        assert np.all(ans["quantiles"] == truth[:, :, None])
        ev = protocol.evaluate_answer(ans, qi, truth, world.sigma_ref(), agent.tau)
        assert np.all(ev["err"] == 0) and np.all(ev["cov50"]) and np.all(ev["cov99"])
        assert np.all(ev["width90_raw"] == 0) and not np.any(ev["abstain"])
    # the oracle tracks the current SCM: after a shift its answers move with the truth
    rec = world.schedule[0]
    while world._t < rec.t:                        # advance to the first shift episode
        world.apply_shift(world._t + 1)
    qi, qv = world.query_set(rec.t)
    truth = np.stack([world.truth(int(i), float(v)) for i, v in zip(qi, qv)])
    assert np.array_equal(agent.answer(qi, qv)["point"], truth)


# --------------------------------------------------------------------------- integration

class ShortRun:
    """World(seed=1000) truncated to `T` episodes for protocol.run_agent.

    World refuses T < 55 because the shift schedule lives in [7, 55]; for a 3-episode smoke
    run no shift can occur, so delegating everything and overriding only T is exact."""

    def __init__(self, world, T):
        self._world = world
        self.T = int(T)

    def __getattr__(self, name):
        return getattr(self._world, name)


BASELINE_NAMES = ("marginal-mean", "obs-window", "obs-cumulative", "int-pairwise", "oracle")


@pytest.mark.parametrize("B", [10, 50])
@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_each_baseline_runs_three_episodes_through_the_protocol(name, B):
    T, Q = 3, 200
    world = ShortRun(World(1000, B=B), T)
    agent = build_agent(name, world._world, 1000, {"tau": 1.0, "W": 2})
    raw = protocol.run_agent(world, agent)

    # ---- schema (INTERFACES section 6)
    d = D
    expected = {
        "episode_t": (np.int32, (T,)), "shift_j": (np.int32, (T,)),
        "shift_type_code": (np.int8, (T,)), "shift_m": (np.float32, (T,)),
        "shift_delta": (np.float32, (T,)), "shift_delta_sigma": (np.float32, (T,)),
        "true_adj": (np.bool_, (T, d, d)), "true_W": (np.float32, (T, d, d)),
        "true_b": (np.float32, (T, d)), "true_sigma": (np.float32, (T, d)),
        "sd_ref": (np.float32, (T, d)), "F_obs": (np.float32, (T,)),
        "conf_c": (np.float32, (T,)), "conf_F": (np.float32, (T,)),
        "err": (np.float32, (T, Q, d - 1)), "width50": (np.float32, (T, Q, d - 1)),
        "width90": (np.float32, (T, Q, d - 1)), "width99": (np.float32, (T, Q, d - 1)),
        "width90_raw": (np.float32, (T, Q, d - 1)), "cov50": (np.bool_, (T, Q, d - 1)),
        "cov90": (np.bool_, (T, Q, d - 1)), "cov99": (np.bool_, (T, Q, d - 1)),
        "abstain": (np.bool_, (T, Q, d - 1)),
        "query_i": (np.int32, (T, Q)), "query_v": (np.float32, (T, Q)),
        "intervention_i": (np.int32, (T, B)), "intervention_v": (np.float32, (T, B)),
        "sample_score": (np.float32, (T, B)),
        "reset_events": (np.int32, (0, 2)), "struct_refit_events": (np.int32, (0, 2)),
        "learned_adj": (np.bool_, (T, d, d)), "has_learned_adj": (np.bool_, (T,)),
        "g_stat": (np.float32, (T, d)), "glr_stat": (np.float32, (T, d)),
        "self_score": (np.float32, (T,)), "var_obj_before": (np.float32, (T,)),
        "var_obj_after": (np.float32, (T,)),
        "agent_id": (np.int32, ()), "seed": (np.int32, ()), "d": (np.int32, ()),
        "T": (np.int32, ()), "B": (np.int32, ()), "n_obs": (np.int32, ()), "Q": (np.int32, ()),
        "hetero": (np.int32, ()), "hidden": (np.bool_, ()), "shifts_enabled": (np.bool_, ()),
        "learns_structure": (np.bool_, ()), "has_detector": (np.bool_, ()),
        "intervenes": (np.bool_, ()), "uses_floor": (np.bool_, ()), "is_oracle": (np.bool_, ()),
        "tau": (np.float32, ()), "wall_time_s": (np.float32, ()),
        "hidden_A": (np.int32, ()), "hidden_B": (np.int32, ()),
    }
    for key, (dtype, shape) in expected.items():
        assert key in raw, key
        arr = np.asarray(raw[key])
        assert arr.dtype == dtype, (key, arr.dtype)
        assert arr.shape == shape, (key, arr.shape)
    assert set(raw) == set(expected) | {"agent_name", "shift_type"}
    assert str(raw["agent_name"]) == name and int(raw["agent_id"]) == AGENT_IDS[name]
    assert int(raw["T"]) == T and int(raw["B"]) == B and int(raw["Q"]) == Q

    # ---- content
    assert np.all(np.isfinite(raw["err"])) and np.all(np.isfinite(raw["width90_raw"]))
    assert np.all(raw["width90_raw"] >= 0) and np.all(raw["shift_j"] == -1)
    assert not raw["learns_structure"] and not raw["has_detector"]
    assert not np.any(raw["has_learned_adj"]) and np.all(np.isnan(raw["sample_score"]))
    assert np.all(np.isnan(raw["self_score"]))
    assert bool(raw["is_oracle"]) == REGISTRY[name].is_oracle

    if name == "int-pairwise":
        assert raw["intervenes"] and raw["uses_floor"]
        assert np.all(raw["intervention_i"] >= 0) and np.all(raw["intervention_i"] < d)
        assert set(np.unique(raw["intervention_v"])).issubset(ALLOWED_VALUES)
        # episodes 1-3 are warm-up: the whole budget is round-robin, so per-variable counts
        # differ by at most one and the cycle continues across episodes
        flat = raw["intervention_i"].ravel()
        assert np.array_equal(flat, np.arange(T * B) % d)
        for r in range(T):
            counts = np.bincount(raw["intervention_i"][r], minlength=d)
            assert counts.sum() == B and counts.max() - counts.min() <= 1
    else:
        assert not raw["intervenes"]
        assert np.all(raw["intervention_i"] == -1) and np.all(np.isnan(raw["intervention_v"]))

    if name in ("marginal-mean", "oracle"):
        assert raw["tau"] == np.float32(np.inf) and not np.any(raw["abstain"])
    else:
        assert raw["tau"] == np.float32(1.0)
        assert np.array_equal(raw["abstain"], raw["width90_raw"] > 1.0)

    if name == "oracle":
        assert np.all(raw["err"] == 0) and np.all(raw["cov50"]) and np.all(raw["width99"] == 0)
    else:
        # a real learner: nonzero errors, nonzero widths, and a sane nMSE for a 3-episode run
        assert np.any(raw["err"] != 0) and np.all(raw["width90_raw"] > 0)
        assert np.mean(raw["err"] ** 2) < 5.0
        # marginal-mean ignores (i, v) so its interval contains the truth less often than the
        # interventional learner does, which is the point of the floor reference
