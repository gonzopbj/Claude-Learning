"""Tests for world.py.

SPEC.md "Tests" items implemented here:
  1.  exact interventional mean vs 1e5 mutilated-SCM samples (gamma_h in {0, 1}); affine in v;
      equals E[X_k] for non-descendants; unchanged on j-independent (i, k) after a shift in j
  2.  (world side) two dummy agents with different policies see byte-identical batches,
      query sets, schedules and interventional noise rows - through the REAL World and the
      real protocol, reusing the dummy agents of test_protocol.py
  3.  WorldView leaks nothing; agents/ never imports world (grep); RelabelledWorld, the
      helper for "mech-full on a relabelled copy gives identical predictions up to
      relabelling", really is the same world up to relabelling
  9.  hidden generator: the H -> A -> B, H -> B motif on a visible edge; H is never a query
      or intervention target; c and F_conf match a 1e5-sample OLS
  10. schedule: exactly 8 shifts in [7, 55], min gap 5, identical across agents / instances;
      m matches a Monte Carlo estimate over the query distribution
plus the generator's marginals, the keyed streams and the shifts_enabled=False path.
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np
import pytest

import world as world_mod
from protocol import run_agent
from test_protocol import FixedTargetAgent, RandomTargetAgent, make
from world import (E_V2, K_SHIFTS, MIN_GAP, SHIFT_WINDOW, OracleAccess, RelabelledWorld,
                   World, WorldView, _generate_visible_scm, structural_sample)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------- helpers

def transitive_closure(adj: np.ndarray) -> np.ndarray:
    """R[a, b] = True iff there is a directed path a -> ... -> b (a != b).  adj is children
    in rows (adj[j, p]: p -> j)."""
    A = adj.T.astype(int)                 # A[p, j] = 1 iff p -> j
    R = A.copy()
    for _ in range(adj.shape[0]):
        R = ((R + R @ A) > 0).astype(int)
    return R.astype(bool)


def j_dependent(i: int, k: int, j: int, R: np.ndarray) -> bool:
    """SPEC Metric 3: query (i, k) depends on mechanism j iff k == j or (j in desc(i) and
    j in anc(k)); if j == i the intervention cuts j's mechanism, so the query is independent."""
    if j == i:
        return False
    return k == j or (R[i, j] and R[j, k])


def step_to(w, t: int):
    """Call apply_shift for 1..t in order (the world requires the order); return records."""
    recs = []
    for u in range(1, t + 1):
        r = w.apply_shift(u)
        if r is not None:
            recs.append(r)
    return recs


def full_scm(w: World):
    """(W, b, sigma, order) of the current full SCM, for Monte Carlo checks."""
    s = w._current
    return s.W, s.b, s.sigma, w._order


# --------------------------------------------------------------------------- generator

def test_generator_marginals_and_relabelling():
    n_edges = []
    for s in range(300):
        rng = np.random.default_rng([s, 0])
        adj, W, b, sigma, perm = _generate_visible_scm(rng, 6)
        assert adj.any()
        n_edges.append(adj.sum())
        assert np.array_equal(adj, W != 0)
        # strictly lower-triangular in the generation order: perm is a topological order
        W_gen = W[np.ix_(perm, perm)]
        assert np.allclose(np.triu(W_gen), 0.0)
        mags = np.abs(W[adj])
        assert np.all((mags >= 0.5) & (mags <= 1.5))
        assert np.all((b >= -1) & (b <= 1)) and np.all((sigma >= 0.5) & (sigma <= 1.5))
        assert sorted(perm.tolist()) == list(range(6))
    # p_edge = 0.6 on 15 possible edges: mean 9 (conditioning on >= 1 edge is negligible)
    assert 8.4 <= np.mean(n_edges) <= 9.6
    # the relabelling is not the identity in general
    assert any(not np.array_equal(World(s).permutation(), np.arange(6)) for s in range(5))


def test_world_uses_only_keyed_generators():
    """Every `np.random.<name>` in world.py's CODE (docstrings do not count) is default_rng
    or the Generator type: no global np.random.* functions anywhere."""
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "world.py")).read())
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "np" and node.value.attr == "random"):
            used.add(node.attr)
    assert used and used <= {"default_rng", "Generator"}, used


# --------------------------------------------------------------------------- SPEC test 1

@pytest.mark.parametrize("hetero", [0, 1])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_mean_matches_monte_carlo(seed, hetero):
    w = World(seed, hetero=hetero)
    step_to(w, 1)
    W, b, sigma, order = full_scm(w)
    n = 100_000
    rng = np.random.default_rng(12345)
    for i, v in [(0, 2.0), (3, -1.3), (5, 0.7)]:
        X = structural_sample(W, b, sigma, hetero, order, rng.standard_normal((n, w.d_full)),
                              (i, v))[:, :w.d]
        truth = w.truth(i, v)
        assert truth[i] == v and np.all(X[:, i] == v)     # the clamped coordinate is exact
        others = np.arange(w.d) != i
        se = X[:, others].std(axis=0, ddof=1) / np.sqrt(n)
        assert np.all(np.abs(X[:, others].mean(axis=0) - truth[others]) <= 4 * se)
    # observational moments too: sample mean within 4 SE of mu, and for gamma_h = 0 the
    # sample covariance diagonal close to Var_ref
    X = structural_sample(W, b, sigma, hetero, order, rng.standard_normal((n, w.d_full)))[:, :w.d]
    se = X.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(X.mean(axis=0) - w.mean_obs()) <= 4 * se)
    if hetero == 0:
        assert np.allclose(X.var(axis=0, ddof=1), w.var_ref(), rtol=0.05)
    # the reference scale does not depend on the knob
    assert np.array_equal(w.var_ref(), World(seed, hetero=0).var_ref())
    assert np.allclose(w.sigma_ref(), np.sqrt(w.var_ref()))


def test_truth_is_affine_in_v_and_uses_total_effects():
    for seed in range(5):
        w = World(seed)
        step_to(w, 1)
        Theta = np.linalg.inv(np.eye(w.d) - w.true_W())
        for i in range(w.d):
            t0, t1 = w.truth(i, 0.0), w.truth(i, 1.0)
            # slope in v equals Theta[:, i] = ((I - W_i)^-1)[:, i]  (SPEC "Exact means")
            assert np.allclose(t1 - t0, Theta[:, i])
            for v in (-2.0, -0.5, 1.7):
                assert np.allclose(w.truth(i, v), t0 + v * (t1 - t0))
            # c(i) = (I - W_i)^-1 b_i is what truth(i, 0) must be
            W_i = w.true_W()
            W_i[i] = 0.0
            b_i = w.true_b()
            b_i[i] = 0.0
            assert np.allclose(t0, np.linalg.solve(np.eye(w.d) - W_i, b_i))


def test_non_descendants_get_the_observational_mean():
    for seed in range(5):
        w = World(seed)
        step_to(w, 1)
        R = transitive_closure(w.true_adjacency())
        mu = w.mean_obs()
        for i in range(w.d):
            for v in (-2.0, 0.0, 1.5):
                tr = w.truth(i, v)
                for k in range(w.d):
                    if k != i and not R[i, k]:
                        assert np.isclose(tr[k], mu[k])
                    if k != i and R[i, k]:
                        # descendants do move (a total effect is a product of weights >= 0.25)
                        assert abs(w.truth(i, 2.0)[k] - w.truth(i, -2.0)[k]) > 1e-6


def test_shift_leaves_unaffected_queries_unchanged():
    """SPEC test 1, last clause, made precise.  truth(i, k, v) = c_k(i) + Theta[k, i] v.
    After a shift in mechanism j:
      * the SLOPE Theta[k, i] is unchanged whenever j is not on a directed i -> k path
        (j == i included: Theta[:, i] uses no edge into i);
      * the whole truth is unchanged when j == i (the intervention cuts j's mechanism) or
        when j is not an ancestor of k and j != k.
    The SPEC's Metric 3 calls (i, k) "j-independent" whenever j is not on an i -> k path;
    but if j is an ancestor of k through a path that avoids i (e.g. 0 -> 2, 1 -> 2, shift in
    1, query (0, 2)), the LEVEL c_k(i) still moves with b_j.  The last block below records
    that such queries exist, so metrics.py knows the mask is about slopes, not levels."""
    vs = np.linspace(-2, 2, 5)
    n_dep_changed = n_level_only = 0
    for seed in range(4):
        w = World(seed)
        R = transitive_closure(w.true_adjacency())
        before = None
        for t in range(1, w.T + 1):
            rec = w.apply_shift(t)
            now = {(i, v): w.truth(i, v) for i in range(w.d) for v in vs}
            if rec is not None:
                j = rec.j
                for i in range(w.d):
                    slope_now = now[(i, 1.0)] - now[(i, 0.0)]
                    slope_before = before[(i, 1.0)] - before[(i, 0.0)]
                    for k in range(w.d):
                        if k == i:
                            continue
                        on_path = j_dependent(i, k, j, R)          # SPEC Metric 3 mask
                        if not on_path:
                            assert np.isclose(slope_now[k], slope_before[k]), (seed, t, i, k, j)
                        truth_invariant = (j == i) or (j != k and not R[j, k])
                        changed = any(not np.isclose(now[(i, v)][k], before[(i, v)][k]) for v in vs)
                        if truth_invariant:
                            assert not changed, (seed, t, i, k, j)
                        elif on_path and changed:
                            n_dep_changed += 1
                        elif not on_path and changed:
                            n_level_only += 1
            before = now
    assert n_dep_changed > 0          # the shifts really move dependent queries
    assert n_level_only > 0           # ... and the level of some "j-independent" ones


# --------------------------------------------------------------------------- SPEC test 2

def test_two_policies_see_identical_streams_in_the_real_world():
    kw = dict(T=60, B=10, n_obs=20, n_queries=5)
    wa, wb = World(3, **kw), World(3, **kw)
    a = make(RandomTargetAgent, wa, agent_id=0)
    b = make(FixedTargetAgent, wb, agent_id=1)
    ra, rb = run_agent(wa, a), run_agent(wb, b)

    assert not np.array_equal(ra["intervention_i"][3:], rb["intervention_i"][3:])
    for (ta, Xa), (tb, Xb) in zip(a.obs_seen, b.obs_seen):
        assert ta == tb and Xa.tobytes() == Xb.tobytes()
    for (ia, va), (ib, vb) in zip(a.queries_seen, b.queries_seen):
        assert ia.tobytes() == ib.tobytes() and va.tobytes() == vb.tobytes()
    for key in ("shift_j", "shift_m", "true_W", "true_b", "true_sigma", "sd_ref", "F_obs",
                "query_i", "query_v"):
        assert ra[key].tobytes() == rb[key].tobytes(), key
    assert (ra["shift_j"] >= 0).sum() == K_SHIFTS

    # every row handed to either agent is the episode-t SCM pushed through noise row k of
    # stream [seed, t, 3], whatever (i, v) the agent chose -> the two agents share noise rows
    for agent, w in ((a, wa), (b, wb)):
        replay = World(3, **kw)
        assert len(agent.rows_seen) == 60 * 10
        for n, (i, v, row) in enumerate(agent.rows_seen):
            t, k = n // 10 + 1, n % 10
            if k == 0:
                replay.apply_shift(t)
            eps = np.random.default_rng([3, t, 3]).standard_normal((10, replay.d_full))
            W, bb, sg, order = full_scm(replay)
            expect = structural_sample(W, bb, sg, 0, order, eps[k:k + 1], (i, v))[0, :6]
            assert row[i] == v
            assert np.allclose(row, expect)


def test_streams_are_pure_and_keyed():
    w = World(7, B=12, n_obs=30, n_queries=11)
    step_to(w, 4)
    # observational batch = keyed noise pushed through the current SCM, visible columns
    eps = np.random.default_rng([7, 4, 2]).standard_normal((30, w.d_full))
    W, b, sigma, order = full_scm(w)
    assert np.array_equal(w.observational_batch(4), structural_sample(W, b, sigma, 0, order, eps)[:, :6])
    assert w.observational_batch(4).tobytes() == w.observational_batch(4).tobytes()
    assert w.observational_batch(4).shape == (30, 6)
    # interventional noise: stream [seed, t, 3], shape (B, d_full), pure, a copy each time
    noise = w.interventional_noise(4)
    assert noise.shape == (12, w.d_full)
    assert np.array_equal(noise, np.random.default_rng([7, 4, 3]).standard_normal((12, w.d_full)))
    noise[:] = 0.0
    assert np.array_equal(w.interventional_noise(4), np.random.default_rng([7, 4, 3]).standard_normal((12, w.d_full)))
    # query set: targets first, then values, from [seed, t, 4]
    rng = np.random.default_rng([7, 4, 4])
    qi, qv = w.query_set(4)
    assert qi.dtype == np.int64 and qv.dtype == np.float64 and qi.shape == qv.shape == (11,)
    assert np.array_equal(qi, rng.integers(0, 6, size=11)) and np.array_equal(qv, rng.uniform(-2, 2, size=11))
    assert np.all((qi >= 0) & (qi < 6)) and np.all((qv >= -2) & (qv <= 2))
    # intervene is pure and uses row k whatever (i, v): different targets, same noise row
    r1 = w.intervene(4, 5, 2, 2.0)
    w.intervene(4, 5, 0, -1.0)
    w.intervene(4, 5, 4, 1.0)
    assert np.array_equal(w.intervene(4, 5, 2, 2.0), r1)
    r_other = w.intervene(4, 5, 0, -1.0)
    # a variable that is a non-descendant of both targets takes the identical value
    R = transitive_closure(w.true_adjacency())
    for k in range(6):
        if k not in (0, 2) and not R[2, k] and not R[0, k]:
            assert r1[k] == r_other[k]
    with pytest.raises(ValueError):
        w.intervene(4, 12, 0, 1.0)
    with pytest.raises(ValueError):
        w.intervene(4, 0, 6, 1.0)
    # apply_shift must be called in order and not beyond T
    with pytest.raises(ValueError):
        w.apply_shift(6)
    w2 = World(7)
    step_to(w2, 60)
    with pytest.raises(ValueError):
        w2.apply_shift(61)


# --------------------------------------------------------------------------- SPEC test 3

def test_worldview_leaks_nothing():
    w = World(0, hidden=True)
    view = w.view()
    assert isinstance(view, WorldView)
    forbidden = {"W", "b", "sigma", "permutation", "schedule", "world", "truth"}
    public = {n for n in dir(view) if not n.startswith("_")}
    assert public == {"d", "B", "T", "n_obs", "n_queries"}
    assert set(vars(view)) == {"d", "B", "T", "n_obs", "n_queries"}
    assert not (public & forbidden) and not (set(vars(view)) & forbidden)
    for value in vars(view).values():
        assert isinstance(value, int)         # no arrays, no objects, no closures
    assert (view.d, view.B, view.T, view.n_obs, view.n_queries) == (6, 50, 60, 200, 200)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.d = 3
    # the oracle side is the only object that holds the world
    oa = w.oracle_access()
    assert isinstance(oa, OracleAccess) and oa.view == view
    assert np.array_equal(oa.true_adjacency(), w.true_adjacency())
    assert oa.hidden_pair() == w.hidden_pair()
    assert oa.shifted_mechanism(1) == -1
    assert [oa.shifted_mechanism(r.t) for r in w.schedule] == [r.j for r in w.schedule]


def test_agents_never_import_world():
    agents_dir = os.path.join(ROOT, "agents")
    pattern = re.compile(r"^\s*(from\s+world\b|import\s+world\b|from\s+\.\.\s*world\b)", re.M)
    for name in sorted(os.listdir(agents_dir)):
        if name.endswith(".py"):
            src = open(os.path.join(agents_dir, name)).read()
            assert not pattern.search(src), f"agents/{name} imports world internals"


@pytest.mark.parametrize("hidden", [False, True])
def test_relabelled_world_is_the_same_world_up_to_relabelling(hidden):
    kw = dict(B=10, n_obs=15, n_queries=9, hidden=hidden)
    pi = np.array([3, 5, 0, 4, 1, 2])          # new label of old variable a is pi[a]
    w = World(5, **kw)
    r = RelabelledWorld(World(5, **kw), pi)
    assert r.view() == w.view()
    assert np.array_equal(r.true_adjacency()[np.ix_(pi, pi)], w.true_adjacency())
    assert [rec.t for rec in r.schedule] == [rec.t for rec in w.schedule]
    assert [rec.j for rec in r.schedule] == [int(pi[rec.j]) if rec.j < 6 else rec.j for rec in w.schedule]
    if hidden:
        A, B = w.hidden_pair()
        assert r.hidden_pair() == (int(pi[A]), int(pi[B]))
    for t in range(1, 61):
        ra, rb = w.apply_shift(t), r.apply_shift(t)
        assert (ra is None) == (rb is None)
        if ra is not None:
            assert rb.j == (int(pi[ra.j]) if ra.j < 6 else ra.j) and rb.m == ra.m
        assert np.array_equal(r.observational_batch(t)[:, pi], w.observational_batch(t))
        qi_w, qv_w = w.query_set(t)
        qi_r, qv_r = r.query_set(t)
        assert np.array_equal(qi_r, pi[qi_w]) and np.array_equal(qv_r, qv_w)
        assert np.array_equal(r.true_W()[np.ix_(pi, pi)], w.true_W())
        assert np.array_equal(r.true_b()[pi], w.true_b())
        assert np.array_equal(r.true_sigma()[pi], w.true_sigma())
        assert np.array_equal(r.var_ref()[pi], w.var_ref())
        assert np.array_equal(r.mean_obs()[pi], w.mean_obs())
        assert r.F_obs() == w.F_obs()
        if hidden:
            cw, cr = w.confounding(), r.confounding()
            assert cr["c"] == cw["c"] and cr["F_conf"] == cw["F_conf"]
            assert (cr["A"], cr["B"]) == (int(pi[cw["A"]]), int(pi[cw["B"]]))
        for i, v, k in [(0, 2.0, 0), (4, -1.0, 3), (2, 1.0, 9)]:
            assert np.array_equal(r.truth(int(pi[i]), v)[pi], w.truth(i, v))
            assert np.array_equal(r.intervene(t, k, int(pi[i]), v)[pi], w.intervene(t, k, i, v))
    assert np.array_equal(r.permutation(), pi[w.permutation()])


# --------------------------------------------------------------------------- SPEC test 9

def test_hidden_generator_motif_and_visibility():
    for seed in range(6):
        main = World(seed)
        h = World(seed, hidden=True)
        assert h.d == 6 and h.d_full == 7 and main.d_full == 6
        # same visible SCM as the main run (before any shift), same relabelling
        step_to(main, 1)
        step_to(h, 1)
        assert np.array_equal(h.true_adjacency(), main.true_adjacency())
        assert np.array_equal(h.true_W(), main.true_W())
        assert np.array_equal(h.true_b(), main.true_b())
        assert np.array_equal(h.true_sigma(), main.true_sigma())
        assert np.array_equal(h.permutation(), main.permutation())
        # the motif: A -> B is a visible edge, H -> A and H -> B, H has no parents
        A, B = h.hidden_pair()
        W_full, b_full, sigma_full, order = full_scm(h)
        assert h.true_adjacency()[B, A]
        assert W_full[A, 6] != 0 and W_full[B, 6] != 0
        assert np.all(W_full[6, :] == 0) and order[0] == 6
        assert np.array_equal(W_full[:6, :6], main.true_W())
        for w_h in (W_full[A, 6], W_full[B, 6]):
            assert 0.5 <= abs(w_h) <= 1.5
        assert -1 <= b_full[6] <= 1 and 0.5 <= sigma_full[6] <= 1.5
        # H is removed from every agent-facing quantity
        assert h.observational_batch(1).shape == (200, 6)
        assert h.intervene(1, 0, A, 2.0).shape == (6,) and h.truth(A, 2.0).shape == (6,)
        assert h.true_W().shape == (6, 6) and h.true_adjacency().shape == (6, 6)
        assert h.var_ref().shape == (6,) and h.view() == main.view()
        for t in range(1, 61):
            qi, _ = h.query_set(t)
            assert np.all(qi < 6)
        with pytest.raises(ValueError):
            h.intervene(1, 0, 6, 1.0)
        with pytest.raises(ValueError):
            h.truth(6, 1.0)
        # the schedule: same episodes as the main run, 4th shift is a large shift of H,
        # every other shift identical to the main run's (same j, same new parameters; the
        # logged delta_sigma may differ when the skipped visible shift hit the same j)
        assert [r.t for r in h.schedule] == [r.t for r in main.schedule]
        for k, (rm, rh) in enumerate(zip(main.schedule, h.schedule)):
            step_to_t = rh.t
            m2, h2 = World(seed), World(seed, hidden=True)
            step_to(m2, step_to_t)
            step_to(h2, step_to_t)
            if k == 3:
                assert rh.j == 6 and rh.shift_type == "large"
                assert rh.delta == 0.0 and rh.delta_sigma > 0
                # the H shift leaves every visible parameter alone
                h_prev = World(seed, hidden=True)
                step_to(h_prev, step_to_t - 1)
                assert np.array_equal(h2.true_W(), h_prev.true_W())
                assert np.array_equal(h2.true_b(), h_prev.true_b())
                assert np.array_equal(h2.true_sigma(), h_prev.true_sigma())
                # ... but moves the visible means through b_H
                assert not np.allclose(h2.mean_obs(), h_prev.mean_obs())
            else:
                assert (rh.j, rh.shift_type) == (rm.j, rm.shift_type)
                j = rh.j
                assert np.array_equal(h2.true_W()[j], m2.true_W()[j])
                assert h2.true_b()[j] == m2.true_b()[j] and h2.true_sigma()[j] == m2.true_sigma()[j]
        # H shifts the visible means (through b_H) but not the visible total effects
        assert np.allclose(h.truth(A, 2.0) - h.truth(A, 0.0), main.truth(A, 2.0) - main.truth(A, 0.0))


def test_hidden_confounding_matches_sample_ols():
    n = 100_000
    for seed in range(4):
        h = World(seed, hidden=True)
        step_to(h, 1)
        conf = h.confounding()
        A, B = conf["A"], conf["B"]
        assert (A, B) == h.hidden_pair()
        W_full, b_full, sigma_full, order = full_scm(h)
        X = structural_sample(W_full, b_full, sigma_full, 0, order,
                              np.random.default_rng(99 + seed).standard_normal((n, 7)))
        pa = np.flatnonzero(h.true_adjacency()[B])
        assert A in pa
        D = np.column_stack([np.ones(n), X[:, pa]])
        beta, *_ = np.linalg.lstsq(D, X[:, B], rcond=None)
        resid = X[:, B] - D @ beta
        s2 = resid @ resid / (n - D.shape[1])
        cov_beta = s2 * np.linalg.inv(D.T @ D)
        col = 1 + list(pa).index(A)
        beta_A, se_A = beta[col], np.sqrt(cov_beta[col, col])
        c_hat = beta_A - W_full[B, A]
        assert abs(c_hat - conf["c"]) <= 4 * se_A
        # F_conf = c^2 (4/3) / Var_ref(X_B), propagated tolerance from c's standard error
        F_hat = c_hat ** 2 * E_V2 / X[:, B].var(ddof=1)
        tol = 4 * (2 * abs(conf["c"]) * E_V2 / h.var_ref()[B]) * se_A + 0.02 * conf["F_conf"] + 1e-4
        assert abs(F_hat - conf["F_conf"]) <= tol
        assert conf["F_conf"] == pytest.approx(conf["c"] ** 2 * E_V2 / h.var_ref()[B])
    # the main run has no confounding record
    assert World(0).confounding() is None and World(0).hidden_pair() is None


# --------------------------------------------------------------------------- SPEC test 10

@pytest.mark.parametrize("shift_type", ["large", "small", "noise-only"])
def test_schedule_shape_and_identity(shift_type):
    for seed in range(10):
        w = World(seed, shift_type=shift_type)
        ts = [r.t for r in w.schedule]
        assert len(ts) == K_SHIFTS and ts == sorted(ts)
        assert all(SHIFT_WINDOW[0] <= t <= SHIFT_WINDOW[1] for t in ts)
        assert all(b - a >= MIN_GAP for a, b in zip(ts, ts[1:]))
        assert all(r.shift_type == shift_type for r in w.schedule)
        assert all(0 <= r.j < 6 for r in w.schedule)
        # identical for a second instance (the schedule depends on the seed only) and
        # the episodes do not depend on the shift type or the hidden flag
        assert World(seed, shift_type=shift_type).schedule == w.schedule
        assert [r.t for r in World(seed, hidden=True).schedule] == ts
        assert [r.t for r in World(seed, shift_type="large").schedule] == ts
        # apply_shift returns exactly the schedule's records at their episodes
        got = []
        for t in range(1, 61):
            rec = w.apply_shift(t)
            if rec is not None:
                got.append(rec)
                assert rec.t == t
        assert got == w.schedule


def test_shift_types_change_what_they_should():
    for seed in range(6):
        # large: j has a parent; incoming weights, b_j, sigma_j all change (in law)
        w = World(seed, shift_type="large")
        adj = w.true_adjacency()
        W_prev, b_prev, s_prev = w.true_W(), w.true_b(), w.true_sigma()
        for t in range(1, 61):
            rec = w.apply_shift(t)
            W_now, b_now, s_now = w.true_W(), w.true_b(), w.true_sigma()
            if rec is None:
                assert np.array_equal(W_now, W_prev) and np.array_equal(b_now, b_prev)
                assert np.array_equal(s_now, s_prev)
            else:
                j = rec.j
                assert adj[j].any()
                assert np.array_equal(W_now != 0, adj)                 # skeleton fixed
                others = np.arange(6) != j
                assert np.array_equal(W_now[others], W_prev[others])   # only row j moves
                assert np.array_equal(b_now[others], b_prev[others])
                assert np.array_equal(s_now[others], s_prev[others])
                assert rec.delta_sigma == abs(s_now[j] - s_prev[j])
                mags = np.abs(W_now[j][adj[j]])
                assert np.all((mags >= 0.5) & (mags <= 1.5))
            W_prev, b_prev, s_prev = W_now, b_now, s_now
        # small: only the incoming weights of j move, by a N(0, 0.25^2) step, never below 0.25
        w = World(seed, shift_type="small")
        W_prev, b_prev, s_prev = w.true_W(), w.true_b(), w.true_sigma()
        for t in range(1, 61):
            rec = w.apply_shift(t)
            if rec is not None:
                j = rec.j
                assert rec.delta_sigma == 0.0
                assert np.array_equal(w.true_b(), b_prev) and np.array_equal(w.true_sigma(), s_prev)
                dW = w.true_W() - W_prev
                assert np.all(dW[np.arange(6) != j] == 0)
                assert np.all(np.abs(w.true_W()[j][adj[j]]) >= 0.25)
                assert np.all(np.abs(dW[j]) < 5 * 0.25)              # a 5-sigma step
                assert rec.m > 0 and rec.delta > 0
            W_prev, b_prev, s_prev = w.true_W(), w.true_b(), w.true_sigma()
        # noise-only: only sigma_j changes, truth is untouched, m = delta = 0 exactly
        w = World(seed, shift_type="noise-only")
        truth_prev = None
        for t in range(1, 61):
            rec = w.apply_shift(t)
            truth_now = np.stack([w.truth(i, v) for i in range(6) for v in (-2.0, 1.0)])
            if rec is not None:
                assert rec.m == 0.0 and rec.delta == 0.0 and rec.delta_sigma > 0
                assert np.array_equal(truth_now, truth_prev)
            truth_prev = truth_now


def test_effective_magnitude_matches_monte_carlo():
    n = 6000
    for seed, hidden in [(0, False), (1, False), (2, True)]:
        w = World(seed, hidden=hidden)
        for rec in w.schedule:
            before, after = World(seed, hidden=hidden), World(seed, hidden=hidden)
            step_to(before, rec.t - 1)
            step_to(after, rec.t)
            var_ref = after.var_ref()
            rng = np.random.default_rng([seed, rec.t, 77])
            qi = rng.integers(0, 6, size=n)
            qv = rng.uniform(-2, 2, size=n)
            per_query = np.empty(n)
            for q, (i, v) in enumerate(zip(qi, qv)):
                diff = after.truth(int(i), float(v)) - before.truth(int(i), float(v))
                ratio = diff ** 2 / var_ref
                per_query[q] = np.delete(ratio, i).mean()              # all k != i
            m_hat, se = per_query.mean(), per_query.std(ddof=1) / np.sqrt(n)
            assert abs(m_hat - rec.m) <= 5 * se + 1e-12, (seed, rec)
            # delta is the Frobenius norm of the change of the visible total-effect matrix
            Th_b = np.linalg.inv(np.eye(6) - before.true_W())
            Th_a = np.linalg.inv(np.eye(6) - after.true_W())
            assert rec.delta == pytest.approx(np.linalg.norm(Th_a - Th_b), abs=1e-9)


def test_shifts_disabled_keeps_the_scm_and_the_streams():
    on, off = World(4), World(4, shifts_enabled=False)
    assert off.schedule == [] and off.shifts_enabled is False
    W0 = on.true_W()
    for t in range(1, 61):
        on.apply_shift(t)
        assert off.apply_shift(t) is None
        assert np.array_equal(off.true_W(), W0) and np.array_equal(off.true_b(), World(4).true_b())
        # identical observational data as long as the enabled world has not shifted yet
        if t < on.schedule[0].t:
            assert on.observational_batch(t).tobytes() == off.observational_batch(t).tobytes()
        assert on.interventional_noise(t).tobytes() == off.interventional_noise(t).tobytes()
        assert on.query_set(t)[0].tobytes() == off.query_set(t)[0].tobytes()


# --------------------------------------------------------------------------- F_obs (P1)

def test_F_obs_matches_its_definition_and_sample_regression():
    for seed in range(3):
        w = World(seed)
        step_to(w, 1)
        d = w.d
        Theta = np.linalg.inv(np.eye(d) - w.true_W())
        Sig = Theta @ np.diag(w.true_sigma() ** 2) @ Theta.T
        mu = Theta @ w.true_b()
        acc = []
        for i in range(d):
            c_i = w.truth(i, 0.0)
            for j in range(d):
                if j == i:
                    continue
                beta1 = Sig[j, i] / Sig[i, i]
                beta0 = mu[j] - beta1 * mu[i]
                ev = (beta0 - c_i[j]) ** 2 + (beta1 - Theta[j, i]) ** 2 * E_V2
                acc.append(ev / Sig[j, j])
        assert w.F_obs() == pytest.approx(np.mean(acc))
        # the population OLS slope of X_j on X_i really is Sig[j, i] / Sig[i, i]
        n = 200_000
        W, b, sigma, order = full_scm(w)
        X = structural_sample(W, b, sigma, 0, order, np.random.default_rng(5).standard_normal((n, d)))
        for (i, j) in [(0, 1), (3, 2), (5, 4)]:
            slope = np.cov(X[:, i], X[:, j])[0, 1] / X[:, i].var()
            assert slope == pytest.approx(Sig[j, i] / Sig[i, i], abs=0.03)
    # F_obs is positive whenever the graph has an edge (see is not do somewhere)
    assert all(World(s).F_obs() > 0 for s in range(5))
