"""SPEC test 8: metrics on hand-computed cases (recovery on fixed nMSE series - no bump,
censored, immediate; the j-dependence mask on the 3-node chain; coverage on a fixed list;
AURC on 3 predictions; attribution confusion on a fixed event log), plus the bootstrap and
Holm helpers on tiny inputs, plus one end-to-end check: the raw dict of the `oracle` agent
produced by `protocol.run_agent` on `World(seed=1000)` has nMSE == 0 and coverage == 1.

The hand-built raw dicts below carry only the keys each metric reads (INTERFACES.md
section 6 names them); `_tiny_raw` fills the rest with neutral values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import metrics as M

# --------------------------------------------------------------------------- fixtures

CHAIN3 = np.zeros((3, 3), dtype=bool)          # 0 -> 1 -> 2, children in rows
CHAIN3[1, 0] = True
CHAIN3[2, 1] = True


def _tiny_raw(T=4, Q=3, d=3, true_adj=CHAIN3, query_i=None):
    """A minimal raw dict in the INTERFACES.md section 6 schema (per-prediction arrays zero,
    no shifts, no resets) that the tests then overwrite field by field."""
    f32 = np.float32
    qi = np.tile(np.arange(Q) % d, (T, 1)).astype(np.int32) if query_i is None else np.asarray(query_i, np.int32)
    raw = {
        "episode_t": np.arange(1, T + 1, dtype=np.int32),
        "shift_j": np.full(T, -1, np.int32), "shift_type_code": np.full(T, -1, np.int8),
        "shift_m": np.full(T, np.nan, f32), "shift_delta": np.full(T, np.nan, f32),
        "shift_delta_sigma": np.full(T, np.nan, f32),
        "true_adj": np.broadcast_to(true_adj, (T, d, d)).copy(),
        "sd_ref": np.ones((T, d), f32), "F_obs": np.zeros(T, f32),
        "err": np.zeros((T, Q, d - 1), f32),
        "width50": np.zeros((T, Q, d - 1), f32), "width90": np.zeros((T, Q, d - 1), f32),
        "width99": np.zeros((T, Q, d - 1), f32),
        "cov50": np.ones((T, Q, d - 1), bool), "cov90": np.ones((T, Q, d - 1), bool),
        "cov99": np.ones((T, Q, d - 1), bool), "abstain": np.zeros((T, Q, d - 1), bool),
        "query_i": qi, "query_v": np.zeros((T, Q), f32),
        "reset_events": np.zeros((0, 2), np.int32), "struct_refit_events": np.zeros((0, 2), np.int32),
        "learned_adj": np.zeros((T, d, d), bool), "has_learned_adj": np.zeros(T, bool),
        "self_score": np.full(T, np.nan, f32), "var_obj_before": np.full(T, np.nan, f32),
        "var_obj_after": np.full(T, np.nan, f32),
        "hidden_A": np.int32(-1), "hidden_B": np.int32(-1),
    }
    return raw


def _add_shift(raw, t, j, m, code=0, delta_sigma=0.0):
    raw["shift_j"][t - 1] = j
    raw["shift_type_code"][t - 1] = code
    raw["shift_m"][t - 1] = m
    raw["shift_delta"][t - 1] = 0.0
    raw["shift_delta_sigma"][t - 1] = delta_sigma


# --------------------------------------------------------------------------- helpers

def test_column_to_j_layout():
    qi = np.array([[0, 2, 1]])
    J = M.column_to_j(qi, 3)
    assert J.tolist() == [[[1, 2], [0, 1], [0, 2]]]


def test_transitive_closure_chain():
    R = M.transitive_closure(CHAIN3)
    assert R[2, 0] and R[1, 0] and R[2, 1]          # 0 reaches 1 and 2, 1 reaches 2
    assert not R[0, 2] and not R[0, 0]


# --------------------------------------------------------------------------- recovery (metric 2)

def _series(T, level, t, post):
    s = np.full(T, level)
    s[t - 1:t - 1 + len(post)] = post
    return s


def test_recovery_threshold_rule():
    # L = mean(0.10, 0.10) = 0.10 -> max(1.5 L, L + 0.02) = 0.15
    s = _series(12, 0.10, 5, [0.5])
    assert M.recovery_threshold(s, 5) == pytest.approx(0.15)
    # Small L: the additive floor L + 0.02 wins over 1.5 L
    s = _series(12, 0.01, 5, [0.5])
    assert M.recovery_threshold(s, 5) == pytest.approx(0.03)


def test_recovery_immediate_no_bump():
    s = _series(12, 0.10, 5, [0.12, 0.11])              # never leaves the band
    assert M.recovery_time(s, 5) == (0, True)


def test_recovery_delayed():
    s = _series(12, 0.10, 5, [0.5, 0.3, 0.14, 0.1])     # recovers at u = 7 -> delay 2
    assert M.recovery_time(s, 5) == (2, True)


def test_recovery_censored_at_next_shift():
    s = _series(12, 0.10, 5, [0.5, 0.3, 0.2, 0.1])      # would recover at u = 8 ...
    assert M.recovery_time(s, 5, next_shift_t=8) == (2, False)   # ... but 8 is the next shift
    assert M.recovery_time(s, 5, next_shift_t=9) == (3, True)


def test_recovery_censored_at_T():
    s = _series(8, 0.10, 5, [0.5, 0.5, 0.5, 0.5])
    assert M.recovery_time(s, 5) == (3, False)


def test_kaplan_meier_hand_case():
    times, events = [0, 2, 2, 1], [True, True, False, True]
    grid, S = M.kaplan_meier(times, events)
    assert grid.tolist() == [0, 1, 2]
    # S(0) = 3/4; S(1) = 3/4 * 2/3 = 1/2; at t = 2 both remaining units are at risk -> 1/4
    assert S == pytest.approx([0.75, 0.5, 0.25])
    assert M.km_median(grid, S) == 1
    summ = M.recovery_summary(times, events)
    assert summ["within_1"] == pytest.approx(0.25)      # recovered in the shift episode
    assert summ["within_2"] == pytest.approx(0.5)
    assert summ["within_3"] == pytest.approx(0.75)
    assert summ["n_censored"] == 1


def test_kaplan_meier_all_censored_has_no_median():
    grid, S = M.kaplan_meier([3, 3], [False, False])
    assert grid.size == 0 and math.isnan(M.km_median(grid, S))


def test_post_shift_regret_window():
    a = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    ref = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
    assert M.post_shift_regret(a, ref, 3) == pytest.approx(2.0)     # u = 3..6
    assert M.post_shift_regret(a, ref, 6) == pytest.approx(0.5)     # clipped at T = 7


# --------------------------------------------------------------------------- j-dependence (metric 3)

def test_j_dependence_mask_chain_spec_case():
    # chain 0 -> 1 -> 2, shift in j = 1 (SPEC Metric 3 test case)
    qi = np.array([0, 1, 2])
    dep = M.j_dependent_mask(CHAIN3, qi, j=1)
    J = M.column_to_j(qi, 3)
    table = {(int(i), int(k)): bool(dep[q, c]) for q, i in enumerate(qi) for c, k in enumerate(J[q])}
    assert table[(0, 2)] is True and table[(0, 1)] is True         # dependent
    assert table[(1, 2)] is False and table[(2, 0)] is False       # independent
    assert table[(1, 0)] is False and table[(2, 1)] is True        # k == j -> dependent


def test_j_dependence_mask_accepts_T_Q_shape():
    qi = np.array([[0, 1], [2, 0]])
    dep = M.j_dependent_mask(CHAIN3, qi, j=1)
    assert dep.shape == (2, 2, 2)


def test_forgetting_bump_uses_independent_queries_only():
    raw = _tiny_raw(T=4, Q=3, d=3)
    dep = M.j_dependent_mask_run(raw, 1)
    err = np.zeros_like(raw["err"], dtype=np.float64)
    err[3][dep[3]] = 10.0            # huge error on dependent queries at t = 4 ...
    err[3][~dep[3]] = 0.5            # ... and 0.25 nMSE on independent ones
    raw["err"] = err.astype(np.float32)
    assert M.forgetting_bump(raw, 4, 1) == pytest.approx(0.25)       # pre-level was 0
    assert math.isnan(M.forgetting_bump(raw, 2, 1))


# --------------------------------------------------------------------------- coverage (metric 4)

def test_coverage_on_fixed_list():
    raw = _tiny_raw(T=1, Q=2, d=3)                       # 4 predictions
    raw["cov50"] = np.array([[[1, 0], [0, 0]]], bool)
    raw["cov90"] = np.array([[[1, 1], [1, 0]]], bool)
    raw["cov99"] = np.array([[[1, 1], [1, 1]]], bool)
    raw["width90"] = np.array([[[1.0, 2.0], [3.0, 4.0]]], np.float32)
    c = M.coverage(raw)
    assert c["cov50"] == pytest.approx(0.25)
    assert c["cov90"] == pytest.approx(0.75)
    assert c["cov99"] == pytest.approx(1.0)
    assert c["width90"] == pytest.approx(2.5)
    assert c["n"] == 4
    # a mask restricts every number to the selected predictions
    m = np.array([[[True, False], [False, True]]])
    cm = M.coverage(raw, m)
    assert cm["cov90"] == pytest.approx(0.5) and cm["width90"] == pytest.approx(2.5) and cm["n"] == 2


def test_episode_kinds():
    raw = _tiny_raw(T=6)
    _add_shift(raw, 3, 1, 0.5)
    k = M.episode_kinds(raw)
    assert k["shift"].tolist() == [0, 0, 1, 0, 0, 0]
    assert k["post_shift"].tolist() == [0, 0, 0, 1, 0, 0]
    assert k["stable"].tolist() == [1, 1, 0, 0, 1, 1]


def test_calibration_table_strata_and_path_equality():
    raw = _tiny_raw(T=2, Q=3, d=3)
    learned = CHAIN3.copy()
    learned[2, 0] = True                                 # extra edge 0 -> 2: path set (0, 2) differs
    raw["learned_adj"][:] = learned
    raw["has_learned_adj"][:] = True
    pe = M.path_equal_mask(raw)
    J = M.column_to_j(raw["query_i"], 3)
    for q in range(3):
        i = int(raw["query_i"][0, q])
        for c in range(2):
            k = int(J[0, q, c])
            assert bool(pe[0, q, c]) == ((i, k) != (0, 2)), (i, k)
    table = M.calibration_table(raw)
    assert set(table) >= {"stable", "shift", "post_shift", "shd_zero", "shd_positive", "path_equal", "path_unequal"}
    assert table["shd_positive"]["n"] == 12 and table["shd_zero"]["n"] == 0
    assert table["path_unequal"]["n"] == 2                # (0, 2) appears once per episode


def _enumerate_paths(adj, i, k):
    """All directed paths i -> k as tuples of nodes (brute force, for the property test)."""
    d, out = adj.shape[0], set()

    def dfs(node, path):
        if node == k:
            out.add(tuple(path))
            return
        for nxt in range(d):
            if adj[nxt, node]:
                dfs(nxt, path + [nxt])
    dfs(i, [i])
    return out


def test_path_equal_matrix_matches_brute_force_enumeration():
    rng = np.random.default_rng([8, 0])
    d = 5

    def random_dag():
        perm, A = rng.permutation(d), np.zeros((d, d), bool)
        for a in range(d):
            for b in range(a + 1, d):
                if rng.random() < 0.5:
                    A[perm[b], perm[a]] = True
        return A

    for _ in range(60):
        A, L = random_dag(), random_dag()
        fast = M.path_equal_matrix(A, L)
        for i in range(d):
            for k in range(d):
                if i != k:
                    assert fast[i, k] == (_enumerate_paths(A, i, k) == _enumerate_paths(L, i, k))


# --------------------------------------------------------------------------- selective risk (metric 5)

def test_risk_coverage_curve_and_aurc_on_three_predictions():
    err2 = np.array([0.0, 1.0, 4.0])
    good = np.array([0.1, 0.2, 0.3])                     # perfect ordering
    bad = np.array([0.3, 0.2, 0.1])                      # reversed
    grid = M.COVERAGE_GRID
    # hand-computed step function: k = ceil(3c) kept predictions
    k = np.ceil(3 * grid).astype(int)
    expected_good = np.where(k == 1, 0.0, np.where(k == 2, 0.5, 5 / 3))
    assert M.risk_coverage_curve(err2, good) == pytest.approx(expected_good)
    assert M.aurc(err2, good) == pytest.approx(np.trapezoid(expected_good, grid) / 0.99)
    assert M.aurc(err2, good) == pytest.approx(M.aurc(err2, err2))      # equals oracle ordering
    assert M.aurc(err2, bad) > M.aurc(err2, good)
    # a constant curve integrates to its value, so random ordering == overall nMSE
    assert M.aurc(np.full(3, 5 / 3), good) == pytest.approx(5 / 3)
    assert M.risk_coverage_curve(err2, good, np.array([1.0]))[0] == pytest.approx(5 / 3)


def test_selective_risk_mistake_recall_precision():
    raw = _tiny_raw(T=1, Q=2, d=3)
    raw["err"] = np.array([[[0.1, 1.0], [0.6, 0.0]]], np.float32)     # mistakes: err² > 0.25 -> 2 of 4
    raw["width90"] = np.array([[[0.5, 2.0], [1.0, 0.1]]], np.float32)
    raw["abstain"] = np.array([[[False, True], [False, False]]])
    sr = M.selective_risk(raw)
    assert sr["mistake_rate"] == pytest.approx(0.5)
    assert sr["mistake_recall"] == pytest.approx(0.5)     # 1 of the 2 mistakes abstained
    assert sr["mistake_precision"] == pytest.approx(1.0)  # the one abstention was a mistake
    assert sr["abstain_rate"] == pytest.approx(0.25)
    assert sr["aurc_random"] == pytest.approx(np.mean([0.01, 1.0, 0.36, 0.0]))
    assert sr["aurc_oracle"] <= sr["aurc"] <= sr["aurc_random"] + 1e-12
    assert sr["risk_at"][1.0] == pytest.approx(sr["aurc_random"])


# --------------------------------------------------------------------------- SHD (metric 6)

def test_shd_hand_case():
    learned = np.zeros((3, 3), bool)
    learned[0, 1] = True                                 # 1 -> 0 (reversed)
    learned[2, 0] = True                                 # 0 -> 2 (extra); 1 -> 2 missing
    assert M.shd(CHAIN3, learned) == 3
    assert M.shd(CHAIN3, CHAIN3) == 0
    raw = _tiny_raw(T=2)
    raw["learned_adj"][1] = learned
    raw["has_learned_adj"][1] = True
    s = M.shd_series(raw)
    assert math.isnan(s[0]) and s[1] == 3


# --------------------------------------------------------------------------- credit assignment (metric 8)

def test_credit_assignment_on_fixed_event_log():
    raw = _tiny_raw(T=20, Q=2, d=3)
    _add_shift(raw, 7, 1, 0.5)               # qualifying
    _add_shift(raw, 13, 2, 0.5)              # qualifying
    _add_shift(raw, 19, 0, 0.01)             # below the m filter: excluded, but still a window
    raw["reset_events"] = np.array([[3, 2],      # false alarm (stable period)
                                    [7, 1],      # correct, delay 0
                                    [13, 0],     # wrong mechanism in shift 13's window
                                    [14, 2]],    # correct for shift 13, delay 1
                                   np.int32)
    ca = M.credit_assignment(raw)
    assert ca["n_shifts"] == 2 and ca["n_shifts_excluded"] == 1 and ca["n_resets"] == 4
    assert [p["outcome"] for p in ca["per_shift"]] == ["correct", "correct"]
    assert ca["delays"] == [0, 1]
    assert ca["accuracy_delay0"] == pytest.approx(0.5)
    assert ca["recall_window"] == pytest.approx(1.0)
    assert ca["misattribution_rate"] == pytest.approx(0.5)          # shift 13 also reset mechanism 0
    assert ca["attribution_precision"] == pytest.approx(0.5)        # (7,1), (14,2) of 4 resets
    assert ca["n_misattributed_resets"] == 1
    assert ca["confusion"] == {"correct": 2, "wrong-mechanism": 0, "miss": 0, "false-alarm": 1}
    # stable episodes: 20 minus the windows {7,8}, {13,14}, {19,20} -> 14 episodes x 3 mechanisms
    assert ca["false_reset_rate"] == pytest.approx(1 / 42)


def test_credit_assignment_miss_and_wrong_mechanism_and_censoring():
    raw = _tiny_raw(T=20, Q=2, d=3)
    _add_shift(raw, 7, 1, 0.5)
    _add_shift(raw, 13, 2, 0.5)
    raw["reset_events"] = np.array([[8, 0]], np.int32)   # only a wrong-mechanism reset for shift 7
    ca = M.credit_assignment(raw)
    assert [p["outcome"] for p in ca["per_shift"]] == ["wrong-mechanism", "miss"]
    assert ca["delays"] == [12 - 7, 20 - 13]              # censored lengths: next shift - 1, T
    assert ca["detected"] == [False, False]
    assert ca["accuracy_delay0"] == 0.0 and ca["attribution_precision"] == 0.0
    assert ca["false_reset_rate"] == 0.0


def test_qualifying_shifts_noise_only_uses_delta_sigma():
    raw = _tiny_raw(T=20)
    _add_shift(raw, 7, 1, 0.0, code=M.CODE_NOISE_ONLY, delta_sigma=0.4)
    _add_shift(raw, 13, 2, 0.0, code=M.CODE_NOISE_ONLY, delta_sigma=0.01)
    kept, excluded = M.qualifying_shifts(raw)
    assert [s.t for s in kept] == [7] and excluded == 1
    assert kept[0].next_shift_t == 13


# --------------------------------------------------------------------------- nMSE (metric 1)

def test_nmse_descendant_split_and_abstention():
    raw = _tiny_raw(T=1, Q=3, d=3)                       # queries i = 0, 1, 2 on the chain
    desc = M.descendant_mask(raw)
    # i = 0: both 1, 2 descendants; i = 1: only 2; i = 2: none
    assert desc[0].tolist() == [[True, True], [False, True], [False, False]]
    err = np.where(desc[0], 1.0, 2.0)
    raw["err"] = err[None].astype(np.float32)
    raw["abstain"][0, 0, 0] = True                       # one abstention on a descendant prediction
    s = M.nmse_summary(raw)
    assert s["nmse_descendant"] == pytest.approx(1.0)
    assert s["nmse_non_descendant"] == pytest.approx(4.0)
    assert s["nmse"] == pytest.approx((3 * 1 + 3 * 4) / 6)
    assert s["abstain_rate"] == pytest.approx(1 / 6)
    assert s["nmse_non_abstained"] == pytest.approx((2 * 1 + 3 * 4) / 5)
    assert M.nmse_series(raw).shape == (1,)


# --------------------------------------------------------------------------- bootstrap and Holm

def test_bootstrap_ci_and_pvalue_small_input():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = M.bootstrap_ci(v, n_boot=500, seed=1)
    assert ci["mean"] == pytest.approx(3.0) and 1.0 <= ci["lo"] <= 3.0 <= ci["hi"] <= 5.0
    assert M.bootstrap_pvalue(v, 0.0, ">", n_boot=500, seed=1) == pytest.approx(1 / 501)
    assert M.bootstrap_pvalue(v, 10.0, ">", n_boot=500, seed=1) == pytest.approx(1.0)
    assert M.bootstrap_pvalue(v, 10.0, "<", n_boot=500, seed=1) == pytest.approx(1 / 501)
    # deterministic given the seed; NaNs are dropped
    assert M.bootstrap_ci(v, 200, 3) == M.bootstrap_ci(v + [np.nan], 200, 3)
    assert M.bootstrap_ci([np.nan])["n"] == 0


def test_paired_bootstrap_identical_agents():
    a = np.array([0.3, 0.5, 0.2])
    pb = M.paired_bootstrap(a, a, n_boot=100)
    assert pb["mean"] == 0.0 and pb["lo"] == 0.0 and pb["hi"] == 0.0
    with pytest.raises(ValueError):
        M.paired_bootstrap(a, a[:2])


def test_holm_hand_case():
    out = M.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")}, alpha=0.05)
    assert out["a"]["p_adj"] == pytest.approx(0.03) and out["a"]["reject"]
    assert out["c"]["p_adj"] == pytest.approx(0.06) and not out["c"]["reject"]
    assert out["b"]["p_adj"] == pytest.approx(0.06) and not out["b"]["reject"]   # monotone
    assert not out["d"]["reject"] and math.isnan(out["d"]["p_adj"])


def test_cluster_bootstrap_pooled_mean():
    clusters = [[1.0, 1.0], [3.0], [2.0, 2.0, 2.0]]
    stat = lambda cl: float(np.mean([x for c in cl for x in c]))  # noqa: E731
    cb = M.cluster_bootstrap(clusters, stat, n_boot=300, seed=2)
    assert cb["point"] == pytest.approx(11 / 6) and cb["n_seeds"] == 3
    assert 1.0 <= cb["lo"] <= cb["point"] <= cb["hi"] <= 3.0


def test_evaluate_primaries_kinds_and_holm():
    values = {"P2": [-0.5, -0.4, -0.6, -0.5],            # mean < 0: should reject
              "P8": [-1.0, -2.0, -1.5, -1.2],            # mean > 0 predicted: should not
              "P3": [0.0, 0.01, -0.01, 0.0],             # inside [-0.05, 0.05]
              "P1": [-1.0, -1.0, -1.0, 1.0]}             # 75 % inside the band, threshold 80 %
    res = M.evaluate_primaries(values, n_boot=300, seed=0)
    assert res["P2"]["held"] and not res["P8"]["held"] and res["P3"]["held"]
    assert res["P1"]["fraction"] == pytest.approx(0.75) and not res["P1"]["held"]
    assert all("p_adj" in r and "ci" in r for r in res.values())


def test_primaries_report_reasons_when_not_computable():
    raw = _tiny_raw(T=6)
    v, why = M.primary_P2({"mech-full": raw})
    assert math.isnan(v) and "missing" in why
    v, why = M.primary_P6({"mech-full": raw})
    assert math.isnan(v) and "hidden" in why
    v, why = M.primary_P7({"mech-full": raw})
    assert math.isnan(v) and "no visible shift" in why


# --------------------------------------------------------------------------- end to end: oracle agent

def test_oracle_raw_has_zero_nmse_and_full_coverage():
    """SPEC test 8's integration check.  `World` refuses T < 55 (its schedule lives in
    [7, 55]), so the oracle runs the full T = 60 (well under a second) and the first 8
    episodes are checked explicitly on top of the whole run."""
    from world import World
    from agents import build_agent
    import protocol

    world = World(seed=1000)
    raw = protocol.run_agent(world, build_agent("oracle", world, 1000, None))
    T = M.dims(raw)[0]
    assert T == 60

    first8 = M.episode_prediction_mask(raw, M._episode_range(T, 1, 8))
    assert M.nmse_summary(raw, first8)["nmse"] == 0.0
    assert M.coverage(raw, first8)["cov90"] == 1.0

    s = M.nmse_summary(raw)
    assert s["nmse"] == 0.0 and s["nmse_descendant"] == 0.0 and s["nmse_non_descendant"] == 0.0
    assert s["abstain_rate"] == 0.0
    c = M.coverage(raw)
    assert c["cov50"] == 1.0 and c["cov90"] == 1.0 and c["cov99"] == 1.0
    assert c["width90"] == 0.0
    assert np.all(M.nmse_series(raw) == 0.0)
    # the metrics that read the shift log and events run on a real raw file
    shifts, excluded = M.qualifying_shifts(raw)
    assert len(shifts) + excluded == 8
    ca = M.credit_assignment(raw)
    assert ca["n_resets"] == 0 and ca["false_reset_rate"] == 0.0
    sr = M.selective_risk(raw)
    assert sr["aurc"] == 0.0 and sr["mistake_rate"] == 0.0
    table = M.calibration_table(raw)
    assert table["stable"]["cov90"] == 1.0 and "shd_zero" not in table   # oracle learns no DAG
    for sh in shifts:
        assert M.recovery_time(M.nmse_series(raw), sh.t, sh.next_shift_t) == (0, True)
