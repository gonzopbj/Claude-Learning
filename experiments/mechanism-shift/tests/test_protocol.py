"""Tests for protocol.py against a tiny fake world and two dummy agents.

Covers: the hook order of SPEC steps 1-8; SPEC test 2 (two agents with different
intervention policies see byte-identical observational batches, query sets, shift schedules
and interventional noise rows); the budget arithmetic table of SPEC "Episode protocol"; the
warm-up rule; the raw .npz schema of INTERFACES.md section 6; the evaluator formulas.

`FakeWorld` is also importable by agent implementers to exercise an agent through the real
protocol without world.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import protocol
from agents.base import ALLOWED_VALUES, VALUE_CYCLE, WARMUP_EPISODES, Agent
from protocol import floor_size, run_agent, save_raw


# --------------------------------------------------------------------------- fake world

@dataclass(frozen=True)
class ShiftRecord:
    t: int
    j: int
    shift_type: str
    m: float
    delta: float
    delta_sigma: float


@dataclass(frozen=True)
class WorldView:
    d: int
    B: int
    T: int
    n_obs: int
    n_queries: int


class FakeOracleAccess:
    def __init__(self, world):
        self._world = world
        self.view = world.view()
        self.d, self.B, self.T, self.n_obs, self.n_queries = (
            world.d, world.B, world.T, world.n_obs, world.n_queries)

    def true_adjacency(self):
        return self._world.true_adjacency()

    def truth(self, i, v):
        return self._world.truth(i, v)

    def shifted_mechanism(self, t):
        for rec in self._world.schedule:
            if rec.t == t:
                return rec.j
        return -1

    def hidden_pair(self):
        return None


class FakeWorld:
    """A fixed chain 0 -> 1 -> ... -> d-1 with unit weights, intercept 0.5, unit noise, and a
    single scheduled shift (intercept of mechanism `shift_j` flips sign at episode `shift_t`).
    Implements every World method that protocol.run_agent touches, with the SPEC's keyed RNG
    streams, and records which noise row served each interventional sample."""

    def __init__(self, seed=0, d=6, T=6, B=10, n_obs=20, n_queries=7, shift_t=5, shift_j=2):
        self.seed, self.d, self.d_full, self.T, self.B = seed, d, d, T, B
        self.n_obs, self.n_queries = n_obs, n_queries
        self.shift_type, self.hetero, self.hidden, self.shifts_enabled = "large", 0, False, True
        self.W = np.zeros((d, d))
        for j in range(1, d):
            self.W[j, j - 1] = 1.0                       # children in rows
        self.b0 = np.full(d, 0.5)
        self.b = self.b0.copy()
        self.sigma = np.ones(d)
        self.shift_t, self.shift_j = shift_t, shift_j
        self.schedule = [ShiftRecord(shift_t, shift_j, "large", 0.3, 0.0, 0.0)]
        self._t = 0
        self.noise_rows_used = {}                        # (t, k) -> noise row used

    # -- episode protocol
    def apply_shift(self, t):
        if t != self._t + 1:
            raise ValueError("apply_shift must be called for t = 1, 2, ... in order")
        self._t = t
        if t == self.shift_t:
            self.b = self.b0.copy()
            self.b[self.shift_j] = -0.5
            return self.schedule[0]
        return None

    def _push(self, eps, clamp=None):
        x = np.zeros(self.d)
        for j in range(self.d):                          # chain order is topological
            x[j] = self.W[j] @ x + self.b[j] + self.sigma[j] * eps[j]
            if clamp is not None and j == clamp[0]:
                x[j] = clamp[1]
        return x

    def observational_batch(self, t):
        eps = np.random.default_rng([self.seed, t, 2]).standard_normal((self.n_obs, self.d))
        return np.stack([self._push(e) for e in eps])

    def interventional_noise(self, t):
        return np.random.default_rng([self.seed, t, 3]).standard_normal((self.B, self.d))

    def query_set(self, t):
        rng = np.random.default_rng([self.seed, t, 4])
        qi = rng.integers(0, self.d, size=self.n_queries)
        qv = rng.uniform(-2.0, 2.0, size=self.n_queries)
        return qi, qv

    def intervene(self, t, k, i, v):
        eps = self.interventional_noise(t)[k]
        self.noise_rows_used[(t, k)] = eps.copy()
        return self._push(eps, (i, v))

    # -- ground truth
    def truth(self, i, v):
        Wi = self.W.copy()
        Wi[i] = 0.0
        bi = self.b.copy()
        bi[i] = v                                        # (I - W_i)^-1 (b_i + v e_i)
        return np.linalg.solve(np.eye(self.d) - Wi, bi)

    def var_ref(self):
        theta = np.linalg.inv(np.eye(self.d) - self.W)
        return np.diag(theta @ np.diag(self.sigma ** 2) @ theta.T)

    def sigma_ref(self):
        return np.sqrt(self.var_ref())

    def mean_obs(self):
        return np.linalg.solve(np.eye(self.d) - self.W, self.b)

    def F_obs(self):
        return 0.123

    def confounding(self):
        return None

    def true_adjacency(self):
        return self.W != 0

    def true_W(self):
        return self.W.copy()

    def true_b(self):
        return self.b.copy()

    def true_sigma(self):
        return self.sigma.copy()

    def hidden_pair(self):
        return None

    def permutation(self):
        return np.arange(self.d)

    def view(self):
        return WorldView(self.d, self.B, self.T, self.n_obs, self.n_queries)

    def oracle_access(self):
        return FakeOracleAccess(self)


# --------------------------------------------------------------------------- dummy agents

QUANTS = np.array([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])


class RecordingAgent(Agent):
    """Logs every hook call and everything it is handed; answers a constant interval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []
        self.obs_seen = []          # (t, X_obs copy)
        self.rows_seen = []         # (i, v, row copy)
        self.queries_seen = []      # (qi copy, qv copy)
        self.free_calls = []        # (t, n_free)

    def observe(self, X_obs, t):
        self.log.append("observe")
        self.t = t
        self.obs_seen.append((t, np.array(X_obs, copy=True)))

    def detect_and_reset(self):
        self.log.append("detect_and_reset")
        return []

    def detector_stats(self):
        self.log.append("detector_stats")
        return None

    def refit(self):
        self.log.append("refit")

    def free_targets(self, n_free):
        raise NotImplementedError

    def choose_free_interventions(self, n_free, t):
        self.log.append("choose_free")
        self.free_calls.append((t, n_free))
        return [(int(i), self.cycle.next_value(int(i))) for i in self.free_targets(n_free)]

    def score_before_reveal(self, i, v, row):
        self.log.append("score")
        return None

    def receive_intervention(self, i, v, row):
        self.log.append("receive")
        self.rows_seen.append((i, v, np.array(row, copy=True)))

    def update_structure(self, t):
        self.log.append("update_structure")
        return []

    def answer(self, query_i, query_v):
        self.log.append("answer")
        self.queries_seen.append((np.array(query_i, copy=True), np.array(query_v, copy=True)))
        Q = len(query_i)
        return {"point": np.zeros((Q, self.d)),
                "quantiles": np.broadcast_to(QUANTS, (Q, self.d, 6)).copy()}

    def learned_adjacency(self):
        self.log.append("learned_adjacency")
        return None

    def internal_objectives(self):
        self.log.append("internal_objectives")
        return None

    def episodes(self):
        """Split the log into per-episode hook sequences (each episode starts with observe)."""
        out, cur = [], None
        for h in self.log:
            if h == "observe":
                cur = []
                out.append(cur)
            cur.append(h)
        return out


class RandomTargetAgent(RecordingAgent):
    def free_targets(self, n_free):
        return self.rng.integers(0, self.d, size=n_free)


class FixedTargetAgent(RecordingAgent):
    """Always targets variable 0; also exercises the event, detector and objective channels."""
    has_detector = True
    learns_structure = True

    def free_targets(self, n_free):
        return np.zeros(n_free, dtype=int)

    def detect_and_reset(self):
        self.log.append("detect_and_reset")
        return [1] if self.t == 4 else []

    def detector_stats(self):
        self.log.append("detector_stats")
        g = np.full(self.d, np.nan)
        g[0] = float(self.t)
        return {"g": g, "glr": np.zeros(self.d)}

    def score_before_reveal(self, i, v, row):
        self.log.append("score")
        return 0.5

    def update_structure(self, t):
        self.log.append("update_structure")
        return [2, 3] if t == 2 else []

    def learned_adjacency(self):
        self.log.append("learned_adjacency")
        return np.eye(self.d, k=-1, dtype=bool)

    def internal_objectives(self):
        self.log.append("internal_objectives")
        return {"self_score": 0.5, "var_obj_before": 2.0, "var_obj_after": 1.0}


def make(cls, world, agent_id, constants=None):
    return cls(cls.__name__, agent_id, world.d, world.seed, constants, world.view())


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("B,F,n_floor,n_free", [(10, 1, 6, 4), (25, 3, 18, 7),
                                                 (50, 5, 30, 20), (100, 9, 54, 46)])
def test_floor_arithmetic_matches_spec(B, F, n_floor, n_free):
    assert floor_size(B, 6) == (F, n_floor, n_free)


def test_floor_rejects_budget_below_d():
    with pytest.raises(ValueError):
        floor_size(5, 6)


def test_hooks_called_in_spec_order():
    world = FakeWorld(T=5, B=10)
    agent = make(FixedTargetAgent, world, agent_id=1)
    run_agent(world, agent)
    eps = agent.episodes()
    assert len(eps) == 5
    head = ["observe", "detect_and_reset", "detector_stats", "refit"]
    tail = ["update_structure", "answer", "learned_adjacency", "internal_objectives"]
    sample = ["score", "receive"]                     # score BEFORE the row is received
    # Warm-up episodes 1-3: all B samples round-robin, no free choice at all.
    for t in range(1, WARMUP_EPISODES + 1):
        assert eps[t - 1] == head + sample * 10 + tail
    # Episodes 4+: floor (dF = 6), then the agent chooses the free 4, then those samples.
    for t in range(WARMUP_EPISODES + 1, 6):
        assert eps[t - 1] == head + sample * 6 + ["choose_free"] + sample * 4 + tail
    assert agent.free_calls == [(4, 4), (5, 4)]


def test_two_policies_see_identical_streams():
    """SPEC test 2: byte-identical batches, queries, schedules and noise rows."""
    wa, wb = FakeWorld(seed=3, T=6, B=10), FakeWorld(seed=3, T=6, B=10)
    a = make(RandomTargetAgent, wa, agent_id=0)
    b = make(FixedTargetAgent, wb, agent_id=1)
    ra, rb = run_agent(wa, a), run_agent(wb, b)

    # the two policies really differ
    assert not np.array_equal(ra["intervention_i"][3:], rb["intervention_i"][3:])
    # observational batches
    assert len(a.obs_seen) == len(b.obs_seen) == 6
    for (ta, Xa), (tb, Xb) in zip(a.obs_seen, b.obs_seen):
        assert ta == tb and Xa.tobytes() == Xb.tobytes()
    # query sets
    for (ia, va), (ib, vb) in zip(a.queries_seen, b.queries_seen):
        assert ia.tobytes() == ib.tobytes() and va.tobytes() == vb.tobytes()
    assert ra["query_i"].tobytes() == rb["query_i"].tobytes()
    # shift schedule
    assert ra["shift_j"].tobytes() == rb["shift_j"].tobytes()
    assert list(ra["shift_j"]) == [-1, -1, -1, -1, 2, -1]
    # interventional noise: sample k of episode t used the same row for both agents
    assert wa.noise_rows_used.keys() == wb.noise_rows_used.keys()
    assert len(wa.noise_rows_used) == 6 * 10
    for key in wa.noise_rows_used:
        assert wa.noise_rows_used[key].tobytes() == wb.noise_rows_used[key].tobytes()
    # and the row handed to the agent is the row of the episode-t SCM for that (target,
    # value, noise row): replay the schedule on a fresh world so the shift is respected
    replay = FakeWorld(seed=3, T=6, B=10)
    assert len(a.rows_seen) == 6 * 10
    for n, (i, v, row) in enumerate(a.rows_seen):
        t, k = n // 10 + 1, n % 10
        if k == 0:
            replay.apply_shift(t)
        assert row[i] == v
        assert np.allclose(row, replay._push(wa.noise_rows_used[(t, k)], (i, v)))


def test_warmup_floor_and_value_cycle():
    world = FakeWorld(T=5, B=10)
    agent = make(FixedTargetAgent, world, agent_id=1)
    R = run_agent(world, agent)
    d, F, n_floor, n_free = 6, 1, 6, 4
    ii, vv = R["intervention_i"], R["intervention_v"]
    # warm-up: whole budget round-robin -> per-variable counts differ by at most one
    for r in range(WARMUP_EPISODES):
        counts = np.bincount(ii[r], minlength=d)
        assert counts.max() - counts.min() <= 1
    # episodes 4, 5: exactly F floor samples per variable, then the agent's free targets (0)
    for r in range(WARMUP_EPISODES, 5):
        assert list(np.bincount(ii[r, :n_floor], minlength=d)) == [F] * d
        assert list(ii[r, n_floor:]) == [0] * n_free
    # every value in the allowed set, and per variable the values follow the cycle in order
    assert set(np.unique(vv)) <= ALLOWED_VALUES
    for i in range(d):
        seq = list(vv.ravel()[ii.ravel() == i])
        assert seq == [VALUE_CYCLE[n % 4] for n in range(len(seq))]
    # the floor sequence persists across episodes: episode 4 starts where episode 3 stopped
    flat = ii[:, :].ravel()
    rr = list(flat[:3 * 10]) + list(ii[3, :n_floor]) + list(ii[4, :n_floor])
    assert rr == [n % d for n in range(len(rr))]


def test_nofloor_and_non_intervening_flags():
    class NoFloor(FixedTargetAgent):
        uses_floor = False

    world = FakeWorld(T=4, B=10)
    agent = make(NoFloor, world, agent_id=6)
    R = run_agent(world, agent)
    assert agent.free_calls == [(4, 10)]                 # all B on the agent's rule after warm-up
    assert list(R["intervention_i"][3]) == [0] * 10

    class Passive(RecordingAgent):
        intervenes = False

    world = FakeWorld(T=4, B=10)
    agent = make(Passive, world, agent_id=0)
    R = run_agent(world, agent)
    assert "score" not in agent.log and "receive" not in agent.log
    assert np.all(R["intervention_i"] == -1) and np.all(np.isnan(R["sample_score"]))


def test_bad_free_plan_is_rejected():
    class Greedy(FixedTargetAgent):
        def choose_free_interventions(self, n_free, t):
            return [(0, 2.0)] * (n_free + 1)

    class BadValue(FixedTargetAgent):
        def choose_free_interventions(self, n_free, t):
            return [(0, 0.5)] * n_free

    for cls in (Greedy, BadValue):
        world = FakeWorld(T=4, B=10)
        with pytest.raises(ValueError):
            run_agent(world, make(cls, world, agent_id=1))


def test_raw_schema_and_evaluator(tmp_path):
    T, B, Q, d = 6, 10, 7, 6
    world = FakeWorld(seed=1, T=T, B=B, n_queries=Q)
    agent = make(FixedTargetAgent, world, agent_id=1, constants={"tau": 0.9, "lambda": 3.0})
    R = run_agent(world, agent)
    path = tmp_path / protocol.raw_filename(agent.name, world.seed)
    save_raw(path, R)
    Z = np.load(path, allow_pickle=False)

    expected = {
        # per-episode world state
        "episode_t": (np.int32, (T,)), "shift_j": (np.int32, (T,)),
        "shift_type_code": (np.int8, (T,)), "shift_m": (np.float32, (T,)),
        "shift_delta": (np.float32, (T,)), "shift_delta_sigma": (np.float32, (T,)),
        "true_adj": (np.bool_, (T, d, d)), "true_W": (np.float32, (T, d, d)),
        "true_b": (np.float32, (T, d)), "true_sigma": (np.float32, (T, d)),
        "sd_ref": (np.float32, (T, d)), "F_obs": (np.float32, (T,)),
        "conf_c": (np.float32, (T,)), "conf_F": (np.float32, (T,)),
        # per-prediction
        "err": (np.float32, (T, Q, d - 1)), "width50": (np.float32, (T, Q, d - 1)),
        "width90": (np.float32, (T, Q, d - 1)), "width99": (np.float32, (T, Q, d - 1)),
        "width90_raw": (np.float32, (T, Q, d - 1)), "cov50": (np.bool_, (T, Q, d - 1)),
        "cov90": (np.bool_, (T, Q, d - 1)), "cov99": (np.bool_, (T, Q, d - 1)),
        "abstain": (np.bool_, (T, Q, d - 1)),
        # per-query / per-sample
        "query_i": (np.int32, (T, Q)), "query_v": (np.float32, (T, Q)),
        "intervention_i": (np.int32, (T, B)), "intervention_v": (np.float32, (T, B)),
        "sample_score": (np.float32, (T, B)),
        # events and agent state
        "reset_events": (np.int32, (1, 2)), "struct_refit_events": (np.int32, (2, 2)),
        "learned_adj": (np.bool_, (T, d, d)), "has_learned_adj": (np.bool_, (T,)),
        "g_stat": (np.float32, (T, d)), "glr_stat": (np.float32, (T, d)),
        "self_score": (np.float32, (T,)), "var_obj_before": (np.float32, (T,)),
        "var_obj_after": (np.float32, (T,)),
        # scalars
        "agent_id": (np.int32, ()), "seed": (np.int32, ()), "d": (np.int32, ()),
        "T": (np.int32, ()), "B": (np.int32, ()), "n_obs": (np.int32, ()), "Q": (np.int32, ()),
        "hetero": (np.int32, ()), "hidden": (np.bool_, ()), "shifts_enabled": (np.bool_, ()),
        "learns_structure": (np.bool_, ()), "has_detector": (np.bool_, ()),
        "intervenes": (np.bool_, ()), "uses_floor": (np.bool_, ()), "is_oracle": (np.bool_, ()),
        "tau": (np.float32, ()), "wall_time_s": (np.float32, ()),
        "hidden_A": (np.int32, ()), "hidden_B": (np.int32, ()),
    }
    for key, (dtype, shape) in expected.items():
        assert key in Z.files, key
        assert Z[key].dtype == dtype, (key, Z[key].dtype)
        assert Z[key].shape == shape, (key, Z[key].shape)
    for key in ("agent_name", "shift_type"):
        assert Z[key].dtype.kind == "U" and Z[key].shape == ()
    assert set(Z.files) == set(expected) | {"agent_name", "shift_type"}

    # events and shift fields
    assert Z["reset_events"].tolist() == [[4, 1]]
    assert Z["struct_refit_events"].tolist() == [[2, 2], [2, 3]]
    assert Z["shift_j"][4] == 2 and Z["shift_type_code"][4] == 0
    assert np.isclose(Z["shift_m"][4], 0.3) and np.isnan(Z["shift_m"][0])
    assert np.isclose(Z["true_b"][4, 2], -0.5) and np.isclose(Z["true_b"][3, 2], 0.5)
    assert np.all(Z["true_adj"][0] == np.eye(d, k=-1, dtype=bool))
    assert np.all(Z["g_stat"][:, 0] == np.arange(1, T + 1)) and np.all(np.isnan(Z["g_stat"][:, 1]))
    assert np.all(Z["self_score"] == 0.5) and np.all(Z["sample_score"] == 0.5)
    assert np.all(Z["has_learned_adj"]) and float(Z["tau"]) == np.float32(0.9)
    assert str(Z["agent_name"][()]) == "FixedTargetAgent"

    # evaluator: point 0, quantiles QUANTS, against the fake world's exact truth
    world2 = FakeWorld(seed=1, T=T, B=B, n_queries=Q)
    for t in range(1, T + 1):
        world2.apply_shift(t)
        qi, qv = world2.query_set(t)
        sd = world2.sigma_ref()
        truth = np.stack([world2.truth(int(i), float(v)) for i, v in zip(qi, qv)])
        y = protocol.select_off_target(truth, qi)
        s = protocol.select_off_target(np.broadcast_to(sd, (Q, d)), qi)
        r = t - 1
        assert np.allclose(Z["err"][r], -y / s, rtol=1e-5)
        assert np.array_equal(Z["cov50"][r], np.abs(y) <= 0.25)
        assert np.array_equal(Z["cov90"][r], np.abs(y) <= 0.5)
        assert np.array_equal(Z["cov99"][r], np.abs(y) <= 1.0)
        assert np.allclose(Z["width90_raw"][r], 1.0) and np.allclose(Z["width90"][r], 1.0 / s)
        assert np.allclose(Z["width50"][r], 0.5 / s) and np.allclose(Z["width99"][r], 2.0 / s)
        assert np.all(Z["abstain"][r])                    # raw width 1.0 > tau 0.9
        # column c of a per-prediction array is variable j = c + (c >= i)
        c = np.arange(d - 1)
        j = c[None, :] + (c[None, :] >= qi[:, None])
        assert np.allclose(Z["sd_ref"][r][j], s)

    # with no tau the agent never abstains
    world3 = FakeWorld(seed=1, T=2, B=B, n_queries=Q)
    R3 = run_agent(world3, make(FixedTargetAgent, world3, agent_id=1))
    assert not np.any(R3["abstain"])


def test_registry_is_consistent():
    import agents
    assert len(agents.REGISTRY) == 14 and len(agents.MAIN_AGENTS) == 13
    assert "mech-full-diag" not in agents.MAIN_AGENTS and "mech-full-diag" in agents.HIDDEN_AGENTS
    assert sorted(agents.AGENT_IDS.values()) == list(range(14))
    oracle_names = {n for n, s in agents.REGISTRY.items() if s.is_oracle}
    assert oracle_names == {"oracle", "oracle-structure", "mech-oracle-detect", "mech-full-diag"}
    consts = {"lambda": 2.5, "agents": {"obs-window": {"W": 3, "tau": 0.7}, "oracle": {"tau": None}}}
    assert agents.agent_constants(consts, "obs-window") == {"lambda": 2.5, "W": 3, "tau": 0.7}
    assert agents.agent_constants(consts, "oracle") == {"lambda": 2.5, "tau": np.inf}
    assert agents.agent_constants(consts, "mech-full") == {"lambda": 2.5, "tau": np.inf}
    assert agents.agent_constants(None, "mech-full") == {}
