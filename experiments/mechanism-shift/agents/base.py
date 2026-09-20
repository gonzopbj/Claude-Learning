"""Agent base class and shared constants (INTERFACES.md section 2; SPEC.md "Episode protocol").

Every agent in the testbed subclasses `Agent` and overrides the hooks that the episode loop
in `protocol.py` calls.  The hooks are listed here in the order the protocol calls them
within one episode:

    observe(X_obs, t)                      step 3
    detect_and_reset()                     step 4
    detector_stats()                       step 4 (right after detect_and_reset)
    refit()                                step 5
    choose_free_interventions(n_free, t)   step 6, round 2 (not in warm-up episodes 1-3)
    score_before_reveal(i, v, row)         step 6, once per interventional sample, BEFORE
    receive_intervention(i, v, row)        step 6, once per interventional sample
    update_structure(t)                    step 7
    answer(query_i, query_v)               step 8
    learned_adjacency()                    step 8
    internal_objectives()                  step 8

This module must not import `world` (SPEC test 3): agents only ever see what the protocol
hands them.
"""

from __future__ import annotations

import math

import numpy as np

# Interventional values are cycled per variable through this list (SPEC step 6, round 1).
VALUE_CYCLE = (2.0, -2.0, 1.0, -1.0)
ALLOWED_VALUES = frozenset(VALUE_CYCLE)

# Quantile levels returned by answer(), in this fixed order (SPEC "What the agent is asked").
QUANTILE_LEVELS = (0.005, 0.05, 0.25, 0.75, 0.95, 0.995)

# Episodes 1..WARMUP_EPISODES spend the whole budget round-robin (SPEC step 6, round 2).
WARMUP_EPISODES = 3


class ValueCycle:
    """Per-variable value cycle with a position that persists across episodes.

    One instance lives on each agent (`agent.cycle`).  The protocol uses `round_robin()` for
    the floor; agents use `next_value(i)` for the values of their free-budget samples, so
    every sample on variable i - floor or free - takes the next value in i's cycle.
    """

    def __init__(self, d: int):
        self.d = int(d)
        self.pos = np.zeros(self.d, dtype=np.int64)  # cycle position of each variable
        self.next_var = 0                             # next variable in round-robin order

    def next_value(self, i: int) -> float:
        i = int(i)
        v = VALUE_CYCLE[self.pos[i] % len(VALUE_CYCLE)]
        self.pos[i] += 1
        return v

    def round_robin(self) -> tuple[int, float]:
        """Next (i, v) of the round-robin floor: variables 0, 1, ..., d-1, 0, 1, ...

        d*F consecutive calls give exactly F samples per variable from any starting point,
        which is what SPEC step 6 round 1 asks for.
        """
        i = self.next_var
        self.next_var = (i + 1) % self.d
        return i, self.next_value(i)


class Agent:
    """Base class.  Subclasses override the hooks; the defaults below are the ones that make
    sense for every agent (no detector, no structure, no internal objectives)."""

    # Class-level flags read by the protocol and the registry.  Variants may override them
    # in __init__ (e.g. MechAgent sets uses_floor from its `floor` kwarg).
    is_oracle: bool = False          # receives OracleAccess instead of WorldView
    intervenes: bool = True          # False: the protocol skips step 6 for this agent
    uses_floor: bool = True          # False only for mech-full-nofloor
    learns_structure: bool = False   # SHD is computed for these
    has_detector: bool = False       # detector_stats() is meaningful
    never_abstains: bool = False     # tau forced to +inf

    def __init__(self, name: str, agent_id: int, d: int, seed: int,
                 constants: dict | None, access) -> None:
        self.name = str(name)
        self.agent_id = int(agent_id)
        self.d = int(d)
        self.seed = int(seed)
        self.access = access                       # WorldView or OracleAccess
        self.constants = dict(constants or {})
        # The ONLY source of agent-internal randomness (SPEC "RNG streams").
        self.rng = np.random.default_rng([self.seed, self.agent_id, 5])
        self.cycle = ValueCycle(self.d)
        # Tuned constants (INTERFACES.md section 5); missing -> documented defaults.
        tau = self.constants.get("tau", math.inf)
        self.tau = math.inf if (self.never_abstains or tau is None) else float(tau)
        lam = self.constants.get("lambda", math.inf)
        self.lam = math.inf if lam is None else float(lam)
        self.window = self.constants.get("W")      # None -> agent-specific default
        self.gamma = self.constants.get("gamma")   # None -> agent-specific default

    # ------------------------------------------------------------------ step 3
    def observe(self, X_obs: np.ndarray, t: int) -> None:
        """Receive the (n_obs, d) observational batch of episode t."""
        raise NotImplementedError

    # ------------------------------------------------------------------ step 4
    def detect_and_reset(self) -> list[int]:
        """Apply the agent's reset rule; return the mechanisms whose buffer was truncated."""
        return []

    def detector_stats(self) -> dict | None:
        """{"g": (d,), "glr": (d,)} statistics of this episode before the reset zeroed g,
        NaN where the N_min guard skipped the mechanism; None for agents without a detector."""
        return None

    # ------------------------------------------------------------------ step 5
    def refit(self) -> None:
        """Append the observational batch to the buffers and refit every posterior."""
        raise NotImplementedError

    # ------------------------------------------------------------------ step 6
    def choose_free_interventions(self, n_free: int, t: int) -> list[tuple[int, float]]:
        """Return exactly n_free (i, v) pairs; v from self.cycle.next_value(i)."""
        if self.intervenes:
            raise NotImplementedError
        return []

    def score_before_reveal(self, i: int, v: float, row: np.ndarray) -> float | None:
        """Self log-score of the realized row under the current posterior (SPEC "Self
        log-score"); None for agents that do not define one."""
        return None

    def receive_intervention(self, i: int, v: float, row: np.ndarray) -> None:
        """Append the (d,) row do(X_i = v) to the buffers of all mechanisms j != i and keep
        the posteriors consistent with the buffers."""
        if self.intervenes:
            raise NotImplementedError

    # ------------------------------------------------------------------ step 7
    def update_structure(self, t: int) -> list[int]:
        """Re-estimate structure; return mechanisms whose parent set changed."""
        return []

    # ------------------------------------------------------------------ step 8
    def answer(self, query_i: np.ndarray, query_v: np.ndarray) -> dict:
        """Return {"point": (Q, d), "quantiles": (Q, d, 6)}; entries at j == i are ignored."""
        raise NotImplementedError

    def learned_adjacency(self) -> np.ndarray | None:
        """(d, d) bool, children in rows, or None."""
        return None

    def internal_objectives(self) -> dict | None:
        """{"self_score", "var_obj_before", "var_obj_after"} or None."""
        return None
