"""Agent registry (INTERFACES.md section 3; SPEC.md "Agents" table).

Maps the 13 SPEC agent names plus `mech-full-diag` to (module, class, kwargs).  Modules are
imported lazily so that `import agents` works before every implementer file exists.
Nothing here imports `world`.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field

from .base import (ALLOWED_VALUES, QUANTILE_LEVELS, VALUE_CYCLE, WARMUP_EPISODES,  # noqa: F401
                   Agent, ValueCycle)

# agent_id enters the agent RNG key [seed, agent_id, 5]; fixed forever by this table.
AGENT_IDS: dict[str, int] = {
    "marginal-mean": 0,
    "obs-window": 1,
    "obs-cumulative": 2,
    "int-pairwise": 3,
    "mech-full": 4,
    "mech-random": 5,
    "mech-full-nofloor": 6,
    "mech-reset-all": 7,
    "mech-no-detect": 8,
    "mech-oracle-detect": 9,
    "mech-overconfident": 10,
    "oracle-structure": 11,
    "oracle": 12,
    "mech-full-diag": 13,
}


@dataclass(frozen=True)
class AgentSpec:
    module: str                     # module inside this package, e.g. "mech"
    cls: str                        # class name inside that module
    kwargs: dict = field(default_factory=dict)
    is_oracle: bool = False         # gets world.oracle_access() instead of world.view()
    hidden_only: bool = False       # only runs in the hidden-confounder variant


# The flag combination of the proposed system; variants change one flag each.
_MECH_FULL = dict(structure="learned", reset_rule="per-mechanism", selection="active",
                  floor=True, prior="default")

REGISTRY: dict[str, AgentSpec] = {
    "marginal-mean":      AgentSpec("baselines", "MarginalMeanAgent"),
    "obs-window":         AgentSpec("baselines", "ObsPairwiseAgent", dict(windowed=True)),
    "obs-cumulative":     AgentSpec("baselines", "ObsPairwiseAgent", dict(windowed=False)),
    "int-pairwise":       AgentSpec("baselines", "IntPairwiseAgent"),
    "mech-full":          AgentSpec("mech", "MechAgent", dict(_MECH_FULL)),
    "mech-random":        AgentSpec("mech", "MechAgent", {**_MECH_FULL, "selection": "random"}),
    "mech-full-nofloor":  AgentSpec("mech", "MechAgent", {**_MECH_FULL, "floor": False}),
    "mech-reset-all":     AgentSpec("mech", "MechAgent", {**_MECH_FULL, "reset_rule": "all"}),
    "mech-no-detect":     AgentSpec("mech", "MechAgent", {**_MECH_FULL, "reset_rule": "none"}),
    "mech-oracle-detect": AgentSpec("mech", "MechAgent", {**_MECH_FULL, "reset_rule": "oracle"},
                                    is_oracle=True),
    "mech-overconfident": AgentSpec("mech", "MechAgent", {**_MECH_FULL, "prior": "overconfident"}),
    "oracle-structure":   AgentSpec("mech", "MechAgent", {**_MECH_FULL, "structure": "true"},
                                    is_oracle=True),
    "oracle":             AgentSpec("oracle", "OracleAgent", is_oracle=True),
    "mech-full-diag":     AgentSpec("mech", "MechAgent", {**_MECH_FULL, "diag_pair": True},
                                    is_oracle=True, hidden_only=True),
}

assert set(REGISTRY) == set(AGENT_IDS)

# Agent sets per run (SPEC "Compute budget" table).
MAIN_AGENTS = tuple(n for n in AGENT_IDS if not REGISTRY[n].hidden_only)          # 13 agents
SWEEP_AGENTS = ("mech-full", "mech-random", "int-pairwise")                          # B in {10, 25, 100}
SWEEP_NOFLOOR_BUDGETS = (10, 25)                                                     # mech-full-nofloor
HIDDEN_AGENTS = ("mech-full", "mech-random", "int-pairwise", "obs-cumulative", "oracle-structure",
                 "mech-oracle-detect", "oracle", "mech-full-diag")
KNOB_AGENTS = ("mech-full", "mech-oracle-detect", "oracle-structure", "int-pairwise")


def agent_class(name: str):
    """Import and return the class registered under `name`."""
    spec = REGISTRY[name]
    module = importlib.import_module(f"{__name__}.{spec.module}")
    return getattr(module, spec.cls)


def agent_constants(all_constants: dict | None, name: str) -> dict:
    """Flatten constants.json (INTERFACES.md section 5) to the per-agent dict."""
    if all_constants is None:
        return {}
    out = {"lambda": all_constants.get("lambda", math.inf)}
    out.update(all_constants.get("agents", {}).get(name, {}))
    if out.get("tau") is None:
        out["tau"] = math.inf
    if out.get("lambda") is None:
        out["lambda"] = math.inf
    return out


def build_agent(name: str, world, seed: int, constants: dict | None = None) -> Agent:
    """Construct the agent `name` for `world`, giving oracle agents oracle access and every
    other agent the blind WorldView."""
    spec = REGISTRY[name]
    access = world.oracle_access() if spec.is_oracle else world.view()
    cls = agent_class(name)
    agent = cls(name, AGENT_IDS[name], world.d, seed, constants, access, **spec.kwargs)
    # The registry, not the class, is the authority on who holds oracle access.
    agent.is_oracle = spec.is_oracle
    return agent
