# Mechanism-Shift Testbed — integration interfaces

This document is the contract between the five modules that are written in parallel:
`world.py`, `agents/{baselines,mech,oracle}.py`, `metrics.py`, `run.py`/`calibrate.py`,
`report.py`. The architect owns `agents/__init__.py`, `agents/base.py`, `protocol.py` and
`tests/test_protocol.py`, which are already written against this document. Everything below
is normative; where the SPEC is ambiguous the chosen reading is stated and marked
**Decision**. Section references ("SPEC §Episode protocol") point into `SPEC.md`.

Contents

0. Conventions (indices, episodes, coordinates, dtypes)
1. `world.py` — `World`, `ShiftRecord`, `WorldView`, `OracleAccess`
2. `agents/base.py` — `Agent` hooks, flags, `ValueCycle`
3. `agents/__init__.py` — registry, agent ids, constructor kwargs per agent
4. `protocol.py` — the episode loop, hook order, evaluator formulas
5. `constants.json` — schema and the refusal rule for evaluation seeds
6. Raw `.npz` schema — every field, dtype, shape, meaning
7. `run.py`, `calibrate.py`, `metrics.py`, `report.py` — contracts and ownership
8. Fake-world contract used by `tests/test_protocol.py`

---

## 0. Conventions

| Symbol | Meaning | Value |
|---|---|---|
| `d` | number of **visible** variables | 6 |
| `d_full` | number of variables in the generating SCM | `d` (main) or `d + 1` (hidden) |
| `T` | number of episodes | 60 |
| `B` | interventional budget per episode | 50 (sweep: 10, 25, 100) |
| `n_obs` | observational rows per episode | 200 |
| `Q` | queries per episode | 200 |
| `K` | number of shift episodes | 8 |

**Episodes are labelled `t = 1, …, T`** (SPEC: "episodes 1–3 are warm-up", "shifts in
`[7, 55]`"). **Every per-episode array is stored at row `t − 1`.** The raw `.npz` carries an
explicit `episode_t` vector (`[1, …, T]`) so nobody has to remember this.

**Variable indices** are always the *relabelled* visible indices `0 … d − 1` — the only labels
any agent or the evaluator ever sees. The hidden variable `H` (hidden variant only) has index
`d` (i.e. 6) wherever an index can refer to it (`ShiftRecord.j`, `shift_j`,
`OracleAccess.shifted_mechanism`). No agent-facing array ever has a coordinate for `H`.

**Adjacency / weights: children in rows.** For any `(d, d)` weight or adjacency array `W`,
`W[j, p]` is the weight of edge `p → j`; `adj[j, p] = True` iff `p` is a parent of `j`. The
learned adjacency returned by agents uses the same convention. `Θ = (I − W)⁻¹`, `Θ[k, i]` is
the total effect of `i` on `k`.

**Interventional values** are always in `{−2, −1, 1, 2}` and are drawn from the per-variable
cycle `[2, −2, 1, −1]` (SPEC §Episode protocol step 6). Query values are `v ~ U(−2, 2)`.

**Quantile levels** (fixed order, last axis of every quantile array):
`(0.005, 0.05, 0.25, 0.75, 0.95, 0.995)` — indices `0 … 5`. Widths: `width50 = q[3] − q[2]`,
`width90 = q[4] − q[1]`, `width99 = q[5] − q[0]`.

**"j ≠ i" column layout.** Every per-prediction array has shape `[T, Q, d − 1]`. For query
`q` with target `i = query_i[t, q]`, column `c` refers to variable
`j = c if c < i else c + 1` (the visible variables other than `i`, in ascending order).
`metrics.py` reconstructs `j` from `query_i`; a helper for this is specified in §6.

**RNG.** Only `numpy.random.default_rng(key)` with the SPEC's keyed streams. `np.random.*`
global functions are forbidden everywhere. Keys (SPEC §RNG streams):

| Purpose | Key |
|---|---|
| graph, weights, intercepts, noise scales, relabelling permutation, hidden edge choice | `[seed, 0]` |
| shift schedule and all shift parameters, drawn up front | `[seed, 1]` |
| observational noise for episode `t` (`n_obs × d_full`) | `[seed, t, 2]` |
| interventional noise for episode `t` (`B × d_full`) | `[seed, t, 3]` |
| query set for episode `t` | `[seed, t, 4]` |
| agent-internal randomness | `[seed, agent_id, 5]` |

`t` in the keys is the 1-based episode label.

**Import rules (leakage).** `agents/*.py` must not import `world` (SPEC test 3 greps for it).
`protocol.py` and `metrics.py` do not import `world` either; they only use the objects passed
to them. `world.py` imports nothing from `agents/`. Only `run.py`, `calibrate.py` and the
tests import both sides.

**Floating point.** Computation in float64; the raw `.npz` stores float32 as the SPEC says.

---

## 1. `world.py`

### 1.1 `World`

```python
class World:
    def __init__(self, seed: int, d: int = 6, T: int = 60, B: int = 50, n_obs: int = 200,
                 shift_type: str = "large",        # "large" | "small" | "noise-only"
                 hetero: int = 0,                  # γ_h ∈ {0, 1}
                 hidden: bool = False,
                 n_queries: int = 200,
                 shifts_enabled: bool = True):     # False: λ-calibration runs (SPEC §Tuning)
```

Public read-only attributes (set in `__init__`, never changed): `seed, d, d_full, T, B, n_obs,
n_queries, shift_type, hetero, hidden, shifts_enabled, schedule` (see 1.2).

Construction does **all** of the following before returning, so the whole run is determined
before any agent exists (SPEC §Shift process "pre-drawn per seed"):

1. From stream `[seed, 0]`: ER DAG in a generation order (`p_edge = 3 / (d − 1)`, resample if
   no edges), weights `W[j, p] = s·u`, intercepts `b`, noise scales `σ`, the relabelling
   permutation (applied to everything), and — if `hidden` — the uniformly chosen visible edge
   `A → B`, plus `H`'s parameters. **Decision:** draw the hidden-edge choice and `H`'s parameters
   *after* all visible quantities, so the visible SCM is byte-identical to the main run for the
   same seed (SPEC §Hidden-confounder variant, first sentence).
2. From stream `[seed, 1]`: the `K = 8` shift episodes in `[7, 55]` with minimum gap 5
   (rejection-sample the set), then the parameters of all 8 shifts in episode order. In the
   hidden variant the 4th shift is a `large` shift of `H` (`b_H`, `σ_H` resampled). The stream
   is consumed identically whether or not `shifts_enabled`; with `shifts_enabled=False`,
   `schedule` is `[]` and `apply_shift` never changes the SCM.
3. Builds the `K + 1` successive SCM states and precomputes each `ShiftRecord`'s `m`, `delta`,
   `delta_sigma` exactly from consecutive states (SPEC §Shift process, "computed exactly from
   the two SCMs").

The **current SCM** is the state selected by the last `apply_shift(t)` call (initially the
episode-1 state). Every method below that says "current" refers to it.

```python
    # --- episode protocol (called by protocol.py, in this order, once per episode) ---
    def apply_shift(self, t: int) -> "ShiftRecord | None"
        # Step 1. Makes the SCM for episode t current. Must be called for t = 1, 2, ..., T in
        # order (raise ValueError otherwise). Returns the ShiftRecord whose .t == t, else None.
    def observational_batch(self, t: int) -> np.ndarray          # (n_obs, d) float64
        # Step 2. Noise from stream [seed, t, 2] (n_obs × d_full), pushed through the
        # current SCM; visible coordinates only. Pure: same t → identical array.
    def interventional_noise(self, t: int) -> np.ndarray          # (B, d_full) float64
        # Step 2. The pre-drawn ε array from stream [seed, t, 3]. Pure. Row k is used by
        # intervene(t, k, ...) whatever the agent chose (common random numbers).
    def query_set(self, t: int) -> tuple[np.ndarray, np.ndarray]  # (Q,) int64, (Q,) float64
        # Step 2. From stream [seed, t, 4]: query_i ~ uniform{0..d-1} (visible only, never H),
        # query_v ~ U(-2, 2). Pure. Draw order: all Q targets first, then all Q values.
    def intervene(self, t: int, k: int, i: int, v: float) -> np.ndarray   # (d,) float64
        # Step 6. Row of the mutilated current SCM with X_i := v, using noise row k of
        # interventional_noise(t) for ALL other variables (including H). 0 <= k < B,
        # 0 <= i < d. Pure: no state changes, so an agent's choices cannot affect the world.
        # Visible coordinates only; entry i equals v exactly.
    # --- ground truth for the evaluator (protocol.py) and OracleAccess ---
    def truth(self, i: int, v: float) -> np.ndarray               # (d,) float64
        # E[X | do(X_i = v)] under the current SCM (SPEC §Exact interventional means), entry i
        # equals v. In the hidden variant computed on the 7-node SCM, visible coordinates
        # returned.
    def var_ref(self) -> np.ndarray                               # (d,) float64
        # diag(Σ_ref) of the current SCM, Σ_ref = (I−W)⁻¹ diag(σ²) (I−W)⁻ᵀ, computed with
        # γ_h = 0 regardless of the knob (documented reference scale). Visible coordinates.
    def sigma_ref(self) -> np.ndarray                             # (d,) float64 = sqrt(var_ref())
    def mean_obs(self) -> np.ndarray                              # (d,) float64  μ = (I−W)⁻¹ b
    def F_obs(self) -> float
        # P1 population see-for-do floor for the current SCM:
        # β1 = Σ_ref[j,i]/Σ_ref[i,i]; β0 = μ_j − β1 μ_i; bias(v) = (β0 − c_j(i)) + (β1 − Θ[j,i]) v;
        # F_obs = mean_{i≠j} E_v[bias(v)²] / Var_ref(X_j), E_v[v²] = 4/3, E_v[v] = 0.
        # Hidden variant: Σ_ref, μ from the 7-node covariance restricted to visible coordinates.
    def confounding(self) -> "dict | None"
        # None unless hidden. Else {"A": int, "B": int, "c": float, "F_conf": float} for the
        # current SCM: c = β_A − W[B, A] with β the population OLS of X_B on [1, X_pa_vis(B)]
        # from the 7-node observational covariance; F_conf = c² (4/3) / Var_ref(X_B).
    def true_adjacency(self) -> np.ndarray                        # (d, d) bool, children in rows
        # Visible sub-DAG (constant within a seed). Never exposes H.
    def true_W(self) -> np.ndarray                                # (d, d) float64, current
    def true_b(self) -> np.ndarray                                # (d,) float64, current
    def true_sigma(self) -> np.ndarray                            # (d,) float64, current
    def hidden_pair(self) -> "tuple[int, int] | None"             # (A, B) or None
    def permutation(self) -> np.ndarray                           # (d,) int64, the relabelling
    # --- access objects ---
    def view(self) -> "WorldView"                                 # for non-oracle agents
    def oracle_access(self) -> "OracleAccess"                     # for oracle agents
```

Ancestry/descendant relations are **not** exposed by the world; `metrics.py` derives them from
`true_adj` (transitive closure) in the raw file.

### 1.2 `ShiftRecord`

```python
@dataclass(frozen=True)
class ShiftRecord:
    t: int            # episode label at whose START the mechanism was resampled (7 <= t <= 55)
    j: int            # mechanism index, 0..d-1 visible, or d for the hidden H shift
    shift_type: str   # "large" | "small" | "noise-only"  (the H shift is "large")
    m: float          # effective magnitude E_q[(truth_after − truth_before)²]/Var_ref(X_k),
                      # averaged over i uniform, v ~ U(−2, 2), all visible k ≠ i, exact; 0 for
                      # noise-only. Var_ref is that of the SCM *after* the shift.
    delta: float      # ‖Θ_new − Θ_old‖_F over the visible block (0 for noise-only and for H)
    delta_sigma: float  # |σ_new[j] − σ_old[j]| (0 for "small")
```

`world.schedule` is a `list[ShiftRecord]` sorted by `t`, length 8 (0 when
`shifts_enabled=False`). `m` for the H shift is computed like any other (it changes visible
truth through `b_H`).

### 1.3 `WorldView` (non-oracle agents)

**Decision.** The protocol, not the agent, performs every world call; agents receive the
observational batch through `observe()` and every interventional row through
`receive_intervention()`. The SPEC's "`WorldView` exposing only `observe()` and
`intervene(i, v)`" is therefore realised as those two hooks, and the `WorldView` object handed
to agents carries **only the public dimensions**. It must not hold a reference to the `World`.

```python
@dataclass(frozen=True)
class WorldView:
    d: int
    B: int
    T: int
    n_obs: int
    n_queries: int
```

SPEC test 3 asserts that `vars(view)` / `dir(view)` contain none of
`W, b, sigma, permutation, schedule, world, truth`.

### 1.4 `OracleAccess` (oracle agents only)

Handed to `oracle`, `oracle-structure`, `mech-oracle-detect`, `mech-full-diag`. Each may use
**only** the method named for it in §3; the others exist so that one class serves all four.

```python
class OracleAccess:
    view: WorldView                      # same public dimensions
    d: int; B: int; T: int; n_obs: int; n_queries: int
    def true_adjacency(self) -> np.ndarray          # (d, d) bool         — oracle-structure
    def truth(self, i: int, v: float) -> np.ndarray # (d,) current SCM    — oracle
    def shifted_mechanism(self, t: int) -> int      # j of the shift at episode t, -1 if none,
                                                    # d if the H shift  — mech-oracle-detect
    def hidden_pair(self) -> "tuple[int, int] | None"  # (A, B)          — mech-full-diag
    def true_W(self) / true_b(self) / true_sigma(self)  # current SCM     — oracle only
```

`OracleAccess` holds the `World` privately (`self._world`); the agent code that receives it is
by definition oracle code and is listed as such in the registry.

---

## 2. `agents/base.py`

```python
VALUE_CYCLE = (2.0, -2.0, 1.0, -1.0)
ALLOWED_VALUES = frozenset(VALUE_CYCLE)
QUANTILE_LEVELS = (0.005, 0.05, 0.25, 0.75, 0.95, 0.995)
WARMUP_EPISODES = 3
```

### 2.1 `ValueCycle`

One per agent run; used by the protocol for the round-robin floor and by agents for their
free-budget values, so every sample on variable `i` — floor or free — takes the next value in
`i`'s cycle and the position persists across episodes (SPEC step 6).

```python
class ValueCycle:
    def __init__(self, d): ...
    def next_value(self, i: int) -> float   # VALUE_CYCLE[pos[i] % 4], then pos[i] += 1
    def round_robin(self) -> tuple[int, float]
        # i = next_var; next_var = (i + 1) % d; return (i, next_value(i)).
        # d*F consecutive calls give exactly F samples per variable from any starting point.
```

### 2.2 `Agent`

```python
class Agent:
    # class-level flags; the registry (§3) overrides them per variant via kwargs
    is_oracle: bool = False          # receives OracleAccess instead of WorldView
    intervenes: bool = True          # False: protocol skips step 6 entirely (marginal-mean,
                                     # obs-window, obs-cumulative, oracle)
    uses_floor: bool = True          # False only for mech-full-nofloor
    learns_structure: bool = False   # SHD is computed for these
    has_detector: bool = False       # detector_stats() is meaningful
    never_abstains: bool = False     # tau forced to +inf (marginal-mean, oracle)

    def __init__(self, name: str, agent_id: int, d: int, seed: int,
                 constants: dict | None, access: "WorldView | OracleAccess"):
        self.name, self.agent_id, self.d, self.seed = ...
        self.access = access
        self.constants = dict(constants or {})
        self.rng = np.random.default_rng([seed, agent_id, 5])   # the ONLY agent RNG
        self.cycle = ValueCycle(d)
        self.tau = +inf if never_abstains else float(constants.get("tau", inf))
        self.lam = float(constants.get("lambda", inf))            # detector threshold
        self.window = constants.get("W")                          # None → agent default
        self.gamma = constants.get("gamma")                       # None → agent default
```

Hooks, in the order the protocol calls them within one episode (§4 gives the exact loop).
Every hook is called **exactly once per episode** except the two per-sample hooks.

| Hook | Called at | Returns | Default in base |
|---|---|---|---|
| `observe(X_obs, t)` | step 3 | `None` | `NotImplementedError` |
| `detect_and_reset()` | step 4 | `list[int]` mechanisms whose buffer was truncated this episode by the **reset rule** (detector-induced, oracle-induced, or all `d` for `mech-reset-all`). `[]` if none. | `[]` |
| `detector_stats()` | step 4, right after `detect_and_reset()` | `None`, or `{"g": (d,) float, "glr": (d,) float}`: the Page–Hinkley statistic `g_t` **before** the reset rule zeroed it and this episode's `GLR_t`; `NaN` where the `N_min` guard skipped the mechanism. Needed by `calibrate.py` to choose `λ` with `λ = inf`. | `None` |
| `refit()` | step 5, after the batch was appended | `None` | `NotImplementedError` |
| `choose_free_interventions(n_free, t)` | step 6 round 2, only when `n_free > 0` and `t > WARMUP_EPISODES` | `list[(i, v)]` of length exactly `n_free`, `0 <= i < d`, `v ∈ ALLOWED_VALUES`; `v` must come from `self.cycle.next_value(i)` | `NotImplementedError` if `intervenes` |
| `score_before_reveal(i, v, row)` | step 6, per sample, **before** `receive_intervention` | `float` self log-score `Σ_{j≠i} log t(x_j; …)` under the posterior at that moment, or `None` (baselines) | `None` |
| `receive_intervention(i, v, row)` | step 6, per sample | `None` | `NotImplementedError` if `intervenes` |
| `update_structure(t)` | step 7 | `list[int]` mechanisms whose parent set changed (structure-induced refits). `[]` for non-structure agents. | `[]` |
| `answer(query_i, query_v)` | step 8 | `dict` with `point (Q, d) float`, `quantiles (Q, d, 6) float` (levels in `QUANTILE_LEVELS` order, non-decreasing along the last axis). Entries at `j == i` are ignored and may be anything (NaN ok). An optional `abstain` key is **ignored**: the evaluator computes abstention (§4.3). | `NotImplementedError` |
| `learned_adjacency()` | step 8 | `(d, d) bool` children in rows (the structure used for this episode's answers), or `None` | `None` |
| `internal_objectives()` | step 8 | `None`, or `{"self_score": float, "var_obj_before": float, "var_obj_after": float}` (any entry may be NaN) | `None` |

Semantics that agents must respect:

- **Posteriors are a deterministic function of buffers** (SPEC step 5). `refit()` is called
  once per episode (step 5). After every `receive_intervention` the agent must leave its
  posteriors consistent with its buffers before the next hook call, because
  `score_before_reveal` for sample `k + 1` and `choose_free_interventions` are evaluated "under
  the posterior at that moment" (SPEC step 6). For NIG mechanisms this is an incremental
  sufficient-statistics update plus a small solve per non-target mechanism.
- `score_before_reveal(i, v, row)` receives the full realised row but must use it only to
  evaluate the density; the sample enters buffers only in `receive_intervention`.
- Buffers: an interventional row with target `i` goes to every mechanism `j ≠ i`. The
  observational batch goes to every mechanism.
- `var_obj_before` **Decision:** `Σ_i score(i)` under the posterior at the end of step 5
  (after this episode's observational batch and detection, before any interventional row);
  `var_obj_after`: the same at the end of step 7. `self_score`: mean of the `B`
  `score_before_reveal` values of the episode. The protocol also stores the per-sample scores
  it collected, so `self_score` is cross-checkable.
- `detect_and_reset()` for `mech-oracle-detect`: `j = access.shifted_mechanism(t)` for the
  current `t` (remember `t` from `observe`); truncate `j` iff `0 <= j < d` (the H shift cannot
  be attributed to a visible mechanism; no reset). Return `[j]` or `[]`.
- Agents never see `truth`, `Var_ref`, the schedule or the permutation. The only quantities
  from `constants` they read are `tau` (protocol-side only, see §4.3), `lambda`, `W`, `gamma`.
- `mech-full-diag`: identical to `mech-full` except mechanism `B`'s buffer accepts only rows
  whose intervention target was `A` (`(A, B) = access.hidden_pair()`); observational rows and
  other interventional rows are not added to `B`'s buffer. Detection for `B` therefore stays
  under the `N_min` guard for most of the run; that is acceptable for a diagnostic.

---

## 3. `agents/__init__.py` — registry

```python
AGENT_IDS = {"marginal-mean": 0, "obs-window": 1, "obs-cumulative": 2, "int-pairwise": 3,
             "mech-full": 4, "mech-random": 5, "mech-full-nofloor": 6, "mech-reset-all": 7,
             "mech-no-detect": 8, "mech-oracle-detect": 9, "mech-overconfident": 10,
             "oracle-structure": 11, "oracle": 12, "mech-full-diag": 13}
```

`agent_id` enters the agent RNG key `[seed, agent_id, 5]`; it is fixed forever by this table.

`REGISTRY[name] -> AgentSpec(module, cls, kwargs, is_oracle, hidden_only)`; the module is
imported lazily, so `import agents` works before the implementer files exist.

| Name | Module.class | Constructor kwargs (after the six positional args) |
|---|---|---|
| `marginal-mean` | `baselines.MarginalMeanAgent` | — |
| `obs-window` | `baselines.ObsPairwiseAgent` | `windowed=True` (window `W` from constants, default 3) |
| `obs-cumulative` | `baselines.ObsPairwiseAgent` | `windowed=False` |
| `int-pairwise` | `baselines.IntPairwiseAgent` | — (window `W` from constants, default 5) |
| `mech-full` | `mech.MechAgent` | `structure="learned", reset_rule="per-mechanism", selection="active", floor=True, prior="default"` |
| `mech-random` | `mech.MechAgent` | `..., selection="random"` |
| `mech-full-nofloor` | `mech.MechAgent` | `..., floor=False` |
| `mech-reset-all` | `mech.MechAgent` | `..., reset_rule="all"` |
| `mech-no-detect` | `mech.MechAgent` | `..., reset_rule="none"` (forgetting `γ` from constants, default 0.7) |
| `mech-oracle-detect` | `mech.MechAgent` | `..., reset_rule="oracle"` (is_oracle) |
| `mech-overconfident` | `mech.MechAgent` | `..., prior="overconfident"` |
| `oracle-structure` | `mech.MechAgent` | `structure="true", reset_rule="per-mechanism", selection="active", floor=True, prior="default"` (is_oracle) |
| `oracle` | `oracle.OracleAgent` | — (is_oracle; `intervenes=False`, `never_abstains=True`) |
| `mech-full-diag` | `mech.MechAgent` | `..., diag_pair=True` (is_oracle, hidden_only) |

"`...`" means the `mech-full` kwargs with the listed change. `MechAgent`'s signature is

```python
MechAgent(name, agent_id, d, seed, constants, access, *,
          structure="learned",          # "learned" | "true"
          reset_rule="per-mechanism",   # "per-mechanism" | "all" | "oracle" | "none"
          selection="active",           # "active" | "random"
          floor=True,
          prior="default",              # "default" (V0=10I, a0=2, b0=1) | "overconfident" (0.1I, 50, 50)
          diag_pair=False)
```

and it must set `uses_floor = floor`, `learns_structure = (structure == "learned")`,
`has_detector = reset_rule in {"per-mechanism", "all"}`. `is_oracle` is set on the instance by
`build_agent` from the registry, so classes need not set it; an implementer's own tests that
bypass `build_agent` should pass `OracleAccess` to exactly the four oracle names.

Helpers exported by `agents/__init__.py`:

```python
def build_agent(name, world, seed, constants=None) -> Agent
    # access = world.oracle_access() if REGISTRY[name].is_oracle else world.view()
    # d = world.d; passes REGISTRY[name].kwargs
def agent_constants(all_constants: dict | None, name: str) -> dict
    # flattens constants.json (§5) to the per-agent dict handed to the constructor
MAIN_AGENTS, SWEEP_AGENTS, SWEEP_NOFLOOR_BUDGETS, HIDDEN_AGENTS, KNOB_AGENTS   # tuples from SPEC §Compute budget
```

---

## 4. `protocol.py`

```python
def floor_size(B, d) -> (F, n_floor, n_free)      # F = ceil(B / (2d)); n_floor = d·F; B >= d
def run_agent(world, agent) -> dict[str, np.ndarray]   # the whole T-episode loop; returns raw
def save_raw(path, raw) -> None                   # np.savez_compressed
def raw_filename(agent_name, seed) -> str         # "<agent>__seed<s>.npz"
def select_off_target(arr, query_i)               # (Q, d[, ...]) -> (Q, d-1[, ...]), §0 layout
def evaluate_answer(ans, query_i, truth, sd_ref, tau) -> dict   # §4.3, (Q, d-1) arrays
SHIFT_TYPE_CODES = {"large": 0, "small": 1, "noise-only": 2}
```

`select_off_target` is the reference implementation of the column layout; `metrics.py` may
import it (protocol.py imports only `agents.base`).

`run_agent` requires `agent.d == world.d`, `world.B >= world.d`, and runs episodes
`t = 1 … T`:

```
1  rec = world.apply_shift(t)                       # ShiftRecord | None → shift_* fields
   record true_W, true_b, true_sigma, sd_ref = sqrt(var_ref), true_adj, F_obs, confounding
2  X_obs = world.observational_batch(t); qi, qv = world.query_set(t)
3  agent.observe(X_obs, t)
4  resets = agent.detect_and_reset()  → reset_events rows (t, j)
   stats  = agent.detector_stats()   → g_stat[t-1], glr_stat[t-1]
5  agent.refit()
6  if agent.intervenes:
       F, n_floor, _ = floor_size(B, d)
       n_rr = B                      if t <= WARMUP_EPISODES          # warm-up: all round-robin
            = n_floor                elif agent.uses_floor
            = 0                      otherwise (mech-full-nofloor)
       for _ in range(n_rr):  (i, v) = agent.cycle.round_robin();  SAMPLE(i, v)
       n_free = B − n_rr
       if n_free > 0:
           plan = agent.choose_free_interventions(n_free, t)   # validated: length, range, values
           for (i, v) in plan: SAMPLE(i, v)
   where SAMPLE(i, v):  row = world.intervene(t, k, i, v)   # k = 0-based sample counter
                        s = agent.score_before_reveal(i, v, row)   → sample_score[t-1, k]
                        agent.receive_intervention(i, v, row);  k += 1
   (agents with intervenes == False: intervention_i[t-1] = -1, sample_score NaN)
7  refits = agent.update_structure(t)  → struct_refit_events rows (t, j)
8  ans = agent.answer(qi, qv); evaluate (§4.3); adj = agent.learned_adjacency();
   obj = agent.internal_objectives()
```

The round-robin floor is thus identical for every intervening agent (same `ValueCycle`
sequence from the same start) — the paired design of SPEC step 6. Whether warm-up applies is
decided by `t`, not by the agent.

### 4.1 Budget arithmetic (SPEC §Episode protocol) — asserted by `tests/test_protocol.py`

| `B` | `F` | floor `dF` | free |
|---|---|---|---|
| 10 | 1 | 6 | 4 |
| 25 | 3 | 18 | 7 |
| 50 | 5 | 30 | 20 |
| 100 | 9 | 54 | 46 |

### 4.2 Validation performed by the protocol (raises `ValueError`)

- `choose_free_interventions` returns exactly `n_free` pairs, `0 <= i < d` integers,
  `v ∈ {−2, −1, 1, 2}`.
- `answer` returns `point` of shape `(Q, d)` and `quantiles` of shape `(Q, d, 6)`, finite at
  every `j ≠ i`, quantiles non-decreasing along the last axis (tolerance `1e-9`).
- `detect_and_reset` / `update_structure` return iterables of ints in `[0, d)`.

### 4.3 Evaluator (step 8) — computed in `protocol.py`, never by agents

With `truth[q] = world.truth(qi[q], qv[q])` (shape `(Q, d)`), `sd = world.sigma_ref()`,
and `sel(·)` the `j ≠ i` selection of §0 (→ `(Q, d − 1)`):

```
err         = sel((point − truth) / sd)                         # signed, normalized
width50/90/99 = sel((q[3]−q[2]) / sd), sel((q[4]−q[1]) / sd), sel((q[5]−q[0]) / sd)
width90_raw = sel(q[4] − q[1])                                  # agent's raw uncertainty score
cov50 = sel(q[2] <= truth <= q[3]);  cov90 = sel(q[1] <= truth <= q[4]);  cov99 = sel(q[0] <= truth <= q[5])
abstain     = width90_raw > agent.tau                           # SPEC: (q0.95 − q0.05 > τ_agent)
```

`nMSE` for one prediction is `err²` (SPEC Metric 1). The unnormalized signed error needed by
P6(ii) is `err · sd_ref[t, j]`. Coverage is inclusive at the end points so the `oracle`
(width 0, quantiles equal to the truth) covers with probability 1.

---

## 5. `constants.json`

Written by `calibrate.py` from validation seeds 1000–1009 (SPEC §Tuning and calibration).

```json
{
  "spec_version": "v2",
  "validation_seeds": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
  "created_utc": "2026-09-20T12:00:00Z",
  "lambda": 7.31,
  "lambda_details": {"false_reset_rate_at_lambda": 0.009, "grid_multipliers": [0.5, 1, 2, 4, 8]},
  "agents": {
    "marginal-mean":      {"tau": null},
    "obs-window":         {"W": 3,   "tau": 0.912},
    "obs-cumulative":     {"tau": 0.377},
    "int-pairwise":       {"W": 5,   "tau": 1.204},
    "mech-full":          {"tau": 0.655},
    "mech-random":        {"tau": 0.701},
    "mech-full-nofloor":  {"tau": 0.733},
    "mech-reset-all":     {"tau": 0.690},
    "mech-no-detect":     {"gamma": 0.7, "tau": 0.802},
    "mech-oracle-detect": {"tau": 0.648},
    "mech-overconfident": {"tau": 0.112},
    "oracle-structure":   {"tau": 0.601},
    "oracle":             {"tau": null},
    "mech-full-diag":     {"tau": 0.655}
  }
}
```

- `lambda` is one number for every detector agent (top level). `W`, `gamma`, `tau` are per
  agent; `tau: null` means never abstain (`+inf`). Numbers above are placeholders.
- `mech-full-diag` is not run on validation (hidden only); `calibrate.py` copies `mech-full`'s
  `tau` to it and records that in `lambda_details`/a `notes` key.
- `agent_constants(all_constants, name)` (in `agents/__init__.py`) returns
  `{"lambda": λ, **all_constants["agents"].get(name, {})}` with `null → inf` for `tau`; with
  `all_constants=None` it returns `{}` so every agent falls back to its documented defaults
  (`tau = inf`, `lambda = inf` → no resets, `W`/`gamma` defaults). Those defaults are only
  ever exercised on validation seeds.
- **Refusal rule.** `run.py` must exit with an error (before constructing any world) when
  any requested seed is `< 1000` and `--constants` was not given or the file does not parse
  into the schema above (top-level `lambda` number and `agents` object with a `tau` for every
  requested agent). Validation seeds (`>= 1000`) may run without it (that is how
  `calibrate.py` produces it). `run.py` writes the file's SHA-256 (`hashlib.sha256(bytes)`)
  into `results/<run>/config.json` as `constants_sha256` (`null` when absent) — SPEC
  §Tuning. `calibrate.py` passes grid values by building the per-agent dict itself
  (e.g. `{"W": 8}` or `{"lambda": 2 * lam_star}`), never by editing `constants.json`.

---

## 6. Raw `.npz` schema — `results/<run>/raw/<agent>__seed<s>.npz`

Written by `protocol.save_raw` with `np.savez_compressed`. Load with
`np.load(path, allow_pickle=False)`. Shapes use `T, Q, d, B` of the run; `R`, `R2` are event
counts (may be 0 — arrays then have shape `(0, 2)`). Scalars are 0-d arrays; `str` scalars
are `numpy.str_` 0-d unicode arrays (`str(arr[()])`).

**Run identity (scalars)**

| Key | dtype | Meaning |
|---|---|---|
| `agent_name` | str | registry name |
| `agent_id` | int32 | from `AGENT_IDS` |
| `seed` | int32 | world seed |
| `d, T, B, n_obs, Q` | int32 | dimensions |
| `shift_type` | str | run knob |
| `hetero` | int32 | `γ_h` |
| `hidden` | bool | variant flag |
| `shifts_enabled` | bool | False only in λ-calibration runs |
| `learns_structure`, `has_detector`, `intervenes`, `uses_floor`, `is_oracle` | bool | agent flags |
| `tau` | float32 | the τ used for `abstain` (`inf` = never) |
| `wall_time_s` | float32 | wall time of `run_agent` |
| `hidden_A`, `hidden_B` | int32 | confounded visible pair, `-1` when not hidden |

**Per-episode world state** (row `t − 1`)

| Key | dtype | shape | Meaning |
|---|---|---|---|
| `episode_t` | int32 | `[T]` | `1 … T` |
| `shift_j` | int32 | `[T]` | shifted mechanism at the start of `t`; `-1` none; `d` = hidden `H` |
| `shift_type_code` | int8 | `[T]` | `-1` none, `0` large, `1` small, `2` noise-only |
| `shift_m`, `shift_delta`, `shift_delta_sigma` | float32 | `[T]` | `ShiftRecord` fields; NaN where no shift |
| `true_adj` | bool | `[T, d, d]` | visible DAG, children in rows (constant in `t`) |
| `true_W` | float32 | `[T, d, d]` | current visible weights |
| `true_b` | float32 | `[T, d]` | current intercepts |
| `true_sigma` | float32 | `[T, d]` | current noise scales |
| `sd_ref` | float32 | `[T, d]` | `sqrt(Var_ref)` of the current SCM (normalizer) |
| `F_obs` | float32 | `[T]` | P1 population floor of the current SCM |
| `conf_c`, `conf_F` | float32 | `[T]` | P6 `c`, `F_conf`; NaN when not hidden |

**Per-prediction** (`[T, Q, d−1]`, column layout of §0)

| Key | dtype | Meaning |
|---|---|---|
| `err` | float32 | signed normalized error `(point − truth)/sd_ref(X_j)`; nMSE = `err²` |
| `width50`, `width90`, `width99` | float32 | normalized interval widths |
| `width90_raw` | float32 | raw `q0.95 − q0.05` (what `τ` is compared to; `calibrate.py` uses it) |
| `cov50`, `cov90`, `cov99` | bool | truth inside the 50/90/99 % interval |
| `abstain` | bool | `width90_raw > tau` |

**Per-query** (`[T, Q]`)

| Key | dtype | Meaning |
|---|---|---|
| `query_i` | int32 | target `i` |
| `query_v` | float32 | value `v` |

**Per-sample interventions** (`[T, B]`; `-1`/NaN for agents with `intervenes == False`)

| Key | dtype | Meaning |
|---|---|---|
| `intervention_i` | int32 | chosen target of sample `k` (floor samples first) |
| `intervention_v` | float32 | chosen value |
| `sample_score` | float32 | self log-score recorded before reveal; NaN if the agent returned `None` |

**Events** (`[R, 2]` int32, columns `(t, j)`, `t` is the episode label)

| Key | Meaning |
|---|---|
| `reset_events` | resets by the reset rule (Metric 8 uses these) |
| `struct_refit_events` | structure-induced refits (logged separately, not counted in Metric 8) |

**Per-episode agent state**

| Key | dtype | shape | Meaning |
|---|---|---|---|
| `learned_adj` | bool | `[T, d, d]` | agent's adjacency after step 7; all False if the agent returned `None` (check `learns_structure`) |
| `has_learned_adj` | bool | `[T]` | True where `learned_adjacency()` returned an array |
| `g_stat`, `glr_stat` | float32 | `[T, d]` | detector statistics (NaN when absent/guarded) |
| `self_score`, `var_obj_before`, `var_obj_after` | float32 | `[T]` | Metric 9; NaN when the agent returned `None` |

Reconstructing `j` for column `c` of a per-prediction array (for `metrics.py`):

```python
def column_to_j(query_i, d):        # query_i: [T, Q] int → [T, Q, d-1] int
    c = np.arange(d - 1)
    return c[None, None, :] + (c[None, None, :] >= query_i[..., None])
```

Metric conventions derived from the raw file: **shift episode** = `shift_j[t−1] >= 0`;
**post-shift** = the episode after; **stable** = all others (SPEC §Metrics). Shift-conditional
metrics use `shift_m >= 0.05` (or `shift_delta_sigma` for `noise-only`). The number of shifts
excluded by the `m` filter is reported. Descendant/ancestor relations come from the
transitive closure of `true_adj`.

---

## 7. Other modules — contracts

### `run.py` (CLI, SPEC §Deliverables)

`--agents all|a,b,c --seeds 0-19 --episodes 60 --budget 50 --n-obs 200 --shift-type
large|small|noise-only --hetero 0|1 --hidden --workers 4 --constants constants.json --out
results/<name>`. Per `(agent, seed)` job: `world = World(seed, ...)`,
`agent = build_agent(name, world, seed, agent_constants(consts, name))`,
`raw = run_agent(world, agent)`, `save_raw(out/raw/raw_filename(name, seed), raw)`. Jobs run in
`multiprocessing.Pool(workers)`; **a fresh `World` per job** so nothing is shared. `all` means
`MAIN_AGENTS`, or `HIDDEN_AGENTS` with `--hidden`. Refusal rule in §5. `config.json`: every CLI
parameter, `git rev-parse HEAD`, `constants_sha256`, per-`(agent, seed)` wall time (from
`raw["wall_time_s"]`), start/end timestamps.

### `calibrate.py`

Runs validation seeds only (`>= 1000`), builds per-agent constants dicts directly, and writes
`constants.json` in the §5 schema. λ: run `mech-full` with `World(..., shifts_enabled=False)`
and `{"lambda": inf}`, pool `g_stat` over all mechanism-episodes where it is finite, choose the
smallest λ with `mean(g > λ) <= 0.01`. τ per agent: 90th percentile of `width90_raw` over
stable episodes of its validation runs. `W`, `γ` grids as in SPEC.

### `metrics.py`

Consumes only the raw files (§6) plus `config.json`; never imports `world` or `agents`. All
nine metrics, the `j`-dependence mask, Kaplan–Meier recovery, AURC, bootstrap CIs.

### `report.py`

Reads `metrics.json`, writes `figures/` and `REPORT.md`.

### Ownership

| File | Owner |
|---|---|
| `SPEC.md` | frozen |
| `INTERFACES.md`, `__init__.py`, `agents/__init__.py`, `agents/base.py`, `protocol.py`, `tests/conftest.py`, `tests/test_protocol.py` | architect |
| `world.py`, `tests/test_world.py` | world implementer |
| `agents/baselines.py`, `agents/mech.py`, `agents/oracle.py`, `tests/test_agents*.py` | agent implementer(s) |
| `metrics.py`, `tests/test_metrics.py` | metrics implementer |
| `run.py`, `calibrate.py`, `report.py`, `tests/test_smoke.py` | runner implementer |

---

## 8. Fake-world contract (for agent and protocol tests)

`tests/test_protocol.py` contains `FakeWorld`, a fixed 6-node chain SCM (unit weights,
intercepts 0.5, unit noise, one scheduled intercept shift of mechanism 2 at episode 5)
implementing every method of §1.1 that `protocol.run_agent` touches, with the same keyed RNG
streams, plus a minimal `WorldView` / `FakeOracleAccess`. Agent implementers may import it
(`from test_protocol import FakeWorld` inside `tests/`, where pytest puts the directory on
`sys.path`) to unit-test an agent through the real protocol without `world.py`:

```python
world = FakeWorld(seed=0, T=6, B=10)
agent = MechAgent("mech-full", 4, world.d, 0, {}, world.view(), **REGISTRY["mech-full"].kwargs)
raw = protocol.run_agent(world, agent)
```
