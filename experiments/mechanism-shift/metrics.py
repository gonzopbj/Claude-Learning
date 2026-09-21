"""All nine metrics, the analysis helpers and the pre-registered per-seed scalars
(SPEC.md "Metrics", "Analysis plan", "Pre-registered predictions"; INTERFACES.md section 6
for the raw file this module reads).

Everything here is a pure function of *loaded raw dictionaries* - the `.npz` files written by
`protocol.save_raw` - and nothing else.  This module never imports `world` or `agents`, so it
can only see what the evaluator wrote down: signed normalized errors `err` (nMSE = err**2),
normalized widths, coverage booleans, abstention, the query targets, the world's true DAG and
shift log, the agent's learned DAG, reset events and internal objectives.

Layout of the module
    1. loading and small helpers        load_raw, scalar, column_to_j, transitive_closure
    2. episode kinds and the shift log  episode_kinds, shift_log, qualifying_shifts
    3. prediction masks                 descendant_mask, j_dependent_mask, path_equal_mask, ...
    4. metric 1  nMSE                   nmse_series, nmse_summary
    5. metric 2  regret and recovery    post_shift_regret, recovery_time, kaplan_meier
    6. metric 3  forgetting             forgetting_bump
    7. metric 4  calibration            coverage, calibration_table
    8. metric 5  selective risk         risk_coverage_curve, aurc, selective_risk
    9. metric 6  SHD                    shd, shd_series
   10. metric 8  credit assignment      credit_assignment
   11. metric 9  internal objectives    internal_objectives
   12. analysis  bootstrap, Holm        bootstrap_ci, bootstrap_pvalue, holm, cluster_bootstrap
   13. primaries P1-P9                  primary_P1 ... primary_P9, PRIMARIES, evaluate_primary

Conventions (INTERFACES.md section 0): episodes are labelled t = 1..T and stored at row t-1;
per-prediction arrays are [T, Q, d-1] with column c standing for variable j = c if c < i else
c + 1; adjacency has children in rows (adj[j, p] is the edge p -> j).

Where the SPEC is ambiguous the reading chosen here is marked "Decision:" in a comment next
to the code that implements it.  None of these decisions look at anything an agent could not
see, and all of them treat every agent identically, so the paired design is preserved.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# Shift-conditional metrics only count shifts whose magnitude is at least this
# (SPEC "Shift process": "reported for shifts with m >= 0.05").
MAGNITUDE_MIN = 0.05
# A mistake is absolute: normalized squared error above this (SPEC Metric 5).
MISTAKE_THRESHOLD = 0.25
# Regret window: the shift episode and the three following (SPEC Metric 2).
REGRET_WINDOW = 4
# shift_type_code values written by protocol.py.
CODE_LARGE, CODE_SMALL, CODE_NOISE_ONLY = 0, 1, 2
# Coverage grid for the risk-coverage curve: 100 points in (0, 1] (SPEC Metric 5).
COVERAGE_GRID = np.linspace(0.01, 1.0, 100)
FIXED_COVERAGES = (0.5, 0.8, 0.9, 1.0)
# Bootstrap resamples (SPEC "Analysis plan").
N_BOOT = 2000


# =========================================================================== 1. loading

def load_raw(path) -> dict:
    """Load one raw `.npz` into a plain dict of numpy arrays (scalars stay 0-d arrays)."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_results(raw_dir) -> dict:
    """Load every `<agent>__seed<s>.npz` under `raw_dir` as {agent_name: {seed: raw}}."""
    out: dict = {}
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".npz") or "__seed" not in name:
            continue
        raw = load_raw(os.path.join(raw_dir, name))
        out.setdefault(scalar(raw, "agent_name"), {})[scalar(raw, "seed")] = raw
    return out


def scalar(raw: dict, key: str):
    """Python value of a 0-d entry of the raw dict (str, int, float or bool)."""
    v = raw[key]
    if isinstance(v, np.ndarray):
        v = v[()]
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def dims(raw: dict) -> tuple[int, int, int]:
    """(T, Q, d) read from the per-prediction array, not from the scalars, so a truncated or
    hand-built raw dict works too."""
    T, Q, dm1 = raw["err"].shape
    return int(T), int(Q), int(dm1) + 1


def column_to_j(query_i: np.ndarray, d: int) -> np.ndarray:
    """Variable index j for every column of a per-prediction array (INTERFACES.md section 6).

    query_i: [..., Q] int -> [..., Q, d-1] int; column c is j = c if c < i else c + 1.
    """
    qi = np.asarray(query_i)
    c = np.arange(d - 1)
    return c + (c >= qi[..., None])


def transitive_closure(adj: np.ndarray) -> np.ndarray:
    """reach[k, i] = True iff there is a directed path of length >= 1 from i to k.

    Warshall's algorithm on the (children in rows) adjacency; R[a, b] |= R[a, k] & R[k, b].
    """
    R = np.asarray(adj, dtype=bool).copy()
    d = R.shape[0]
    for k in range(d):
        R |= R[:, k][:, None] & R[k, :][None, :]
    return R


def _reach_reflexive(adj: np.ndarray) -> np.ndarray:
    """Transitive closure plus the identity (a node reaches itself)."""
    return transitive_closure(adj) | np.eye(adj.shape[0], dtype=bool)


# =========================================================================== 2. episodes and shifts

def episode_kinds(raw: dict) -> dict:
    """Boolean [T] arrays: shift (a mechanism was resampled at the start of t), post_shift
    (t + 1 of a shift) and stable (everything else), per SPEC "Metrics" definitions.

    Every logged shift counts here, whatever its magnitude and including the hidden H shift:
    the strata describe what happened to the world, not what a metric later filters on.
    """
    shift = np.asarray(raw["shift_j"]) >= 0
    post = np.zeros_like(shift)
    post[1:] = shift[:-1]
    post &= ~shift                       # cannot happen with min gap 5, kept for safety
    return {"shift": shift, "post_shift": post, "stable": ~shift & ~post}


@dataclass(frozen=True)
class Shift:
    """One logged shift (SPEC "Shift process": the world logs (seed, t, j, type, m, delta))."""
    t: int              # episode label at whose start the mechanism was resampled
    j: int              # mechanism; d means the hidden H (never a visible mechanism)
    type_code: int      # 0 large, 1 small, 2 noise-only
    m: float
    delta: float
    delta_sigma: float
    next_shift_t: int   # episode of the next logged shift, or T + 1 if none (censoring point)

    @property
    def magnitude(self) -> float:
        # Decision: the shift-conditional filter uses m, except for noise-only shifts where
        # m = 0 by construction and the SPEC says to use delta_sigma instead.
        return self.delta_sigma if self.type_code == CODE_NOISE_ONLY else self.m


def shift_log(raw: dict) -> list[Shift]:
    """All logged shifts of the run, in episode order, from the raw file's shift_* fields."""
    T = len(raw["shift_j"])
    ts = [int(t) for t in np.flatnonzero(np.asarray(raw["shift_j"]) >= 0) + 1]
    out = []
    for n, t in enumerate(ts):
        r = t - 1
        out.append(Shift(t=t, j=int(raw["shift_j"][r]), type_code=int(raw["shift_type_code"][r]),
                         m=float(raw["shift_m"][r]), delta=float(raw["shift_delta"][r]),
                         delta_sigma=float(raw["shift_delta_sigma"][r]),
                         next_shift_t=ts[n + 1] if n + 1 < len(ts) else T + 1))
    return out


def qualifying_shifts(raw: dict, magnitude_min: float = MAGNITUDE_MIN,
                      visible_only: bool = False) -> tuple[list[Shift], int]:
    """Shifts that shift-conditional metrics count, plus the number excluded by the filter.

    `visible_only` drops the hidden H shift (j == d), for metrics that need a visible
    mechanism (forgetting mask, credit assignment).
    """
    d = dims(raw)[2]
    kept, excluded = [], 0
    for s in shift_log(raw):
        if visible_only and s.j >= d:
            continue
        if s.magnitude >= magnitude_min:
            kept.append(s)
        else:
            excluded += 1
    return kept, excluded


# =========================================================================== 3. prediction masks

def episode_prediction_mask(raw: dict, episodes: np.ndarray) -> np.ndarray:
    """Broadcast a boolean [T] episode selector to the [T, Q, d-1] prediction shape."""
    T, Q, d = dims(raw)
    return np.broadcast_to(np.asarray(episodes, dtype=bool)[:, None, None], (T, Q, d - 1)).copy()


def descendant_mask(raw: dict) -> np.ndarray:
    """[T, Q, d-1] bool: the predicted variable j is a true descendant of the target i.

    Truth is E[X_j] on the complement (SPEC Metric 1: the non-descendant half is the
    false-effect rate).  Uses the transitive closure of each episode's true_adj.
    """
    T, Q, d = dims(raw)
    qi = np.asarray(raw["query_i"], dtype=np.int64)
    J = column_to_j(qi, d)                                       # [T, Q, d-1]
    out = np.zeros((T, Q, d - 1), dtype=bool)
    for r in range(T):
        reach = transitive_closure(raw["true_adj"][r])           # reach[k, i]: path i -> k
        out[r] = reach[J[r], qi[r][:, None]]
    return out


def j_dependent_mask(true_adj: np.ndarray, query_i: np.ndarray, j: int) -> np.ndarray:
    """SPEC Metric 3 rule.  Query (i, k) is j-dependent iff k == j or (j in desc(i) and
    j in anc(k)) in the true DAG; if j == i the query is independent (the intervention cuts
    j's mechanism).  query_i may be [Q] or [T, Q]; the result adds a trailing d-1 axis.

    Chain 0 -> 1 -> 2, j = 1: (0, 2) and (0, 1) dependent; (1, 2) and (2, 0) independent.
    """
    adj = np.asarray(true_adj, dtype=bool)
    d = adj.shape[0]
    qi = np.asarray(query_i, dtype=np.int64)
    K = column_to_j(qi, d)                                       # variable k of each column
    reach = transitive_closure(adj)                              # reach[k, i]: path i -> k
    i = qi[..., None]
    j_desc_of_i = reach[j, i]                                    # j in desc(i)
    j_anc_of_k = reach[K, j]                                     # j in anc(k)
    dep = (K == j) | (j_desc_of_i & j_anc_of_k)
    return dep & (i != j)


def j_dependent_mask_run(raw: dict, j: int) -> np.ndarray:
    """j_dependent_mask for every episode of a run: [T, Q, d-1]."""
    T = dims(raw)[0]
    return np.stack([j_dependent_mask(raw["true_adj"][r], raw["query_i"][r], j) for r in range(T)])


def pair_mask(raw: dict, i: int, j: int) -> np.ndarray:
    """[T, Q, d-1] bool: predictions of variable j under do(X_i) (used for P6's (A, B) pair)."""
    T, Q, d = dims(raw)
    qi = np.asarray(raw["query_i"], dtype=np.int64)
    J = column_to_j(qi, d)
    return (qi[..., None] == i) & (J == j)


def _path_edge_sets(adj: np.ndarray) -> np.ndarray:
    """E[i, k, b, a] = True iff edge a -> b lies on some directed path i -> k in `adj`.

    Two DAGs have the same set of directed paths i -> k iff they have the same set of edges on
    such paths (every path is made of such edges, and every such edge is on some path), so
    comparing these edge sets compares the path sets without enumerating paths.
    """
    adj = np.asarray(adj, dtype=bool)
    R = _reach_reflexive(adj)                                     # R[x, y]: y reaches x
    # a reachable from i: R[a, i]; k reachable from b: R[k, b]; edge a -> b: adj[b, a]
    return (R.T[:, None, None, :]           # [i, 1, 1, a]  = R[a, i]
            & R[None, :, :, None]           # [1, k, b, 1]  = R[k, b]
            & adj[None, None, :, :])        # [1, 1, b, a]  = adj[b, a]


def path_equal_matrix(true_adj: np.ndarray, learned_adj: np.ndarray) -> np.ndarray:
    """[d, d] bool, entry [i, k]: the set of directed paths i -> k is the same in both DAGs."""
    return np.all(_path_edge_sets(true_adj) == _path_edge_sets(learned_adj), axis=(2, 3))


def path_equal_mask(raw: dict) -> np.ndarray:
    """[T, Q, d-1] bool: learned path set i -> j equals the true one for that prediction
    (SPEC Metric 4 stratum (b)).  False everywhere in episodes without a learned adjacency.
    """
    T, Q, d = dims(raw)
    qi = np.asarray(raw["query_i"], dtype=np.int64)
    J = column_to_j(qi, d)
    out = np.zeros((T, Q, d - 1), dtype=bool)
    cache: dict = {}
    for r in range(T):
        if not bool(raw["has_learned_adj"][r]):
            continue
        key = (raw["true_adj"][r].tobytes(), raw["learned_adj"][r].tobytes())
        if key not in cache:
            cache[key] = path_equal_matrix(raw["true_adj"][r], raw["learned_adj"][r])
        out[r] = cache[key][qi[r][:, None], J[r]]
    return out


# =========================================================================== 4. metric 1: nMSE

def _masked_mean(values: np.ndarray, mask: np.ndarray | None) -> float:
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    return float(v.mean()) if v.size else math.nan


def nmse_series(raw: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-episode nMSE [T]: mean of err**2 over the predictions selected by `mask`
    ([T, Q, d-1] bool, default all); NaN in episodes where the mask is empty."""
    err2 = np.asarray(raw["err"], dtype=np.float64) ** 2
    if mask is None:
        return err2.mean(axis=(1, 2))
    m = np.asarray(mask, dtype=bool)
    n = m.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (err2 * m).sum(axis=(1, 2)) / np.maximum(n, 1), math.nan)


def nmse_summary(raw: dict, mask: np.ndarray | None = None) -> dict:
    """SPEC Metric 1 for one run restricted to `mask`: overall nMSE, the descendant /
    non-descendant split, and the abstention-aware variants (nMSE on the predictions the
    agent kept, nMSE on the ones it abstained on, realized abstention rate)."""
    T, Q, d = dims(raw)
    err2 = np.asarray(raw["err"], dtype=np.float64) ** 2
    m = np.ones_like(err2, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    desc = descendant_mask(raw)
    abst = np.asarray(raw["abstain"], dtype=bool)
    return {
        "nmse": _masked_mean(err2, m),
        "nmse_descendant": _masked_mean(err2, m & desc),
        "nmse_non_descendant": _masked_mean(err2, m & ~desc),
        "nmse_non_abstained": _masked_mean(err2, m & ~abst),
        "nmse_abstained": _masked_mean(err2, m & abst),
        "abstain_rate": _masked_mean(abst, m),
        "n_predictions": int(m.sum()),
    }


# =========================================================================== 5. metric 2: regret, recovery

def post_shift_regret(agent_series: np.ndarray, ref_series: np.ndarray, t: int,
                      window: int = REGRET_WINDOW) -> float:
    """SPEC Metric 2 regret: sum_{u = t}^{t + window - 1} (nMSE_agent(u) - nMSE_ref(u)),
    the window clipped at T.  Series are per-episode nMSE [T] stored at row u - 1."""
    T = len(agent_series)
    rows = slice(t - 1, min(t - 1 + window, T))
    return float(np.sum(np.asarray(agent_series[rows], dtype=np.float64)
                        - np.asarray(ref_series[rows], dtype=np.float64)))


def cumulative_nmse(series: np.ndarray, t: int, window: int = REGRET_WINDOW) -> float:
    """Exploratory companion: raw cumulative nMSE over the same window."""
    T = len(series)
    return float(np.sum(np.asarray(series[t - 1:min(t - 1 + window, T)], dtype=np.float64)))


def regret_per_shift(raw: dict, ref_raw: dict, shifts: list[Shift] | None = None) -> list[dict]:
    """Regret of `raw` against the reference agent `ref_raw` for every qualifying shift."""
    if shifts is None:
        shifts = qualifying_shifts(raw)[0]
    a, b = nmse_series(raw), nmse_series(ref_raw)
    return [{"t": s.t, "j": s.j, "magnitude": s.magnitude,
             "regret": post_shift_regret(a, b, s.t),
             "cumulative_nmse": cumulative_nmse(a, s.t)} for s in shifts]


def recovery_threshold(series: np.ndarray, t: int) -> float:
    """Pre-shift level L = mean nMSE over the 2 episodes before the shift; recovered when
    nMSE(u) <= max(1.5 L, L + 0.02) (SPEC Metric 2)."""
    pre = np.asarray(series[max(t - 3, 0):t - 1], dtype=np.float64)   # rows of t-2, t-1
    if pre.size == 0 or not np.all(np.isfinite(pre)):
        return math.nan
    L = float(pre.mean())
    return max(1.5 * L, L + 0.02)


def recovery_time(series: np.ndarray, t: int, next_shift_t: int | None = None) -> tuple[int, bool]:
    """(time, recovered) for the shift at episode t on a per-episode nMSE series [T].

    time = u - t for the first episode u >= t with nMSE(u) <= max(1.5 L, L + 0.02), so 0
    means recovered in the shift episode itself.  Right-censored at the next shift or T:
    if no such u exists before episode c_end = min(next_shift_t - 1, T), the observation is
    (c_end - t, False), i.e. "not recovered during the c_end - t + 1 observed episodes".
    """
    T = len(series)
    c_end = T if next_shift_t is None else min(int(next_shift_t) - 1, T)
    thr = recovery_threshold(series, t)
    if not math.isfinite(thr):
        return c_end - t, False
    for u in range(t, c_end + 1):
        if series[u - 1] <= thr:
            return u - t, True
    return c_end - t, False


def kaplan_meier(times, events) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival estimate on right-censored recovery/detection times.

    Returns (distinct event times, S evaluated just after each).  Units observed censored at
    a time are at risk for events at that same time (the standard convention).
    """
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    grid = np.unique(times[events]) if events.any() else np.array([], dtype=np.float64)
    S, surv = 1.0, []
    for tt in grid:
        at_risk = np.sum(times >= tt)
        n_events = np.sum((times == tt) & events)
        S *= 1.0 - n_events / at_risk
        surv.append(S)
    return grid, np.asarray(surv, dtype=np.float64)


def km_survival_at(grid: np.ndarray, surv: np.ndarray, time: float) -> float:
    """S(time) from a Kaplan-Meier curve (1 before the first event)."""
    idx = np.searchsorted(grid, time, side="right") - 1
    return 1.0 if idx < 0 else float(surv[idx])


def km_median(grid: np.ndarray, surv: np.ndarray) -> float:
    """Smallest time with S(time) <= 0.5; NaN if the curve never gets there."""
    below = np.flatnonzero(surv <= 0.5)
    return float(grid[below[0]]) if below.size else math.nan


def recovery_summary(times, events) -> dict:
    """KM median and the fraction recovered within 1, 2, 3 episodes.  "Within k episodes"
    counts the shift episode as the first, i.e. P(time <= k - 1) = 1 - S(k - 1)."""
    grid, surv = kaplan_meier(times, events)
    out = {"median": km_median(grid, surv), "n": int(len(times)),
           "n_censored": int(len(times) - int(np.sum(np.asarray(events, dtype=bool))))}
    for k in (1, 2, 3):
        out[f"within_{k}"] = 1.0 - km_survival_at(grid, surv, k - 1) if len(times) else math.nan
    return out


def recovery_per_shift(raw: dict, shifts: list[Shift] | None = None) -> list[dict]:
    """(time, recovered) for every qualifying shift of a run, on its per-episode nMSE."""
    if shifts is None:
        shifts = qualifying_shifts(raw)[0]
    series = nmse_series(raw)
    out = []
    for s in shifts:
        time, rec = recovery_time(series, s.t, s.next_shift_t)
        out.append({"t": s.t, "j": s.j, "magnitude": s.magnitude, "time": time, "recovered": rec})
    return out


# =========================================================================== 6. metric 3: forgetting

def forgetting_bump(raw: dict, t: int, j: int) -> float:
    """SPEC Metric 3: nMSE_indep(t) - mean(nMSE_indep(t - 2), nMSE_indep(t - 1)), where
    nMSE_indep(u) is the nMSE over the j-independent queries of episode u.  NaN if t < 3."""
    if t < 3:
        return math.nan
    indep = ~j_dependent_mask_run(raw, j)
    series = nmse_series(raw, indep)
    return float(series[t - 1] - 0.5 * (series[t - 3] + series[t - 2]))


def forgetting_per_shift(raw: dict, shifts: list[Shift] | None = None) -> list[dict]:
    """Forgetting bump for every qualifying shift of a visible mechanism."""
    if shifts is None:
        shifts = qualifying_shifts(raw, visible_only=True)[0]
    return [{"t": s.t, "j": s.j, "magnitude": s.magnitude, "bump": forgetting_bump(raw, s.t, s.j)}
            for s in shifts]


# =========================================================================== 7. metric 4: calibration

def coverage(raw: dict, mask: np.ndarray | None = None) -> dict:
    """Empirical coverage at 50/90/99 % with the mean normalized width at each level, on ALL
    predictions in `mask` (never the non-abstained subset; SPEC Metric 4)."""
    out = {}
    for lvl in ("50", "90", "99"):
        out[f"cov{lvl}"] = _masked_mean(raw[f"cov{lvl}"], mask)
        out[f"width{lvl}"] = _masked_mean(raw[f"width{lvl}"], mask)
    out["abstain_rate"] = _masked_mean(raw["abstain"], mask)
    out["n"] = int(np.asarray(mask, dtype=bool).sum()) if mask is not None else int(raw["err"].size)
    return out


def calibration_table(raw: dict) -> dict:
    """Metric 4 strata: by episode kind, and for learned-structure agents by SHD = 0 vs > 0
    at that episode and by learned-vs-true path-set equality for the query."""
    kinds = episode_kinds(raw)
    table = {k: coverage(raw, episode_prediction_mask(raw, v)) for k, v in kinds.items()}
    if bool(np.any(raw["has_learned_adj"])):          # learned-structure agents only
        s = shd_series(raw)
        table["shd_zero"] = coverage(raw, episode_prediction_mask(raw, s == 0))
        table["shd_positive"] = coverage(raw, episode_prediction_mask(raw, s > 0))
        pe = path_equal_mask(raw)
        known = episode_prediction_mask(raw, raw["has_learned_adj"])
        table["path_equal"] = coverage(raw, pe & known)
        table["path_unequal"] = coverage(raw, ~pe & known)
    return table


# =========================================================================== 8. metric 5: selective risk

def risk_coverage_curve(err2: np.ndarray, score: np.ndarray,
                        grid: np.ndarray = COVERAGE_GRID) -> np.ndarray:
    """risk(c) for each coverage c in `grid`: mean err2 over the ceil(c N) predictions with the
    lowest uncertainty `score` (ties broken by position, stable sort)."""
    err2 = np.asarray(err2, dtype=np.float64).ravel()
    order = np.argsort(np.asarray(score, dtype=np.float64).ravel(), kind="stable")
    cum = np.cumsum(err2[order])
    n = len(err2)
    k = np.clip(np.ceil(np.asarray(grid) * n).astype(int), 1, n)
    return cum[k - 1] / k


def aurc(err2: np.ndarray, score: np.ndarray, grid: np.ndarray = COVERAGE_GRID) -> float:
    """Area under the risk-coverage curve: trapezoid integral of risk(c) over the grid.

    Decision: divided by the grid span (0.99 for 100 points in (0, 1]) so that a constant
    curve integrates to its value; this makes the random-ordering reference AURC exactly the
    overall nMSE, as SPEC Metric 5 states.
    """
    risk = risk_coverage_curve(err2, score, grid)
    return float(np.trapezoid(risk, grid) / (grid[-1] - grid[0]))


def selective_risk(raw: dict, mask: np.ndarray | None = None) -> dict:
    """SPEC Metric 5 for one run: risk-coverage curve, risk at fixed coverages, AURC with the
    oracle-ordering (sort by true error) and random-ordering (overall nMSE) references, and
    mistake recall / precision at the realized abstention rate."""
    err2 = np.asarray(raw["err"], dtype=np.float64) ** 2
    score = np.asarray(raw["width90"], dtype=np.float64)     # the agent's uncertainty score
    abst = np.asarray(raw["abstain"], dtype=bool)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        err2, score, abst = err2[m], score[m], abst[m]
    err2, score, abst = err2.ravel(), score.ravel(), abst.ravel()
    if err2.size == 0:
        return {"n": 0}
    curve = risk_coverage_curve(err2, score)
    mistake = err2 > MISTAKE_THRESHOLD
    return {
        "n": int(err2.size),
        "coverage_grid": COVERAGE_GRID.copy(),
        "risk_curve": curve,
        "risk_at": {c: float(risk_coverage_curve(err2, score, np.array([c]))[0]) for c in FIXED_COVERAGES},
        "aurc": aurc(err2, score),
        "aurc_oracle": aurc(err2, err2),                        # sort by the true error
        "aurc_random": float(err2.mean()),                      # constant curve
        "abstain_rate": float(abst.mean()),
        "mistake_rate": float(mistake.mean()),
        # P(abstain | mistake) and P(mistake | abstain) at the realized abstention rate
        "mistake_recall": float(abst[mistake].mean()) if mistake.any() else math.nan,
        "mistake_precision": float(mistake[abst].mean()) if abst.any() else math.nan,
    }


# =========================================================================== 9. metric 6: SHD

def shd(true_adj: np.ndarray, learned_adj: np.ndarray) -> int:
    """Structural Hamming distance: one per unordered pair whose skeleton differs, plus one per
    pair present in both but oriented differently."""
    A = np.asarray(true_adj, dtype=bool)
    L = np.asarray(learned_adj, dtype=bool)
    skel_A, skel_L = A | A.T, L | L.T
    upper = np.triu(np.ones_like(A), k=1)
    skeleton_diff = (skel_A != skel_L) & upper
    reversed_edge = skel_A & skel_L & (A != L) & upper
    return int(skeleton_diff.sum() + reversed_edge.sum())


def shd_series(raw: dict) -> np.ndarray:
    """SHD per episode [T] between learned_adj and true_adj (SPEC Metric 6); NaN where the
    agent returned no adjacency."""
    T = dims(raw)[0]
    out = np.full(T, math.nan)
    for r in range(T):
        if bool(raw["has_learned_adj"][r]):
            out[r] = shd(raw["true_adj"][r], raw["learned_adj"][r])
    return out


# =========================================================================== 10. metric 8: credit assignment

def credit_assignment(raw: dict, magnitude_min: float = MAGNITUDE_MIN) -> dict:
    """SPEC Metric 8 from `reset_events` (detector-induced; structure refits are NOT counted)
    and the shift log.

    Per qualifying visible shift (t, j): detection delay = episodes from t to the first reset
    of j, right-censored at the next shift; outcome over the window {t, t + 1}:
    'correct' if j was reset in the window, 'wrong-mechanism' if only other mechanisms were,
    'miss' if nothing was.  Per reset: it 'matches' a shift if that shift's mechanism was reset
    in the same or the previous episode (attribution precision = matched / all resets);
    a reset in the window of some logged shift but on a non-shifted mechanism is
    'misattributed'; a reset outside every window is a 'false alarm', and the false-reset rate
    is false alarms per mechanism-episode in stable periods.

    Decision: "stable periods" for the false-reset rate exclude the window {t, t + 1} of EVERY
    logged shift (any magnitude, any type, including the hidden H shift), because a reset
    there is not a false alarm even when the shift is too small to be counted as a hit.
    """
    T, Q, d = dims(raw)
    resets = [(int(t), int(j)) for t, j in np.asarray(raw["reset_events"]).reshape(-1, 2)]
    all_shifts = shift_log(raw)
    shifts, excluded = qualifying_shifts(raw, magnitude_min, visible_only=True)

    in_window = np.zeros(T + 2, dtype=bool)          # index = episode label
    for s in all_shifts:
        in_window[s.t] = True
        if s.t + 1 <= T:
            in_window[s.t + 1] = True
    n_stable_episodes = int(np.sum(~in_window[1:T + 1]))
    # Denominator for the false-reset rate: mechanism-episodes in which the detector was
    # actually tested (finite g statistic; the N_min guard stores NaN), so the evaluation
    # rate is on the same footing as calibrate.py's 1 % target (leakage audit finding).
    # Agents without detector statistics fall back to d x stable episodes.
    stable_mask = ~in_window[1:T + 1]
    g = np.asarray(raw["g_stat"], dtype=np.float64) if "g_stat" in raw else None
    if g is not None and g.shape[0] == T and np.isfinite(g).any():
        n_tested_stable = int(np.isfinite(g[stable_mask]).sum())
    else:
        n_tested_stable = d * n_stable_episodes

    per_shift = []
    for s in shifts:
        own = [t for (t, j) in resets if j == s.j and s.t <= t < s.next_shift_t]
        delay = min(own) - s.t if own else s.next_shift_t - 1 - s.t   # censored length if none
        window_resets = [(t, j) for (t, j) in resets if t in (s.t, s.t + 1)]
        own_in_window = any(j == s.j for _, j in window_resets)
        others_in_window = any(j != s.j for _, j in window_resets)
        outcome = ("correct" if own_in_window else
                   "wrong-mechanism" if others_in_window else "miss")
        per_shift.append({"t": s.t, "j": s.j, "magnitude": s.magnitude, "outcome": outcome,
                          "delay": int(delay), "detected": bool(own),
                          "correct_delay0": bool(own) and min(own) == s.t,
                          "others_reset_in_window": others_in_window})

    matched = [(t, j) for (t, j) in resets
               if any(s.j == j and t in (s.t, s.t + 1) for s in shifts)]
    misattributed = [(t, j) for (t, j) in resets
                     if in_window[t] and not any(s.j == j and t in (s.t, s.t + 1) for s in all_shifts)]
    false_alarms = [(t, j) for (t, j) in resets if not in_window[t]]

    n_s = len(per_shift)
    frac = lambda key: (sum(1 for p in per_shift if p[key]) / n_s) if n_s else math.nan  # noqa: E731
    counts = {k: sum(1 for p in per_shift if p["outcome"] == k)
              for k in ("correct", "wrong-mechanism", "miss")}
    counts["false-alarm"] = len(false_alarms)
    return {
        "per_shift": per_shift,
        "n_shifts": n_s,
        "n_shifts_excluded": excluded,
        "n_resets": len(resets),
        "confusion": counts,
        "accuracy_delay0": frac("correct_delay0"),                 # P7 primary
        "recall_window": (counts["correct"] / n_s) if n_s else math.nan,
        "recall_any": frac("detected"),
        "misattribution_rate": frac("others_reset_in_window"),     # per shift
        "attribution_precision": (len(matched) / len(resets)) if resets else math.nan,
        "n_misattributed_resets": len(misattributed),
        "false_reset_rate": (len(false_alarms) / n_tested_stable) if n_tested_stable else math.nan,
        "n_tested_stable_mechanism_episodes": n_tested_stable,
        "delays": [p["delay"] for p in per_shift],
        "detected": [p["detected"] for p in per_shift],
    }


# =========================================================================== 11. metric 9: internal objectives

def internal_objectives(raw: dict) -> dict:
    """SPEC Metric 9 per episode: self_score_t and the within-episode decrease of the variance
    objective, var_obj_before - var_obj_after (INTERFACES.md section 2 decision), with means
    over the episodes where they are finite."""
    ss = np.asarray(raw["self_score"], dtype=np.float64)
    dec = np.asarray(raw["var_obj_before"], dtype=np.float64) - np.asarray(raw["var_obj_after"], dtype=np.float64)
    return {"self_score": ss, "var_obj_decrease": dec,
            "self_score_mean": float(np.nanmean(ss)) if np.isfinite(ss).any() else math.nan,
            "var_obj_decrease_mean": float(np.nanmean(dec)) if np.isfinite(dec).any() else math.nan}


# =========================================================================== 12. analysis helpers

def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def bootstrap_means(values, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Means of `n_boot` resamples (with replacement) of the finite entries of `values`."""
    v = _finite(values)
    if v.size == 0:
        return np.full(n_boot, math.nan)
    rng = np.random.default_rng([seed, 6])           # analysis stream, separate from the run keys
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    return v[idx].mean(axis=1)


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of the mean over per-seed scalars (SPEC "Analysis plan")."""
    v = _finite(values)
    if v.size == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    bm = bootstrap_means(v, n_boot, seed)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def bootstrap_pvalue(values, threshold: float, direction: str,
                     n_boot: int = N_BOOT, seed: int = 0) -> float:
    """One-sided bootstrap p-value for "mean > threshold" (direction '>') or "< threshold"
    ('<'): the fraction of bootstrap means on the wrong side, with the (k + 1) / (B + 1)
    correction so a p-value is never exactly 0."""
    bm = bootstrap_means(values, n_boot, seed)
    if not np.isfinite(bm).any():
        return math.nan
    wrong = (bm <= threshold) if direction == ">" else (bm >= threshold)
    return float((wrong.sum() + 1) / (n_boot + 1))


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired percentile bootstrap of mean(a - b) over seeds: both agents' scalars from the
    same seed are resampled together, which is where the common random numbers pay off."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap needs one scalar per seed for both agents")
    return bootstrap_ci(a - b, n_boot, seed, alpha)


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down correction.  Returns {name: {"p": p, "p_adj": adjusted, "reject": bool}}.
    NaN p-values are left out of the family (not testable), reported with reject = False."""
    names = [k for k, p in pvalues.items() if p is not None and math.isfinite(p)]
    m = len(names)
    order = sorted(names, key=lambda k: pvalues[k])
    out, running = {}, 0.0
    for rank, k in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[k])
        running = max(running, adj)                  # monotone step-down
        out[k] = {"p": float(pvalues[k]), "p_adj": float(running), "reject": bool(running <= alpha)}
    for k, p in pvalues.items():
        if k not in out:
            out[k] = {"p": math.nan if p is None else float(p), "p_adj": math.nan, "reject": False}
    return out


def cluster_bootstrap(events_by_seed: list, stat_fn, n_boot: int = N_BOOT, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Seed-level cluster bootstrap for event-pooled statistics (recovery, delay, attribution).

    `events_by_seed` is a list with one entry per seed (any object, e.g. a list of events);
    `stat_fn(list_of_entries) -> float` computes the pooled statistic.  Seeds are resampled
    with replacement and their events pooled, so the CI respects within-seed dependence.
    """
    clusters = [c for c in events_by_seed if c is not None]
    n = len(clusters)
    if n == 0:
        return {"point": math.nan, "lo": math.nan, "hi": math.nan, "n_seeds": 0}
    point = float(stat_fn(clusters))
    rng = np.random.default_rng([seed, 7])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn([clusters[i] for i in idx])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return {"point": point, "lo": math.nan, "hi": math.nan, "n_seeds": n}
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_seeds": n}


def pooled_recovery(per_seed_recoveries: list) -> dict:
    """Pool per-seed lists of {"time", "recovered"} dicts into one KM summary (use with
    cluster_bootstrap via `lambda cl: pooled_recovery(cl)["median"]`)."""
    events = [e for cl in per_seed_recoveries for e in cl]
    return recovery_summary([e["time"] for e in events], [e["recovered"] for e in events])


# =========================================================================== 13. primaries P1-P9

NotComputable = tuple[float, str]      # (NaN, reason)


def _need(runs: dict, *names: str) -> str | None:
    missing = [n for n in names if n not in runs]
    return f"missing agent(s): {', '.join(missing)}" if missing else None


def _episode_range(T: int, lo: int, hi: int) -> np.ndarray:
    """Boolean [T] selector for episode labels lo..hi inclusive."""
    t = np.arange(1, T + 1)
    return (t >= lo) & (t <= hi)


def primary_P1(runs: dict, W: int = 3) -> tuple[float, str | None]:
    """P1 (see is not do).  Per seed: the gap mean_t(nMSE_obs-window(t) - F_obs(t)) over stable
    episodes at least W episodes after the last shift (and with W episodes of data, t >= W),
    minus the tolerance 0.1 mean F_obs + 0.01, in absolute value:

        value = |mean(nMSE - F_obs)| - (0.1 * mean(F_obs) + 0.01)

    The prediction holds for a seed iff value <= 0 (the gap lies inside the band); the family
    test is "fraction of seeds with value <= 0 >= 0.8".  `W` is obs-window's tuned window
    (constants.json); the raw file does not carry it, so the caller passes it.
    """
    if (why := _need(runs, "obs-window")):
        return math.nan, why
    raw = runs["obs-window"]
    T = dims(raw)[0]
    kinds = episode_kinds(raw)
    last_shift = -10 ** 6
    keep = np.zeros(T, dtype=bool)
    for t in range(1, T + 1):
        if raw["shift_j"][t - 1] >= 0:
            last_shift = t
        keep[t - 1] = kinds["stable"][t - 1] and (t - last_shift >= W) and t >= W
    if not keep.any():
        return math.nan, "no stable episode at least W after the last shift"
    series = nmse_series(raw)[keep]
    F = np.asarray(raw["F_obs"], dtype=np.float64)[keep]
    gap = float(np.mean(series - F))
    return abs(gap) - (0.1 * float(F.mean()) + 0.01), None


def primary_P2(runs: dict) -> tuple[float, str | None]:
    """P2 (recovery).  Per seed: mean over qualifying shifts of
    regret(mech-full) - 2 * regret(mech-oracle-detect), both against oracle-structure.
    Prediction: <= 0."""
    if (why := _need(runs, "mech-full", "mech-oracle-detect", "oracle-structure")):
        return math.nan, why
    shifts = qualifying_shifts(runs["mech-full"])[0]
    if not shifts:
        return math.nan, "no shift with magnitude >= 0.05"
    ref = nmse_series(runs["oracle-structure"])
    a, b = nmse_series(runs["mech-full"]), nmse_series(runs["mech-oracle-detect"])
    vals = [post_shift_regret(a, ref, s.t) - 2 * post_shift_regret(b, ref, s.t) for s in shifts]
    return float(np.mean(vals)), None


def primary_P3(runs: dict) -> tuple[float, str | None]:
    """P3(a) (calibration).  Per seed: mech-full's 90 % coverage in stable episodes with
    SHD = 0, minus 0.90.  Prediction: the CI of the mean lies inside [-0.05, 0.05]."""
    if (why := _need(runs, "mech-full")):
        return math.nan, why
    raw = runs["mech-full"]
    eps = episode_kinds(raw)["stable"] & (shd_series(raw) == 0)
    if not eps.any():
        return math.nan, "no stable episode with SHD = 0"
    return coverage(raw, episode_prediction_mask(raw, eps))["cov90"] - 0.90, None


def primary_P4(runs: dict) -> tuple[float, str | None]:
    """P4 (memory that does not overwrite).  Per seed: mean over qualifying visible shifts of
    bump(mech-reset-all) - bump(mech-full) in the shift episode.  Prediction: >= 0.04."""
    if (why := _need(runs, "mech-reset-all", "mech-full")):
        return math.nan, why
    shifts = qualifying_shifts(runs["mech-full"], visible_only=True)[0]
    if not shifts:
        return math.nan, "no visible shift with magnitude >= 0.05"
    vals = [forgetting_bump(runs["mech-reset-all"], s.t, s.j) - forgetting_bump(runs["mech-full"], s.t, s.j)
            for s in shifts]
    return float(np.nanmean(vals)), None


def primary_P5(runs: dict, budget: int = 10) -> tuple[float, str | None]:
    """P5(a) (value of structure).  Per seed, at B = `budget` (the sweep dict, default 10):
    paired difference in mean nMSE over episodes 20-60, int-pairwise - mech-full.
    Prediction: > 0 at B = 10 (and within +/-0.02 at B = 100)."""
    if (why := _need(runs, "int-pairwise", "mech-full")):
        return math.nan, why
    Bs = {scalar(r, "B") for r in (runs["int-pairwise"], runs["mech-full"]) if "B" in r}
    if Bs and Bs != {budget}:
        return math.nan, f"runs are at B = {sorted(Bs)}, not {budget}"
    T = dims(runs["mech-full"])[0]
    eps = _episode_range(T, 20, 60)
    if not eps.any():
        return math.nan, "no episode in 20-60"
    diff = nmse_series(runs["int-pairwise"])[eps].mean() - nmse_series(runs["mech-full"])[eps].mean()
    return float(diff), None


def primary_P6(runs: dict) -> tuple[float, str | None]:
    """P6(iii) (hidden confounder).  Per seed: mech-full's 90 % coverage on do(A) -> B queries
    over episodes 40-60.  Prediction: CI upper bound < 0.5."""
    if (why := _need(runs, "mech-full")):
        return math.nan, why
    raw = runs["mech-full"]
    A, B = int(scalar(raw, "hidden_A")), int(scalar(raw, "hidden_B"))
    if A < 0 or B < 0:
        return math.nan, "not a hidden-variant run"
    T = dims(raw)[0]
    mask = pair_mask(raw, A, B) & episode_prediction_mask(raw, _episode_range(T, 40, 60))
    if not mask.any():
        return math.nan, "no do(A) -> B query in episodes 40-60"
    return coverage(raw, mask)["cov90"], None


def primary_P7(runs: dict) -> tuple[float, str | None]:
    """P7 (credit assignment).  Per seed: fraction of qualifying visible shifts that mech-full
    attributed to the correct mechanism with delay 0.  Prediction: CI lower bound >= 0.8."""
    if (why := _need(runs, "mech-full")):
        return math.nan, why
    ca = credit_assignment(runs["mech-full"])
    if ca["n_shifts"] == 0:
        return math.nan, "no visible shift with magnitude >= 0.05"
    return ca["accuracy_delay0"], None


def primary_P8(runs: dict) -> tuple[float, str | None]:
    """P8 (gameable vs non-gameable objective).  Per seed: self_score(mech-full) -
    self_score(mech-overconfident) averaged over episodes 10-60.  Prediction: > 0."""
    if (why := _need(runs, "mech-full", "mech-overconfident")):
        return math.nan, why
    T = dims(runs["mech-full"])[0]
    eps = _episode_range(T, 10, 60)
    a = np.asarray(runs["mech-full"]["self_score"], dtype=np.float64)[eps]
    b = np.asarray(runs["mech-overconfident"]["self_score"], dtype=np.float64)[eps]
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return math.nan, "self_score is NaN in every episode 10-60"
    return float(np.mean(a[ok] - b[ok])), None


def primary_P9(runs: dict) -> tuple[float, str | None]:
    """P9 (noise-only shifts).  Per seed on a noise-only run: mech-full's regret against
    oracle-structure over the shift episode and the three following, averaged over shifts
    (all noise-only shifts with delta_sigma >= 0.05).  Prediction: >= 0.02."""
    if (why := _need(runs, "mech-full", "oracle-structure")):
        return math.nan, why
    raw = runs["mech-full"]
    if "shift_type" in raw and scalar(raw, "shift_type") != "noise-only":
        return math.nan, f"run is shift_type = {scalar(raw, 'shift_type')!r}, not noise-only"
    shifts = qualifying_shifts(raw)[0]
    if not shifts:
        return math.nan, "no noise-only shift with delta_sigma >= 0.05"
    a, ref = nmse_series(raw), nmse_series(runs["oracle-structure"])
    return float(np.mean([post_shift_regret(a, ref, s.t) for s in shifts])), None


@dataclass(frozen=True)
class PrimarySpec:
    """How a per-seed primary scalar is turned into a one-sided p-value for Holm.

    kind 'mean': H1 is mean {direction} threshold.
    kind 'fraction': H1 is (fraction of seeds with value <= 0) > threshold (P1).
    kind 'interval': H1 is mean inside [lo, hi]; p = max of the two one-sided p-values (P3).
    'ci_upper < x' and 'ci_lower >= x' in the SPEC are the one-sided tests '<' / '>' here.
    """
    name: str
    fn: object
    kind: str
    direction: str
    threshold: float
    run: str            # which results directory feeds it
    description: str


PRIMARIES: dict[str, PrimarySpec] = {
    "P1": PrimarySpec("P1", primary_P1, "fraction", ">", 0.8, "main",
                      "obs-window nMSE gap to F_obs inside the 10 % + 0.01 band in >= 80 % of seeds"),
    "P2": PrimarySpec("P2", primary_P2, "mean", "<", 0.0, "main",
                      "regret(mech-full) - 2 regret(mech-oracle-detect) <= 0"),
    "P3": PrimarySpec("P3", primary_P3, "interval", "", 0.05, "main",
                      "mech-full 90 % coverage (stable, SHD = 0) within +/-5 points of nominal"),
    "P4": PrimarySpec("P4", primary_P4, "mean", ">", 0.04, "main",
                      "forgetting bump: mech-reset-all - mech-full >= 0.04"),
    "P5": PrimarySpec("P5", primary_P5, "mean", ">", 0.0, "sweep_B10",
                      "nMSE(int-pairwise) - nMSE(mech-full) > 0 at B = 10, episodes 20-60"),
    "P6": PrimarySpec("P6", primary_P6, "mean", "<", 0.5, "hidden",
                      "mech-full 90 % coverage on do(A) -> B, episodes 40-60, CI upper < 0.5"),
    "P7": PrimarySpec("P7", primary_P7, "mean", ">", 0.8, "main",
                      "mech-full delay-0 attribution accuracy, CI lower >= 0.8"),
    "P8": PrimarySpec("P8", primary_P8, "mean", ">", 0.0, "main",
                      "self_score(mech-full) - self_score(mech-overconfident) > 0"),
    "P9": PrimarySpec("P9", primary_P9, "mean", ">", 0.02, "noise-only",
                      "mech-full regret vs oracle-structure on noise-only shifts >= 0.02"),
}


def per_seed_values(spec: PrimarySpec, runs_by_seed: dict, **kwargs) -> dict:
    """Apply a primary to {seed: {agent: raw}}; returns {"values": [..], "reasons": {seed: why}}."""
    values, reasons = [], {}
    for seed in sorted(runs_by_seed):
        v, why = spec.fn(runs_by_seed[seed], **kwargs)
        values.append(v)
        if why:
            reasons[int(seed)] = why
    return {"values": values, "reasons": reasons}


def evaluate_primary(spec: PrimarySpec, values, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Unadjusted 95 % CI plus the one-sided p-value for `spec` (Holm is applied across
    primaries by the caller with `holm`)."""
    v = _finite(values)
    out = {"name": spec.name, "kind": spec.kind, "n_seeds": int(v.size), "ci": bootstrap_ci(v, n_boot, seed)}
    if v.size == 0:
        out["p"] = math.nan
        return out
    if spec.kind == "mean":
        out["p"] = bootstrap_pvalue(v, spec.threshold, spec.direction, n_boot, seed)
    elif spec.kind == "fraction":
        held = (v <= 0).astype(float)                       # per-seed "inside the band"
        out["fraction"] = float(held.mean())
        out["p"] = bootstrap_pvalue(held, spec.threshold, ">", n_boot, seed)
    elif spec.kind == "interval":
        p_lo = bootstrap_pvalue(v, -spec.threshold, ">", n_boot, seed)
        p_hi = bootstrap_pvalue(v, spec.threshold, "<", n_boot, seed)
        out["p"] = max(p_lo, p_hi)
    else:
        raise ValueError(f"unknown primary kind {spec.kind!r}")
    return out


def evaluate_primaries(values_by_name: dict, alpha: float = 0.05, n_boot: int = N_BOOT,
                       seed: int = 0) -> dict:
    """Evaluate every primary in `values_by_name` ({name: per-seed values}) and apply Holm.
    A prediction "held" iff its Holm-adjusted test rejects (SPEC "Analysis plan")."""
    results = {name: evaluate_primary(PRIMARIES[name], vals, n_boot, seed)
               for name, vals in values_by_name.items()}
    corrected = holm({name: r["p"] for name, r in results.items()}, alpha)
    for name, r in results.items():
        r.update(corrected[name])
        r["held"] = bool(r["reject"])
    return results
