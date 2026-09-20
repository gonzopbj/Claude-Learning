"""The world of the mechanism-shift testbed.

Implements SPEC.md sections "World" (variables and graph, exact interventional means, shift
process, hidden-confounder variant, RNG streams) and the P1 quantity `F_obs`, against the
contract of INTERFACES.md section 1.

What lives here
  * `World`            the linear-Gaussian SCM per seed, its pre-drawn shift schedule, the
                       keyed noise / query streams, exact ground truth and reference scales
  * `ShiftRecord`      one logged shift: (t, j, shift_type, m, delta, delta_sigma)
  * `WorldView`        what a non-oracle agent is handed: the public dimensions and nothing else
  * `OracleAccess`     what the four oracle agents are handed (holds the World privately)
  * `RelabelledWorld`  a wrapper that renames the variables of a World (SPEC test 3 helper)
  * `structural_sample`  the structural equation, vectorized over rows

Conventions (INTERFACES.md section 0): children in rows, `W[j, p]` is the weight of edge
`p -> j`; episodes are labelled t = 1..T; variable indices are the relabelled visible indices;
the hidden variable H has index d in the generating SCM and never appears in agent-facing
arrays.  All randomness comes from `numpy.random.default_rng` with the keyed streams of the
SPEC; `np.random.*` global functions are never used.

This module imports nothing from `agents/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- constants

SHIFT_TYPES = ("large", "small", "noise-only")
K_SHIFTS = 8                 # SPEC "Shift process": exactly K = 8 shift episodes
SHIFT_WINDOW = (7, 55)       # ... in [7, 55]
MIN_GAP = 5                  # ... with minimum gap 5 between consecutive shift episodes
HIDDEN_SHIFT_INDEX = 3       # hidden variant: the 4th shift (0-based index 3) is the H shift
E_V2 = 4.0 / 3.0             # E[v^2] for v ~ U(-2, 2); E[v] = 0 (used for m, F_obs, F_conf)
SMALL_SHIFT_SD = 0.25        # "small": w_new = w + 0.25 N(0, 1), redrawn if |w_new| < 0.25
SMALL_SHIFT_MIN_ABS = 0.25


# --------------------------------------------------------------------------- data classes

@dataclass(frozen=True)
class ShiftRecord:
    """One shift, as logged by the world (SPEC "Shift process": (seed, t, j, type, m, delta))."""
    t: int              # episode label at whose START the mechanism was resampled
    j: int              # mechanism index: 0..d-1 visible, or d for the hidden H shift
    shift_type: str     # "large" | "small" | "noise-only" (the H shift is "large")
    m: float            # effective magnitude E_q[(truth_after - truth_before)^2] / Var_ref(X_k)
    delta: float        # ||Theta_new - Theta_old||_F over the visible block
    delta_sigma: float  # |sigma_new[j] - sigma_old[j]|


@dataclass(frozen=True)
class WorldView:
    """Everything a non-oracle agent may know about the world: the public dimensions.

    Deliberately a plain frozen dataclass of five integers and no reference to the World
    (INTERFACES.md section 1.3, Decision).  The observational batch and the interventional
    rows reach the agent through the protocol's `observe()` / `receive_intervention()` hooks,
    so nothing here can leak W, b, sigma, the permutation or the schedule.
    """
    d: int
    B: int
    T: int
    n_obs: int
    n_queries: int


# --------------------------------------------------------------------------- pure helpers

def sign_magnitude_weights(rng: np.random.Generator, size) -> np.ndarray:
    """Edge weights `s * u`, u ~ U[0.5, 1.5], s = +-1 with probability 1/2 (SPEC "Variables
    and graph").  Draw order: all magnitudes, then all signs."""
    u = rng.uniform(0.5, 1.5, size)
    s = np.where(rng.random(size) < 0.5, -1.0, 1.0)
    return s * u


def structural_sample(W: np.ndarray, b: np.ndarray, sigma: np.ndarray, hetero: int,
                      order, eps: np.ndarray, clamp: tuple[int, float] | None = None
                      ) -> np.ndarray:
    """Push standard-normal noise `eps` (n, d_full) through the structural equation

        X_j = W[j, :] . X + b_j + sigma_j (1 + gamma_h |W[j, :] . X|) eps_j      (SPEC)

    visiting the variables in the topological `order`.  With `clamp = (i, v)` the equation
    of X_i is replaced by X_i := v (the mutilated SCM of do(X_i = v)).  Returns (n, d_full).
    """
    eps = np.asarray(eps, dtype=np.float64)
    X = np.zeros_like(eps)
    for j in order:
        if clamp is not None and j == clamp[0]:
            X[:, j] = clamp[1]
            continue
        lin = X @ W[j]                                   # W[j, :] . X for every row
        scale = sigma[j] * (1.0 + hetero * np.abs(lin))  # heteroscedasticity knob
        X[:, j] = lin + b[j] + scale * eps[:, j]
    return X


def _generate_visible_scm(rng: np.random.Generator, d: int):
    """SPEC "Variables and graph": ER DAG in a generation order, weights, intercepts, noise
    scales, then the relabelling permutation applied to everything.

    Returns (adj, W, b, sigma, perm) in relabelled indices; `perm[g]` is the relabelled index
    of the g-th variable in generation (topological) order, so `perm` is a topological order.
    """
    p_edge = 3.0 / (d - 1)                    # expected in-degree 1.5
    while True:
        # adj_gen[j, p] = True iff edge p -> j with p < j in the generation order
        adj_gen = np.tril(rng.random((d, d)) < p_edge, k=-1)
        if adj_gen.any():                     # resample if the DAG has no edges
            break
    W_gen = np.where(adj_gen, sign_magnitude_weights(rng, (d, d)), 0.0)
    b_gen = rng.uniform(-1.0, 1.0, d)
    sigma_gen = rng.uniform(0.5, 1.5, d)
    perm = rng.permutation(d)                 # relabelling, from the same stream (SPEC)

    # Apply the relabelling: generation node g becomes index perm[g].
    W = np.zeros((d, d))
    W[np.ix_(perm, perm)] = W_gen
    adj = np.zeros((d, d), dtype=bool)
    adj[np.ix_(perm, perm)] = adj_gen
    b = np.empty(d)
    b[perm] = b_gen
    sigma = np.empty(d)
    sigma[perm] = sigma_gen
    return adj, W, b, sigma, perm


def _draw_shift_episodes(rng: np.random.Generator, batch: int = 512) -> np.ndarray:
    """Exactly K_SHIFTS distinct episodes in SHIFT_WINDOW with consecutive gaps >= MIN_GAP,
    rejection-sampled as the SPEC says.

    The acceptance rate is about 5e-4 (C(21, 8) / C(49, 8)), so candidates are drawn in
    batches: each row of `argsort(uniforms)` is a uniform random permutation, its first
    K_SHIFTS entries a uniform random subset; the first acceptable row in draw order is kept,
    which is the same distribution as one-at-a-time rejection, at a fraction of the cost.
    """
    lo, hi = SHIFT_WINDOW
    candidates = np.arange(lo, hi + 1)
    while True:
        subsets = np.argsort(rng.random((batch, len(candidates))), axis=1)[:, :K_SHIFTS]
        episodes = np.sort(candidates[subsets], axis=1)             # (batch, K)
        ok = np.all(np.diff(episodes, axis=1) >= MIN_GAP, axis=1)
        if ok.any():
            return episodes[np.argmax(ok)]


@dataclass
class _VisibleShiftPlan:
    """The outcome of one visible shift draw, stored so it can be re-applied to a different
    chain of states (the hidden variant skips the 4th visible shift; see World.__init__)."""
    j: int
    shift_type: str
    W_row: np.ndarray      # new row j of W after the shift (absolute; large / noise-only)
    dW_row: np.ndarray     # additive change of row j (small shifts are relative to the old w)
    b_j: float
    sigma_j: float

    def apply(self, W: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> int:
        j = self.j
        if self.shift_type == "small":
            W[j] = W[j] + self.dW_row
        else:
            W[j] = self.W_row
        b[j] = self.b_j
        sigma[j] = self.sigma_j
        return j


# --------------------------------------------------------------------------- one SCM state

class _SCMState:
    """One fixed SCM (W, b, sigma) on all d_full nodes, with the cached quantities of SPEC
    "Exact interventional means":

        Theta = (I - W)^-1            total-effect matrix
        mu    = Theta b               observational mean
        Sigma = Theta diag(sigma^2) Theta^T     observational covariance (gamma_h = 0),
                                                 the documented reference scale Sigma_ref
        C[:, i] = c(i) = (I - W_i)^-1 b_i        intercept term of do(X_i = .), visible i only

    so that truth(i, v) = c(i) + Theta[:, i] v is affine in v and costs O(d) per query.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, sigma: np.ndarray, d: int):
        self.W = np.array(W, dtype=np.float64, copy=True)
        self.b = np.array(b, dtype=np.float64, copy=True)
        self.sigma = np.array(sigma, dtype=np.float64, copy=True)
        self.d = int(d)
        n = self.W.shape[0]
        eye = np.eye(n)
        self.Theta = np.linalg.inv(eye - self.W)
        self.mu = self.Theta @ self.b
        self.Sigma = self.Theta @ np.diag(self.sigma ** 2) @ self.Theta.T
        # c(i) needs the mutilated solve: W_i is W with row i zeroed, b_i is b with b_i := 0.
        self.C = np.zeros((n, self.d))
        for i in range(self.d):
            W_i = self.W.copy()
            W_i[i] = 0.0
            b_i = self.b.copy()
            b_i[i] = 0.0
            self.C[:, i] = np.linalg.solve(eye - W_i, b_i)

    def truth(self, i: int, v: float) -> np.ndarray:
        """E[X | do(X_i = v)] on all d_full coordinates."""
        return self.C[:, i] + self.Theta[:, i] * v


def _effective_magnitude(old: _SCMState, new: _SCMState, d: int) -> float:
    """m = E_q[(truth_after - truth_before)^2] / Var_ref(X_k), averaged over i uniform,
    v ~ U(-2, 2), all visible k != i, exactly (SPEC "Shift process").

    truth_new - truth_old at (i, k, v) = dC[k, i] + dTheta[k, i] v, so
    E_v[(.)^2] = dC^2 + dTheta^2 E[v^2] because E[v] = 0.  Var_ref is that of the SCM after
    the shift (INTERFACES 1.2).
    """
    dC = (new.C - old.C)[:d, :]                    # rows k, columns i
    dT = (new.Theta - old.Theta)[:d, :d]
    sq = dC ** 2 + dT ** 2 * E_V2
    var_ref = np.diag(new.Sigma)[:d]
    ratio = sq / var_ref[:, None]                  # divide by Var_ref(X_k)
    off = ~np.eye(d, dtype=bool)
    return float(ratio[off].mean())


# --------------------------------------------------------------------------- the world

class World:
    """One seed's world: SCM, shift schedule, keyed streams, exact truth (INTERFACES 1.1).

    Construction does everything random up front, from streams [seed, 0] (graph) and
    [seed, 1] (schedule), so the whole run is fixed before any agent exists.  The per-episode
    streams [seed, t, 2/3/4] are opened on demand and are pure functions of (seed, t).
    """

    def __init__(self, seed: int, d: int = 6, T: int = 60, B: int = 50, n_obs: int = 200,
                 shift_type: str = "large", hetero: int = 0, hidden: bool = False,
                 n_queries: int = 200, shifts_enabled: bool = True):
        if shift_type not in SHIFT_TYPES:
            raise ValueError(f"shift_type must be one of {SHIFT_TYPES}, got {shift_type!r}")
        if hetero not in (0, 1):
            raise ValueError(f"hetero (gamma_h) must be 0 or 1, got {hetero!r}")
        if d < 2:
            raise ValueError("need at least two variables")
        if T < SHIFT_WINDOW[1]:
            raise ValueError(f"T={T} is shorter than the shift window {SHIFT_WINDOW}; the "
                             "SPEC's schedule needs T >= 55")
        self.seed = int(seed)
        self.d = int(d)
        self.T = int(T)
        self.B = int(B)
        self.n_obs = int(n_obs)
        self.n_queries = int(n_queries)
        self.shift_type = str(shift_type)
        self.hetero = int(hetero)
        self.hidden = bool(hidden)
        self.shifts_enabled = bool(shifts_enabled)
        self.d_full = self.d + 1 if self.hidden else self.d

        # ---- stream [seed, 0]: graph, weights, intercepts, noise scales, permutation, hidden
        rng0 = np.random.default_rng([self.seed, 0])
        adj, W, b, sigma, perm = _generate_visible_scm(rng0, self.d)
        self._adj = adj                       # visible DAG, children in rows, fixed per seed
        self._perm = perm
        self._pair = None                     # (A, B) in the hidden variant
        if self.hidden:
            # Decision (INTERFACES 1.1): H's edge choice and parameters are drawn AFTER all
            # visible quantities, so the visible SCM is byte-identical to the main run.
            W, b, sigma = self._add_hidden_confounder(rng0, W, b, sigma)
            # H (index d) has no parents, so it comes first in the topological order.
            self._order = [self.d] + [int(g) for g in perm]
        else:
            self._order = [int(g) for g in perm]

        # ---- stream [seed, 1]: shift episodes and every shift's parameters, up front
        rng1 = np.random.default_rng([self.seed, 1])
        episodes = _draw_shift_episodes(rng1)
        # First the K visible shifts, drawn along a chain exactly as the main run draws them
        # (each draw depends on the chain only through "small" shifts, which perturb the
        # current weight).  Then, in the hidden variant, H's new (b_H, sigma_H) for the 4th
        # shift.  Decision: drawing H's parameters AFTER the visible chain keeps shifts 1-3
        # and 5-8 identical to the main run's for the same seed (SPEC: "the shift schedule is
        # as in the main run, except that the 4th shift ... is a large-type shift of H").
        W_c, b_c, sigma_c = W.copy(), b.copy(), sigma.copy()
        plans = [self._draw_visible_shift(rng1, W_c, b_c, sigma_c) for _ in range(K_SHIFTS)]
        h_shift = (rng1.uniform(-1.0, 1.0), rng1.uniform(0.5, 1.5)) if self.hidden else None

        # Build the K + 1 successive SCM states actually used, and log every shift exactly.
        states = [_SCMState(W, b, sigma, self.d)]
        records: list[ShiftRecord] = []
        for k, t in enumerate(episodes):
            prev = states[-1]
            W_k, b_k, sigma_k = prev.W.copy(), prev.b.copy(), prev.sigma.copy()
            if self.hidden and k == HIDDEN_SHIFT_INDEX:
                j, shift_type = self.d, "large"           # H: b_H and sigma_H resampled
                b_k[j], sigma_k[j] = h_shift
            else:
                j, shift_type = plans[k].apply(W_k, b_k, sigma_k), self.shift_type
            new = _SCMState(W_k, b_k, sigma_k, self.d)
            states.append(new)
            records.append(ShiftRecord(
                t=int(t), j=j, shift_type=shift_type,
                m=_effective_magnitude(prev, new, self.d),
                delta=float(np.linalg.norm((new.Theta - prev.Theta)[:self.d, :self.d])),
                delta_sigma=float(abs(new.sigma[j] - prev.sigma[j]))))
        # The stream is consumed identically either way; with shifts disabled (lambda
        # calibration) the schedule is empty and the episode-1 SCM is kept for the whole run.
        if self.shifts_enabled:
            self._states, self.schedule = states, records
        else:
            self._states, self.schedule = states[:1], []

        # ---- current-episode bookkeeping (apply_shift advances these)
        self._t = 0                           # last episode label applied
        self._state_idx = 0
        self._current = self._states[0]
        self._noise_cache: tuple[int, np.ndarray] | None = None   # (t, interventional noise)

    # ------------------------------------------------------------------ construction helpers

    def _add_hidden_confounder(self, rng, W, b, sigma):
        """SPEC "Hidden-confounder variant": pick a visible edge A -> B uniformly, add H with
        H -> A, H -> B (same weight rule), b_H ~ U[-1, 1], sigma_H ~ U[0.5, 1.5]."""
        d = self.d
        edges = np.argwhere(self._adj)                    # rows (child j, parent p)
        B_, A_ = (int(x) for x in edges[rng.integers(len(edges))])
        w_HA, w_HB = sign_magnitude_weights(rng, 2)
        b_H = rng.uniform(-1.0, 1.0)
        sigma_H = rng.uniform(0.5, 1.5)
        self._pair = (A_, B_)
        W_full = np.zeros((d + 1, d + 1))
        W_full[:d, :d] = W
        W_full[A_, d] = w_HA
        W_full[B_, d] = w_HB
        return W_full, np.append(b, b_H), np.append(sigma, sigma_H)

    def _draw_visible_shift(self, rng, W, b, sigma) -> _VisibleShiftPlan:
        """Draw one shift of `self.shift_type` of a visible mechanism (SPEC "Shift process"),
        apply it in place to the chain (W, b, sigma) and return it as a re-applicable plan.

        Candidates and parents come from the VISIBLE graph, so j is chosen identically in the
        main and hidden variants for the same seed.  In the hidden variant a large shift of j
        leaves the weight of the H -> j edge (if any) alone: that edge belongs to the
        confounding motif, not to the mechanism under study.
        """
        d = self.d
        if self.shift_type == "noise-only":
            j = int(rng.integers(d))                      # uniform among all d variables
            sigma[j] = rng.uniform(0.5, 1.5)              # only sigma_j changes
        else:
            has_parent = np.flatnonzero(self._adj.any(axis=1))
            j = int(has_parent[rng.integers(len(has_parent))])   # uniform among vars with a parent
            parents = np.flatnonzero(self._adj[j])       # ascending relabelled index
            if self.shift_type == "large":
                # same parents, same sign-magnitude rule; b_j and sigma_j resampled too
                W[j, parents] = sign_magnitude_weights(rng, len(parents))
                b[j] = rng.uniform(-1.0, 1.0)
                sigma[j] = rng.uniform(0.5, 1.5)
            else:  # "small": w_new = w + 0.25 N(0, 1), redrawn while |w_new| < 0.25
                dW = np.zeros(W.shape[1])
                for p in parents:
                    while True:
                        w_new = W[j, p] + SMALL_SHIFT_SD * rng.standard_normal()
                        if abs(w_new) >= SMALL_SHIFT_MIN_ABS:
                            break
                    dW[p] = w_new - W[j, p]
                    W[j, p] = w_new
                return _VisibleShiftPlan(j, "small", W[j].copy(), dW, float(b[j]),
                                         float(sigma[j]))
        return _VisibleShiftPlan(j, self.shift_type, W[j].copy(), np.zeros(W.shape[1]),
                                 float(b[j]), float(sigma[j]))

    # ------------------------------------------------------------------ sampling (private)

    def _sample(self, eps: np.ndarray, clamp=None) -> np.ndarray:
        """Push noise (n, d_full) through the CURRENT SCM; returns all d_full coordinates."""
        s = self._current
        return structural_sample(s.W, s.b, s.sigma, self.hetero, self._order, eps, clamp)

    # ------------------------------------------------------------------ episode protocol

    def apply_shift(self, t: int) -> ShiftRecord | None:
        """Step 1: make the SCM of episode t current.  Must be called for t = 1, 2, ..., T in
        order.  Returns the ShiftRecord with .t == t, else None."""
        if t != self._t + 1:
            raise ValueError(f"apply_shift({t}) out of order: last applied t = {self._t}")
        if t > self.T:
            raise ValueError(f"apply_shift({t}) beyond T = {self.T}")
        self._t = t
        if self._state_idx < len(self.schedule) and self.schedule[self._state_idx].t == t:
            rec = self.schedule[self._state_idx]
            self._state_idx += 1
            self._current = self._states[self._state_idx]
            return rec
        return None

    def observational_batch(self, t: int) -> np.ndarray:
        """Step 2: n_obs rows of the current SCM from stream [seed, t, 2]; visible columns."""
        eps = np.random.default_rng([self.seed, int(t), 2]).standard_normal(
            (self.n_obs, self.d_full))
        return self._sample(eps)[:, :self.d]

    def interventional_noise(self, t: int) -> np.ndarray:
        """Step 2: the pre-drawn (B, d_full) noise array of episode t, stream [seed, t, 3]."""
        t = int(t)
        if self._noise_cache is None or self._noise_cache[0] != t:
            eps = np.random.default_rng([self.seed, t, 3]).standard_normal((self.B, self.d_full))
            self._noise_cache = (t, eps)
        return self._noise_cache[1].copy()

    def query_set(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Step 2: Q queries do(X_i = v), i uniform over VISIBLE variables, v ~ U(-2, 2), from
        stream [seed, t, 4].  Draw order: all Q targets, then all Q values."""
        rng = np.random.default_rng([self.seed, int(t), 4])
        query_i = rng.integers(0, self.d, size=self.n_queries)
        query_v = rng.uniform(-2.0, 2.0, size=self.n_queries)
        return query_i, query_v

    def intervene(self, t: int, k: int, i: int, v: float) -> np.ndarray:
        """Step 6: the row of the mutilated current SCM with X_i := v, generated from noise row
        k of episode t WHATEVER (i, v) is (common random numbers).  Pure.  Visible coords."""
        if not 0 <= k < self.B:
            raise ValueError(f"sample index k={k} outside [0, {self.B})")
        if not 0 <= i < self.d:
            raise ValueError(f"intervention target i={i} outside the visible range [0, {self.d})")
        eps = self.interventional_noise(t)[k][None, :]
        row = self._sample(eps, (int(i), float(v)))[0, :self.d]
        row[i] = float(v)                          # exact, not just numerically equal
        return row

    # ------------------------------------------------------------------ ground truth

    def truth(self, i: int, v: float) -> np.ndarray:
        """E[X | do(X_i = v)] under the current SCM (visible coordinates); entry i is v."""
        if not 0 <= i < self.d:
            raise ValueError(f"query target i={i} outside the visible range [0, {self.d})")
        out = self._current.truth(int(i), float(v))[:self.d]
        out[i] = float(v)
        return out

    def var_ref(self) -> np.ndarray:
        """diag(Sigma_ref) of the current SCM, always with gamma_h = 0 (reference scale)."""
        return np.diag(self._current.Sigma)[:self.d].copy()

    def sigma_ref(self) -> np.ndarray:
        return np.sqrt(self.var_ref())

    def mean_obs(self) -> np.ndarray:
        """mu = (I - W)^-1 b, visible coordinates."""
        return self._current.mu[:self.d].copy()

    def F_obs(self) -> float:
        """P1 population see-for-do floor of obs-window's estimand (SPEC P1):

            beta1 = Sigma_ref[j, i] / Sigma_ref[i, i];  beta0 = mu_j - beta1 mu_i
            bias(v) = (beta0 - c_j(i)) + (beta1 - Theta[j, i]) v
            F_obs = mean_{i != j} E_v[bias(v)^2] / Var_ref(X_j),  E[v^2] = 4/3, E[v] = 0.
        """
        d, s = self.d, self._current
        Sig = s.Sigma[:d, :d]
        mu = s.mu[:d]
        Theta = s.Theta[:d, :d]
        C = s.C[:d, :]                                    # C[j, i] = c_j(i)
        var = np.diag(Sig)
        beta1 = Sig / var[None, :]                        # beta1[j, i]
        beta0 = mu[:, None] - beta1 * mu[None, :]         # beta0[j, i]
        bias0 = beta0 - C
        bias1 = beta1 - Theta
        ev_bias_sq = (bias0 ** 2 + bias1 ** 2 * E_V2) / var[:, None]   # / Var_ref(X_j)
        off = ~np.eye(d, dtype=bool)
        return float(ev_bias_sq[off].mean())

    def confounding(self) -> dict | None:
        """Hidden variant only: the population confounding bias on the constructed edge,

            c = beta_A - W[B, A],   beta = population OLS of X_B on [1, X_pa_vis(B)]
            F_conf = c^2 E[v^2] / Var_ref(X_B)                      (SPEC P6)

        from the 7-node observational covariance of the current SCM."""
        if self._pair is None:
            return None
        A_, B_ = self._pair
        s = self._current
        pa = np.flatnonzero(self._adj[B_])                # visible true parents of B (has A)
        Sig = s.Sigma
        beta = np.linalg.solve(Sig[np.ix_(pa, pa)], Sig[pa, B_])   # OLS slopes (intercept
        beta_A = float(beta[list(pa).index(A_)])                    # absorbs the means)
        c = beta_A - float(s.W[B_, A_])
        F_conf = c ** 2 * E_V2 / float(Sig[B_, B_])
        return {"A": A_, "B": B_, "c": c, "F_conf": F_conf}

    def true_adjacency(self) -> np.ndarray:
        """Visible DAG (d, d) bool, children in rows; constant within a seed; never shows H."""
        return self._adj.copy()

    def true_W(self) -> np.ndarray:
        return self._current.W[:self.d, :self.d].copy()

    def true_b(self) -> np.ndarray:
        return self._current.b[:self.d].copy()

    def true_sigma(self) -> np.ndarray:
        return self._current.sigma[:self.d].copy()

    def hidden_pair(self) -> tuple[int, int] | None:
        return self._pair

    def permutation(self) -> np.ndarray:
        """The relabelling: perm[g] is the index of the g-th variable in generation order."""
        return self._perm.copy()

    # ------------------------------------------------------------------ access objects

    def view(self) -> WorldView:
        return WorldView(self.d, self.B, self.T, self.n_obs, self.n_queries)

    def oracle_access(self) -> "OracleAccess":
        return OracleAccess(self)


# --------------------------------------------------------------------------- oracle access

class OracleAccess:
    """Handed to oracle, oracle-structure, mech-oracle-detect and mech-full-diag only
    (INTERFACES 1.4).  Each may use only the method named for it in the registry."""

    def __init__(self, world):
        self._world = world
        self.view = world.view()
        self.d, self.B, self.T = world.d, world.B, world.T
        self.n_obs, self.n_queries = world.n_obs, world.n_queries

    def true_adjacency(self) -> np.ndarray:          # oracle-structure
        return self._world.true_adjacency()

    def truth(self, i: int, v: float) -> np.ndarray:   # oracle
        return self._world.truth(i, v)

    def shifted_mechanism(self, t: int) -> int:      # mech-oracle-detect
        """j of the shift at episode t, -1 if none, d for the H shift."""
        for rec in self._world.schedule:
            if rec.t == t:
                return int(rec.j)
        return -1

    def hidden_pair(self):                           # mech-full-diag
        return self._world.hidden_pair()

    def true_W(self) -> np.ndarray:                  # oracle only
        return self._world.true_W()

    def true_b(self) -> np.ndarray:
        return self._world.true_b()

    def true_sigma(self) -> np.ndarray:
        return self._world.true_sigma()


# --------------------------------------------------------------------------- relabelling

class RelabelledWorld:
    """The same world with its visible variables renamed: new label of old variable a is
    `pi[a]`.  Implements the whole World interface by delegation, so an agent run on it
    through the protocol sees the identical world up to relabelling (SPEC test 3: "running
    mech-full on a relabelled copy of the same world yields identical predictions up to
    relabelling").  Row `k` of episode `t` still uses noise row `k`, so pairing is kept.

    Pass a fresh `World` (one whose apply_shift has not been called) and never touch it again.
    """

    def __init__(self, world: World, pi):
        pi = np.asarray(pi, dtype=np.int64)
        if sorted(pi.tolist()) != list(range(world.d)):
            raise ValueError("pi must be a permutation of range(d)")
        self._inner = world
        self._pi = pi                       # old -> new
        self._inv = np.argsort(pi)          # new -> old
        for name in ("seed", "d", "d_full", "T", "B", "n_obs", "n_queries", "shift_type",
                     "hetero", "hidden", "shifts_enabled"):
            setattr(self, name, getattr(world, name))
        d = world.d
        self.schedule = [
            ShiftRecord(r.t, int(pi[r.j]) if r.j < d else r.j, r.shift_type, r.m, r.delta,
                        r.delta_sigma)
            for r in world.schedule]

    # vectors / matrices indexed by variable: new index n holds old index inv[n]
    def _vec(self, x):
        return np.asarray(x)[self._inv]

    def _mat(self, M):
        return np.asarray(M)[np.ix_(self._inv, self._inv)]

    def apply_shift(self, t):
        rec = self._inner.apply_shift(t)
        if rec is None:
            return None
        return ShiftRecord(rec.t, int(self._pi[rec.j]) if rec.j < self.d else rec.j,
                           rec.shift_type, rec.m, rec.delta, rec.delta_sigma)

    def observational_batch(self, t):
        return self._inner.observational_batch(t)[:, self._inv]

    def interventional_noise(self, t):
        eps = self._inner.interventional_noise(t)
        cols = np.concatenate([self._inv, np.arange(self.d, self.d_full)])
        return eps[:, cols]

    def query_set(self, t):
        qi, qv = self._inner.query_set(t)
        return self._pi[qi], qv

    def intervene(self, t, k, i, v):
        return self._vec(self._inner.intervene(t, k, int(self._inv[i]), v))

    def truth(self, i, v):
        return self._vec(self._inner.truth(int(self._inv[i]), v))

    def var_ref(self):
        return self._vec(self._inner.var_ref())

    def sigma_ref(self):
        return self._vec(self._inner.sigma_ref())

    def mean_obs(self):
        return self._vec(self._inner.mean_obs())

    def F_obs(self):
        return self._inner.F_obs()          # a mean over all ordered pairs: label-free

    def confounding(self):
        conf = self._inner.confounding()
        if conf is None:
            return None
        return {**conf, "A": int(self._pi[conf["A"]]), "B": int(self._pi[conf["B"]])}

    def true_adjacency(self):
        return self._mat(self._inner.true_adjacency())

    def true_W(self):
        return self._mat(self._inner.true_W())

    def true_b(self):
        return self._vec(self._inner.true_b())

    def true_sigma(self):
        return self._vec(self._inner.true_sigma())

    def hidden_pair(self):
        pair = self._inner.hidden_pair()
        return None if pair is None else (int(self._pi[pair[0]]), int(self._pi[pair[1]]))

    def permutation(self):
        return self._pi[self._inner.permutation()]

    def view(self):
        return WorldView(self.d, self.B, self.T, self.n_obs, self.n_queries)

    def oracle_access(self):
        return OracleAccess(self)
