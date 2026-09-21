"""Mechanism-model agents: every `mech-*` agent and `oracle-structure` as one class.

Implements SPEC.md sections "Mechanism models", "Structure learning", "Shift detection",
"Answering do(X_i = v)", "Active intervention rule" and "Self log-score", behind the hooks of
INTERFACES.md section 2.  The variants of the SPEC "Agents" table are flag combinations
(see `MechAgent`); the registry in `agents/__init__.py` fixes them per name.

One episode, in the hook order of `protocol.run_agent`:

    observe(X_obs, t)          keep the batch; open a new chunk of the structure pool
    detect_and_reset()         GLR + Page-Hinkley per mechanism, under the posterior of the
                               buffer BEFORE the batch is appended; apply the reset rule
    refit()                    append the batch to every buffer; var_obj_before
    score_before_reveal(...)   factorized posterior-predictive log density of the realized row
    receive_intervention(...)  append the row to every mechanism except the target
    update_structure(t)        slope tests -> BH -> order extraction -> parent selection
    answer(qi, qv)             S joint posterior draws pushed through the mutilated graph

What a "buffer" is here.  SPEC: posteriors are a deterministic function of (buffer, parent
set, prior).  For a linear regression of X_j on any subset of the other variables, the
weighted second-moment matrix of the augmented row [1, X] - shape (d+1, d+1) - together with
the weight total n_eff is a lossless summary of the buffer: the SPEC's S_xx, S_xy and S_yy
are sub-blocks of it for whatever the current parent set is.  So mechanism j stores that
matrix (`S[j]`) instead of its rows; a reset zeroes it, a parent-set change selects other
sub-blocks, and exponential forgetting (mech-no-detect) is one multiplication per episode.
Buffers differ across mechanisms because rows where j itself was clamped never enter S[j].
The structure pool (interventional rows of the last W_struct episodes) is stored as rows.

Vectorization notes (SPEC "Compute budget": about 30 ms per agent-episode):
  * all d NIG posteriors are computed in one batched inverse of (d, d+1, d+1) padded
    precision matrices (`nig_posterior` is the single-mechanism reference);
  * the d mutilated inverses (I - W_{s,i})^-1 come from the Sherman-Morrison identity
    I - W_{s,i} = (I - W_s) + e_i w_i', so one inverse of (S, d, d) plus einsums replaces the
    SPEC's inverse of shape (d, S, d, d) - `mutilated_inverses` keeps the latter as the
    reference and the tests check agreement to 1e-10;
  * quantiles come from one contiguous sort along the draw axis plus the same linear
    interpolation `np.quantile` uses (5x faster than np.quantile on this shape).

This module must not import `world` (SPEC test 3).
"""

from __future__ import annotations

import collections
import math

import numpy as np
from scipy.special import gammaln, stdtr

from .base import QUANTILE_LEVELS, Agent

# --------------------------------------------------------------------------- fixed constants
# SPEC "Tuning and calibration": these are not tuned.
PRIORS = {                     # name -> (v0, a0, b0) with V0 = v0 * I  (SPEC "Mechanism models")
    "default": (10.0, 2.0, 1.0),
    "overconfident": (0.1, 50.0, 50.0),
}
N_MIN = 100                    # rows since last reset before the detector may run
W_STRUCT = 10                  # structure pool: interventional rows of the last 10 episodes
ALPHA_BH = 0.01                # Benjamini-Hochberg level of the ancestor slope tests
ALPHA_PARENT = 0.05            # per-regressor t-test level of parent selection
S_DRAWS = 1000                 # joint posterior draws per answer
DEFAULT_GAMMA = 0.7            # mech-no-detect forgetting factor when constants.json has none
VAR_WINDOW = 3                 # episodes of observational rows behind the active rule's v_hat
MIN_SLOPE_ROWS = 4             # slope test guard: n >= 4 and >= 2 distinct v
CONST_COL_VAR = 1e-8           # a regressor with smaller variance counts as a constant column


# --------------------------------------------------------------------------- small math helpers

def log_t_pdf(y, loc, scale2, nu):
    """log density of a Student-t with `nu` degrees of freedom, location `loc` and squared
    scale `scale2`, evaluated at `y` (all broadcastable arrays)."""
    z2 = (y - loc) ** 2 / (nu * scale2)
    return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * scale2) - (nu + 1.0) / 2.0 * np.log1p(z2))


def two_sided_t_pvalue(t_stat, df):
    """p-value of a two-sided t-test from the statistic and its degrees of freedom."""
    return 2.0 * stdtr(df, -np.abs(t_stat))


def _t_statistic(beta, se):
    """beta / se with se == 0 mapped to +-inf (beta != 0) or 0 (beta == 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(se > 0, se, 1.0)
        return np.where(se > 0, beta / safe, np.where(beta != 0, np.inf, 0.0))


def benjamini_hochberg(pvals, alpha):
    """Boolean rejection mask of the BH procedure at level `alpha` over a 1-d array."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    thresholds = alpha * np.arange(1, m + 1) / m
    below = p[order] <= thresholds
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    k_max = np.max(np.flatnonzero(below))          # largest k with p_(k) <= k alpha / m
    return p <= p[order][k_max]


def quantiles_of_sorted(x_sorted, levels):
    """Quantiles along the last axis of an array already sorted along it; the same linear
    interpolation as np.quantile's default method.  Returns (..., len(levels))."""
    n = x_sorted.shape[-1]
    pos = np.asarray(levels, dtype=np.float64) * (n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    xl = x_sorted[..., lo]
    xh = x_sorted[..., hi]
    return xl + frac * (xh - xl)


# --------------------------------------------------------------------------- NIG regression

def nig_posterior(S_xx, S_xy, S_yy, n_eff, v0, a0, b0):
    """Single-mechanism reference of the SPEC formulas (used by the tests; the agent computes
    all d mechanisms at once in `MechAgent._posteriors` with the same arithmetic):
        V_n = (V0^-1 + S_xx)^-1,  m_n = V_n S_xy,  a_n = a0 + n_eff / 2,
        b_n = b0 + (S_yy - m_n' V_n^-1 m_n) / 2,   with V0 = v0 I.
    Returns (m_n, V_n, a_n, b_n).  Empty statistics give the prior."""
    p = len(S_xy)
    precision = S_xx + np.eye(p) / v0
    V = np.linalg.solve(precision, np.eye(p))
    m = V @ S_xy
    a = a0 + 0.5 * n_eff
    b = b0 + 0.5 * (S_yy - S_xy @ m)      # m' V^-1 m = S_xy' V S_xy because V V^-1 = I
    return m, V, a, float(b)


def nig_draw(rng, m, V, a, b, n_draws):
    """SPEC convention: sigma2_s = 1 / rng.gamma(shape=a_n, scale=1/b_n),
    w_s = m_n + sqrt(sigma2_s) chol(V_n) z_s.  Returns (w (n_draws, p), sigma2 (n_draws,))."""
    sigma2 = 1.0 / rng.gamma(shape=a, scale=1.0 / b, size=n_draws)
    z = rng.standard_normal((n_draws, len(m)))
    return m + np.sqrt(sigma2)[:, None] * (z @ np.linalg.cholesky(V).T), sigma2


def nig_predictive(X, m, V, a, b):
    """Posterior predictive for design rows X (n, p): Student-t with nu = 2 a_n, location
    x' m_n and squared scale (b_n / a_n)(1 + x' V_n x).  Returns (loc, scale2, nu)."""
    quad = np.einsum("np,pq,nq->n", X, V, X)
    return X @ m, (b / a) * (1.0 + quad), 2.0 * a


class _JointPosterior:
    """The d NIG posteriors in zero-padded form over the augmented coordinates [1, X].

    m (d, d+1): m_n of mechanism j in its design columns, 0 elsewhere.
    V (d, d+1, d+1): V_n in the design block, 0 elsewhere (for x' V x on augmented rows).
    V_chol (d, d+1, d+1): V_n in the block and the identity outside, so a batched Cholesky
        exists; draws outside the block are discarded by `mask`.
    a, b (d,); mask (d, d+1) bool design columns.
    """

    __slots__ = ("m", "V", "V_chol", "a", "b", "mask", "_L")

    def __init__(self, m, V, V_chol, a, b, mask):
        self.m, self.V, self.V_chol, self.a, self.b, self.mask = m, V, V_chol, a, b, mask
        self._L = None

    def draw_joint(self, rng, n_draws):
        """(W_s (S, d, d), b_s (S, d)): row j of W_s holds w_s on pa(j), zeros elsewhere."""
        if self._L is None:
            self._L = np.linalg.cholesky(self.V_chol)
        d = self.m.shape[0]
        sigma2 = 1.0 / rng.gamma(shape=self.a, scale=1.0 / self.b, size=(n_draws, d))
        z = rng.standard_normal((n_draws, d, d + 1))
        w = self.m + np.sqrt(sigma2)[:, :, None] * np.einsum("jpq,sjq->sjp", self._L, z)
        w *= self.mask                                             # zero outside the design
        return np.ascontiguousarray(w[:, :, 1:]), np.ascontiguousarray(w[:, :, 0])

    def predictive_row(self, xa):
        """(loc, scale2, nu), each (d,), of every mechanism's predictive at one augmented
        row xa = [1, x]; used for the self log-score."""
        loc = self.m @ xa
        quad = np.einsum("p,jpq,q->j", xa, self.V, xa)
        return loc, (self.b / self.a) * (1.0 + quad), 2.0 * self.a


# --------------------------------------------------------------------------- structure learning

def slope_test_pvalues(rows, targets, values, d):
    """SPEC "Structure learning" step 1.  P[i, j] = two-sided p-value for slope = 0 in the OLS
    of X_j on [1, v] over the pool rows where X_i was clamped (n - 2 degrees of freedom).
    P[i, i] = 1.  Guards: n >= 4 and >= 2 distinct v, otherwise p = 1."""
    P = np.ones((d, d))
    if len(targets) == 0:
        return P
    targets = np.asarray(targets)
    values = np.asarray(values, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    for i in range(d):
        sel = targets == i
        n = int(sel.sum())
        if n < MIN_SLOPE_ROWS:
            continue
        v = values[sel]
        if np.unique(v).size < 2:
            continue
        Y = rows[sel]                                       # (n, d): all j at once
        X = np.column_stack([np.ones(n), v])                # (n, 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ Y)                          # (2, d)
        resid = Y - X @ beta
        sigma2 = np.sum(resid ** 2, axis=0) / (n - 2)       # (d,)
        se = np.sqrt(sigma2 * XtX_inv[1, 1])
        P[i] = two_sided_t_pvalue(_t_statistic(beta[1], se), n - 2)
        P[i, i] = 1.0
    return P


def estimated_ancestors(P, alpha=ALPHA_BH):
    """BH over the d(d-1) ordered pairs; A_hat[i, j] = True iff i is an estimated ancestor of j."""
    d = P.shape[0]
    off = ~np.eye(d, dtype=bool)
    A_hat = np.zeros((d, d), dtype=bool)
    A_hat[off] = benjamini_hochberg(P[off], alpha)
    return A_hat


def extract_order(A_hat, pi):
    """SPEC "Structure learning" step 2.  Repeat d times: among the not-yet-placed nodes pick
    the one with the fewest not-yet-placed estimated ancestors, ties broken by the agent's
    private permutation `pi` (earlier in pi wins).  Always terminates; equals a topological
    sort when A_hat is consistent.  Returns the order as a list."""
    d = A_hat.shape[0]
    rank = np.empty(d, dtype=np.int64)
    rank[np.asarray(pi)] = np.arange(d)
    remaining = list(range(d))
    order = []
    while remaining:
        rem = np.asarray(remaining)
        counts = A_hat[np.ix_(rem, rem)].sum(axis=0)         # ancestors of each remaining node
        best = min(range(len(rem)), key=lambda k: (counts[k], rank[rem[k]]))
        order.append(int(rem[best]))
        remaining.pop(best)
    return order


def prune_to_order(A_hat, order):
    """Drop every (i, j) in A_hat with i placed after j; the result is acyclic."""
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    return A_hat & (pos[:, None] < pos[None, :])


def ols_parent_test(S, j, candidates, n, alpha):
    """SPEC "Structure learning" step 3 on one mechanism's moment matrix S (d+1, d+1) with
    weight total n: OLS of X_j on [1, candidates], keep candidates with two-sided p < alpha.
    Returns the new parent list, or None when a guard says to keep the previous parent set
    (n < p + 5, or a rank-deficient design after dropping constant columns).  `n` is the
    weight total n_eff: for every agent but mech-no-detect that is the row count; for
    mech-no-detect the moments are gamma-weighted and this is the matching weighted OLS
    (decision: the SPEC defines the parent test "over mechanism j's own buffer", and that
    agent's buffer is its weighted sufficient statistics)."""
    if n <= 0:
        return None
    mean = S[0, 1:] / n
    var = S.diagonal()[1:] / n - mean ** 2
    cands = [int(c) for c in candidates if var[c] > CONST_COL_VAR]   # drop constant columns
    idx = [0] + [c + 1 for c in cands]
    p = len(idx)
    if n < p + 5:
        return None
    Sxx = S[np.ix_(idx, idx)]
    if np.linalg.matrix_rank(Sxx) < p:
        return None
    Sxy = S[idx, j + 1]
    Syy = S[j + 1, j + 1]
    Sxx_inv = np.linalg.inv(Sxx)
    beta = Sxx_inv @ Sxy
    rss = max(float(Syy - beta @ Sxy), 0.0)
    sigma2 = rss / (n - p)
    se = np.sqrt(np.maximum(sigma2 * np.diag(Sxx_inv), 0.0))
    pvals = two_sided_t_pvalue(_t_statistic(beta, se), n - p)
    return [c for c, pv in zip(cands, pvals[1:]) if pv < alpha]


# --------------------------------------------------------------------------- vectorized do()

def mutilated_inverses(W_s):
    """SPEC "Answering do()", reference form.  W_s: (S, d, d) draws, children in rows.
    Returns M[i, s] = (I - W_{s,i})^-1 with row i of W_s zeroed, shape (d, S, d, d), from one
    np.linalg.inv call.  The agent uses `do_components` instead (same numbers, faster)."""
    S, d, _ = W_s.shape
    Wm = np.broadcast_to(W_s, (d, S, d, d)).copy()
    for i in range(d):
        Wm[i, :, i, :] = 0.0
    return np.linalg.inv(np.eye(d) - Wm)


def do_components_reference(W_s, b_s):
    """(c, theta), each (d, S, d), from the explicit mutilated inverses:
    c[i, s] = M[i, s] (b_s with b_i := 0),  theta[i, s, k] = M[i, s, k, i].
    The interventional mean for do(X_i = v) is c[i, s] + v * theta[i, s]."""
    M = mutilated_inverses(W_s)
    d, S = M.shape[0], M.shape[1]
    b_mut = np.broadcast_to(b_s, (d, S, d)).copy()
    for i in range(d):
        b_mut[i, :, i] = 0.0
    return np.einsum("iskl,isl->isk", M, b_mut), np.einsum("iski->isk", M)


def do_components(W_s, b_s):
    """Same (c, theta) as `do_components_reference`, without forming (d, S, d, d).

    Sherman-Morrison: I - W_i = (I - W) + e_i w_i' with w_i the i-th row of W, hence
        (I - W_i)^-1 = Theta - Theta e_i (w_i' Theta) / (1 + w_i' Theta e_i),  Theta = (I - W)^-1.
    (In a DAG w_i' Theta e_i = 0: a parent of i is not a descendant of i.  The denominator is
    kept so the identity holds for any W with I - W invertible.)  With mu = Theta b:
        theta[i, :, k] = Theta[k, i] / den_i
        c[i, :, k]     = mu_k - Theta[k, i] (b_i + r_i / den_i),
                         r_i = sum_{l != i} (w_i' Theta)_l b_l.
    """
    S, d, _ = W_s.shape
    Theta = np.linalg.inv(np.eye(d) - W_s)                     # (S, d, d)
    mu = np.einsum("skl,sl->sk", Theta, b_s)                   # observational mean per draw
    wTheta = np.einsum("sil,slk->sik", W_s, Theta)             # row i: w_i' Theta
    diag = np.einsum("sii->si", wTheta)                        # w_i' Theta e_i
    den = 1.0 + diag                                           # (S, d)
    r = np.einsum("sil,sl->si", wTheta, b_s) - diag * b_s      # (S, d)
    shift = b_s + r / den                                      # (S, d)
    ThetaT = np.ascontiguousarray(Theta.transpose(2, 0, 1))    # [i, s, k] = Theta[s, k, i]
    theta = ThetaT / den.T[:, :, None]
    c = mu[None, :, :] - ThetaT * shift.T[:, :, None]
    return c, theta


def interventional_means(c, theta, query_i, query_v):
    """mu[q, s, k] = E_s[X_k | do(X_i = v)] for every query, shape (Q, S, d): all queries in
    one broadcast (the SPEC's einsum over Q)."""
    qi = np.asarray(query_i)
    qv = np.asarray(query_v, dtype=np.float64)
    return c[qi] + qv[:, None, None] * theta[qi]


# --------------------------------------------------------------------------- the agent

class MechAgent(Agent):
    """All mech-* agents and oracle-structure (SPEC "Agents" table), selected by flags:

    structure   "learned": slope tests / order / parent selection every episode (step 7)
                "true":    the true visible DAG from OracleAccess (oracle-structure)
    reset_rule  "per-mechanism": truncate only the mechanism whose detector fires
                "all":           any fire truncates every buffer (mech-reset-all)
                "oracle":        no statistic; truncate the world's shifted mechanism
                "none":          no detector; exponential forgetting gamma (mech-no-detect)
    selection   "active": free budget on argmax_i score(i); "random": uniform targets
    floor       False only for mech-full-nofloor (the protocol applies the floor itself)
    prior       "default" or "overconfident" (mech-overconfident)
    diag_pair   mech-full-diag: mechanism B's buffer takes only rows with target A

    Per-mechanism state lives in stacked arrays indexed by mechanism j:
      S (d, d+1, d+1) moment matrices, n_eff (d,), n_rows (d,) rows since last reset,
      g (d,) Page-Hinkley statistics, parents (list of sorted lists), mask (d, d+1).
    """

    def __init__(self, name, agent_id, d, seed, constants, access, *,
                 structure="learned", reset_rule="per-mechanism", selection="active",
                 floor=True, prior="default", diag_pair=False):
        super().__init__(name, agent_id, d, seed, constants, access)
        if structure not in ("learned", "true"):
            raise ValueError(f"unknown structure {structure!r}")
        if reset_rule not in ("per-mechanism", "all", "oracle", "none"):
            raise ValueError(f"unknown reset_rule {reset_rule!r}")
        if selection not in ("active", "random"):
            raise ValueError(f"unknown selection {selection!r}")
        self.structure = structure
        self.reset_rule = reset_rule
        self.selection = selection
        self.prior = prior
        self.uses_floor = bool(floor)
        self.learns_structure = structure == "learned"
        self.has_detector = reset_rule in ("per-mechanism", "all")
        self.gamma_forget = float(DEFAULT_GAMMA if self.gamma is None else self.gamma)
        self.v0, self.a0, self.b0 = PRIORS[prior]

        # per-mechanism state
        self.S = np.zeros((d, d + 1, d + 1))
        self.n_eff = np.zeros(d)
        self.n_rows = np.zeros(d, dtype=np.int64)
        self.g = np.zeros(d)
        self.parents: list[list[int]] = [[] for _ in range(d)]
        self.mask = np.zeros((d, d + 1), dtype=bool)
        self.mask[:, 0] = True                              # the intercept is always fitted
        if structure == "true":
            adj = np.asarray(access.true_adjacency(), dtype=bool)
            for j in range(d):
                self._set_parents(j, np.flatnonzero(adj[j]))
        self._post: _JointPosterior | None = None            # cache; None = recompute

        # mech-full-diag (SPEC P6 diagnostic): (A, B) from oracle access; None outside the
        # hidden variant, in which case the agent behaves exactly like mech-full.
        self.diag_pair = access.hidden_pair() if diag_pair else None

        # The agent-private tie-break permutation of order extraction (SPEC step 2), drawn
        # once from the agent's stream before episode 1 and kept for the run.
        self.pi = self.rng.permutation(d)

        # per-run bookkeeping
        self._t = 0
        self._X_obs = None                                        # this episode's batch
        self._obs_recent = collections.deque(maxlen=VAR_WINDOW)   # batches behind v_hat
        self._pool = collections.deque()                          # (t, rows, targets, values)
        self._g_before = np.full(d, np.nan)                       # detector_stats of this episode
        self._glr = np.full(d, np.nan)
        self._sample_scores: list[float] = []
        self._var_obj_before = math.nan
        self._var_obj_after = math.nan

    # ------------------------------------------------------------------ buffer bookkeeping
    def _set_parents(self, j, parents):
        self.parents[j] = sorted(int(p) for p in parents)
        self.mask[j, 1:] = False
        self.mask[j, [p + 1 for p in self.parents[j]]] = True
        self._post = None

    def _reset(self, j):
        """SPEC step 4: truncate mechanism j's buffer to empty and zero its detector state."""
        self.S[j] = 0.0
        self.n_eff[j] = 0.0
        self.n_rows[j] = 0
        self.g[j] = 0.0
        self._post = None

    def _accepting(self, target):
        """Mechanisms whose buffer accepts a row with intervention `target` (None = observational):
        all but the target itself; mech-full-diag's B only takes do(A) rows."""
        keep = np.ones(self.d, dtype=bool)
        if target is not None:
            keep[target] = False
        if self.diag_pair is not None:
            A, B = self.diag_pair
            if target != A:
                keep[B] = False
        return keep

    # ------------------------------------------------------------------ posteriors
    def _posteriors(self) -> _JointPosterior:
        """All d NIG posteriors from the buffers and parent sets, in one batched inverse.
        The precision of mechanism j is V0^-1 + S_xx inside its design block and the identity
        outside, so its inverse is block-diagonal: (V0^-1 + S_xx)^-1 in the block, 1 elsewhere."""
        if self._post is None:
            d = self.d
            j_idx = np.arange(d)
            mask = self.mask
            block = mask[:, :, None] & mask[:, None, :]                  # (d, d+1, d+1)
            diag_fill = np.where(mask, 1.0 / self.v0, 1.0)               # V0^-1 in, 1 out
            precision = np.where(block, self.S, 0.0) + np.eye(d + 1) * diag_fill[:, :, None]
            V_chol = np.linalg.inv(precision)
            V = np.where(block, V_chol, 0.0)
            Sxy = np.where(mask, self.S[j_idx, :, j_idx + 1], 0.0)      # column y = X_j
            Syy = self.S[j_idx, j_idx + 1, j_idx + 1]
            m = np.einsum("jpq,jq->jp", V, Sxy)
            a = self.a0 + 0.5 * self.n_eff
            b = np.maximum(self.b0 + 0.5 * (Syy - np.einsum("jp,jp->j", Sxy, m)), 1e-12)
            self._post = _JointPosterior(m, V, V_chol, a, b, mask)
        return self._post

    # ------------------------------------------------------------------ step 3
    def observe(self, X_obs, t):
        self._t = int(t)
        self._X_obs = np.asarray(X_obs, dtype=np.float64)
        self._obs_recent.append(self._X_obs)
        self._pool.append((self._t, [], [], []))
        while self._pool and self._pool[0][0] <= self._t - W_STRUCT:
            self._pool.popleft()
        self._sample_scores = []
        if self.reset_rule == "none":
            # omega_r = gamma^(t - t_r): every row already in the buffers ages by one episode.
            self.S *= self.gamma_forget
            self.n_eff *= self.gamma_forget
            self._post = None

    # ------------------------------------------------------------------ step 4
    def _glr_statistic(self, j):
        """GLR_t = l_new - l_old for mechanism j on this episode's batch, under the posterior
        from the buffer before the batch is appended (SPEC "Shift detection").  l_old: Student-t
        posterior predictive; l_new: Gaussian at the batch-only OLS fit, sigma2 = RSS / N."""
        post = self._posteriors()
        X = self._X_obs
        n = X.shape[0]
        Xa = np.empty((n, self.d + 1))
        Xa[:, 0] = 1.0
        Xa[:, 1:] = X
        y = X[:, j]
        loc = Xa @ post.m[j]
        quad = np.einsum("np,pq,nq->n", Xa, post.V[j], Xa)
        l_old = float(np.sum(log_t_pdf(y, loc, (post.b[j] / post.a[j]) * (1.0 + quad),
                                       2.0 * post.a[j])))
        Xd = Xa[:, self.mask[j]]                              # design [1, X_pa(j)]
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        rss = float(np.sum((y - Xd @ beta) ** 2))
        sigma2 = max(rss / n, 1e-12)
        l_new = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
        return l_new - l_old

    def detect_and_reset(self):
        d = self.d
        self._g_before[:] = np.nan
        self._glr[:] = np.nan
        if self.reset_rule == "oracle":
            j = int(self.access.shifted_mechanism(self._t))
            if 0 <= j < d:                       # the hidden H shift (j == d) cannot be reset
                self._reset(j)
                return [j]
            return []
        if self.reset_rule == "none":
            return []

        # Decision (SPEC "Shift detection" vs SPEC test 6 "no re-fire in the episode after a
        # reset"): the SPEC defines no dead time after a reset, only the N_min = 100 guard,
        # and a buffer truncated at step 4 already holds ~240 rows by the next step 4.  So a
        # mechanism IS tested in the episode after its reset, exactly as specified.  Whether
        # it re-fires is then a statistical matter: with a ~240-row posterior the term
        # l_true - l_old in GLR is ~ (p+1)/2 * N_obs / n_buffer on top of the batch fit's
        # optimism, which the fixed drift p + 2 does not absorb, so P(fire) at lambda* is
        # ~10 % in that episode (and in episodes 2-4 of every run) against ~1 % once the
        # buffer holds ~1400 rows.  We do not add an unspecified dead time to hide this; it
        # is a property of the SPEC's detector and shows up in Metric 8 as post-shift resets.
        fired = []
        for j in range(d):
            if self.n_rows[j] < N_MIN:           # guard: skip and hold g at 0
                self.g[j] = 0.0
                continue
            glr = self._glr_statistic(j)
            p = len(self.parents[j]) + 1
            self.g[j] = max(0.0, self.g[j] + glr - (p + 2))    # Page-Hinkley, drift p + 2
            self._glr[j] = glr
            self._g_before[j] = self.g[j]
            if self.g[j] > self.lam:
                fired.append(j)

        if not fired:
            return []
        if self.reset_rule == "all":
            for j in range(d):
                self._reset(j)
            return list(range(d))
        for j in fired:                          # per-mechanism
            self._reset(j)
        return fired

    def detector_stats(self):
        if not self.has_detector:
            return None
        return {"g": self._g_before.copy(), "glr": self._glr.copy()}

    # ------------------------------------------------------------------ step 5
    def refit(self):
        X = self._X_obs
        Xa = np.empty((X.shape[0], self.d + 1))
        Xa[:, 0] = 1.0
        Xa[:, 1:] = X
        keep = self._accepting(None)
        self.S[keep] += Xa.T @ Xa
        self.n_eff[keep] += X.shape[0]
        self.n_rows[keep] += X.shape[0]
        self._post = None                        # posteriors follow from the buffers
        self._var_obj_before = float(np.sum(self._scores()))

    # ------------------------------------------------------------------ step 6
    def choose_free_interventions(self, n_free, t):
        if self.selection == "random" or not self.mask[:, 1:].any():
            # Random targets for the random policy, and ALSO for the active policy while the
            # learned graph is still empty (SPEC: "the all-zero case when the learned graph
            # is empty" is broken by the agent's stream).  With no edges the literal score
            # would rank variables by how well their intercepts are known, which is arbitrary
            # rather than informative (conformance audit finding).
            targets = self.rng.integers(0, self.d, size=n_free)
        else:
            score = self._scores()
            best = score.max()
            # Decision: score(i) is the SPEC formula taken literally, the variance over
            # posterior draws of the full answer mu_{s,k}(i, v = 2), which includes the
            # intercept uncertainty of every k != i.  The SPEC's remark that an empty learned
            # graph gives "the all-zero case" is therefore not what the formula evaluates to:
            # with no edges score(i) = sum_{k != i} Var(b_k) / v_hat_k, distinct across i, so
            # the stream tie-break below only ever acts on exact ties.  Taking the variance of
            # the effect alone (2 theta) instead would change the rule beyond the SPEC.
            ties = np.flatnonzero(score >= best - 1e-12 * max(1.0, abs(best)))
            i_star = int(ties[self.rng.integers(len(ties))]) if len(ties) > 1 else int(ties[0])
            targets = [i_star] * n_free          # whole free budget on argmax_i score(i)
        return [(int(i), self.cycle.next_value(int(i))) for i in targets]

    def score_before_reveal(self, i, v, row):
        """sum_{j != i} log t_nu_j(x_j; x_pa(j)' m_j, s_j^2) on the realized row."""
        xa = np.empty(self.d + 1)
        xa[0] = 1.0
        xa[1:] = row
        loc, scale2, nu = self._posteriors().predictive_row(xa)
        logp = log_t_pdf(xa[1:], loc, scale2, nu)
        total = float(np.sum(logp) - logp[i])
        self._sample_scores.append(total)
        return total

    def receive_intervention(self, i, v, row):
        row = np.asarray(row, dtype=np.float64)
        xa = np.empty(self.d + 1)
        xa[0] = 1.0
        xa[1:] = row
        keep = self._accepting(int(i))           # the clamped variable's own mechanism is off
        self.S[keep] += np.outer(xa, xa)
        self.n_eff[keep] += 1.0
        self.n_rows[keep] += 1
        self._post = None
        _, rows, targets, values = self._pool[-1]
        rows.append(row)
        targets.append(int(i))
        values.append(float(v))

    # ------------------------------------------------------------------ step 7
    def update_structure(self, t):
        if self.structure != "learned":
            return []
        rows, targets, values = [], [], []
        for _, r, tg, vals in self._pool:
            rows.extend(r)
            targets.extend(tg)
            values.extend(vals)
        P = slope_test_pvalues(rows, targets, values, self.d)
        A_hat = estimated_ancestors(P, ALPHA_BH)
        order = extract_order(A_hat, self.pi)
        A_acyclic = prune_to_order(A_hat, order)

        changed = []
        for j in range(self.d):
            candidates = np.flatnonzero(A_acyclic[:, j])
            new = ols_parent_test(self.S[j], j, candidates, self.n_eff[j], ALPHA_PARENT)
            if new is None or new == self.parents[j]:
                continue
            self._set_parents(j, new)            # the refit follows from the new sub-blocks
            self.g[j] = 0.0                      # SPEC: g_j := 0 on a parent-set change
            changed.append(j)
        return changed

    # ------------------------------------------------------------------ step 8
    def answer(self, query_i, query_v):
        c, theta = self._components()
        self._var_obj_after = float(np.sum(self._scores_from(c, theta)))
        qi = np.asarray(query_i)
        qv = np.asarray(query_v, dtype=np.float64)
        # Draw axis last so the sort below runs on contiguous memory.
        cT = np.ascontiguousarray(c.transpose(0, 2, 1))          # [i, k, s]
        thetaT = np.ascontiguousarray(theta.transpose(0, 2, 1))
        mu = cT[qi] + qv[:, None, None] * thetaT[qi]             # (Q, d, S)
        mu.sort(axis=-1)
        point = mu.mean(axis=-1)
        quant = quantiles_of_sorted(mu, QUANTILE_LEVELS)         # (Q, d, 6)
        return {"point": point, "quantiles": quant}

    def learned_adjacency(self):
        return self.mask[:, 1:].copy()           # adj[j, p] = True iff p in pa(j)

    def internal_objectives(self):
        self_score = float(np.mean(self._sample_scores)) if self._sample_scores else math.nan
        return {"self_score": self_score,
                "var_obj_before": self._var_obj_before,
                "var_obj_after": self._var_obj_after}

    # ------------------------------------------------------------------ internals
    def _components(self, n_draws=S_DRAWS):
        """Fresh joint draws under the posterior at this moment, pushed through the mutilated
        learned graph: (c, theta), each (d, S, d)."""
        W_s, b_s = self._posteriors().draw_joint(self.rng, n_draws)
        return do_components(W_s, b_s)

    def _v_hat(self):
        """Agent-side sample variance of each X_k over the last VAR_WINDOW observational
        batches (never Sigma_ref)."""
        X = np.concatenate(list(self._obs_recent), axis=0)
        return np.maximum(X.var(axis=0), 1e-12)

    def _scores_from(self, c, theta):
        """score(i) = sum_{k != i} Var_s[mu_{s,k}(i, v = 2)] / v_hat_k (SPEC "Active
        intervention rule") from the components of the current draws."""
        mu2 = c + 2.0 * theta                                # (d, S, d) at v = 2
        var = mu2.var(axis=1) / self._v_hat()[None, :]       # [i, k]
        np.fill_diagonal(var, 0.0)                           # k != i
        return var.sum(axis=1)

    def _scores(self):
        c, theta = self._components()
        return self._scores_from(c, theta)
