"""Structure-free baselines (SPEC.md "Agents" table rows 1-4 and "Baseline details").

Four agents live here:

    MarginalMeanAgent   marginal-mean    cumulative observational mean of X_j, ignores (i, v)
    ObsPairwiseAgent    obs-window       per-pair OLS of X_j on [1, X_i], observational rows,
                                         sliding window of the last W episodes ("see" for "do")
    ObsPairwiseAgent    obs-cumulative   same, all observational rows ever seen
    IntPairwiseAgent    int-pairwise     per-pair OLS of X_j on [1, v] over rows where X_i was
                                         clamped, sliding window W; consistent for the
                                         interventional mean without any graph

Common machinery (SPEC "Baseline details"):

  * Every fit is a two-parameter OLS with intercept; no ridge penalty.
  * Uncertainty is a nonparametric bootstrap: 200 resamples of the pool's rows.  ONE resample
    of row indices is shared across all pairs, drawn from the agent's own RNG stream.  Each
    resample refits every pair and the six quantiles of the resampled prediction are returned.
  * Guards: a fit needs n >= 4 rows and (int-pairwise) >= 2 distinct v; otherwise, and whenever
    the regressor variance is < 1e-8, the pair falls back to the marginal-mean rule (sample
    mean of X_j over the agent's observational pool).  The guards are applied to every fit,
    i.e. to the full pool AND to each bootstrap resample.

Implementation note.  A nonparametric resample of n rows is the same thing as giving row r
the integer weight "number of times r was drawn".  All fits below are therefore written as
weighted moment computations with a weight matrix of shape (1 + 200, n): row 0 holds all
ones (the full pool -> the point prediction), rows 1..200 hold the resample counts.  This
keeps every fit a matrix product and avoids materialising 200 copies of the pool.

Nothing here reads truth, Var_ref, the schedule or the permutation, and this module never
imports `world` (SPEC test 3).
"""

from __future__ import annotations

import math

import numpy as np

from .base import QUANTILE_LEVELS, Agent

N_BOOT = 200          # bootstrap resamples (SPEC "Baseline details")
MIN_ROWS = 4          # a fit needs n >= 4 rows
MIN_VAR = 1e-8        # regressor variance below this -> fall back to the marginal mean
DEFAULT_W_OBS = 3     # obs-window default window (INTERFACES section 3)
DEFAULT_W_INT = 5     # int-pairwise default window (INTERFACES section 3)


# --------------------------------------------------------------------------- shared helpers

def bootstrap_weights(rng: np.random.Generator, n: int, n_boot: int = N_BOOT) -> np.ndarray:
    """(1 + n_boot, n) float weights: row 0 is all ones (the full pool); row b >= 1 counts how
    often each row index was drawn in the b-th nonparametric resample of n row indices.

    One call = one shared resample for all pairs (SPEC: "one resample of row indices shared
    across all pairs, drawn from the agent's stream")."""
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.stack([np.bincount(row, minlength=n) for row in idx])
    return np.vstack([np.ones((1, n)), counts]).astype(np.float64)


def weighted_moments(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sample mean (R, d) and covariance (R, d, d) of the rows of X (n, d) under each
    of the R weight vectors in w (R, n).  Covariance uses the 1/n convention; only ratios of
    covariances are used downstream, so the convention cancels."""
    n, d = X.shape
    n_r = w.sum(axis=1)                                   # total weight per resample (= n)
    mean = (w @ X) / n_r[:, None]
    # second moments: (w @ Z) with Z[r] = vec(x_r x_r'), one BLAS call for all resamples
    Z = (X[:, :, None] * X[:, None, :]).reshape(n, d * d)
    second = (w @ Z).reshape(-1, d, d) / n_r[:, None, None]
    cov = second - mean[:, :, None] * mean[:, None, :]
    return mean, cov


def quantiles_over_resamples(pred: np.ndarray) -> np.ndarray:
    """pred: (1 + n_boot, Q, d) predictions (row 0 = full pool).  Returns the six quantiles of
    rows 1.. along the resample axis, shape (Q, d, 6), non-decreasing along the last axis."""
    q = np.quantile(pred[1:], QUANTILE_LEVELS, axis=0)   # (6, Q, d)
    return np.moveaxis(q, 0, -1)


# --------------------------------------------------------------------------- marginal-mean

class MarginalMeanAgent(Agent):
    """`marginal-mean`: predicts the cumulative observational sample mean of X_j for every
    query, ignoring (i, v).  Uncertainty: bootstrap of the mean.  Never abstains, never
    intervenes (SPEC "Agents" table, row 1)."""

    intervenes = False
    never_abstains = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batches: list[np.ndarray] = []     # every observational batch ever seen
        self._pending: np.ndarray | None = None  # this episode's batch, appended in refit()
        self._mean: np.ndarray | None = None     # (1 + N_BOOT, d): row 0 = point estimate

    def observe(self, X_obs, t):
        self._pending = np.array(X_obs, dtype=np.float64, copy=True)

    def refit(self):
        # step 5: append the batch, then the "posterior" is a function of the pool alone
        self._batches.append(self._pending)
        self._pending = None
        pool = np.concatenate(self._batches, axis=0)
        w = bootstrap_weights(self.rng, pool.shape[0])
        self._mean = (w @ pool) / w.sum(axis=1)[:, None]  # (1 + N_BOOT, d)

    def answer(self, query_i, query_v):
        Q = len(query_i)
        point = np.broadcast_to(self._mean[0], (Q, self.d)).copy()
        q = np.quantile(self._mean[1:], QUANTILE_LEVELS, axis=0).T   # (d, 6)
        quantiles = np.broadcast_to(q, (Q, self.d, 6)).copy()
        return {"point": point, "quantiles": quantiles}


# --------------------------------------------------------------------------- obs-window / obs-cumulative

class ObsPairwiseAgent(Agent):
    """`obs-window` (windowed=True) and `obs-cumulative` (windowed=False): per-pair OLS of X_j
    on [1, X_i] over observational rows only, predicting beta0 + beta1 v.  This is the
    "see for do" baseline: its estimand is the Gaussian conditional
    mu_j + Sigma_ji / Sigma_ii (v - mu_i), not the interventional mean (SPEC P1).

    Pool: the last W observational batches (W from constants, default 3) or all of them.
    Never intervenes (ignores the budget)."""

    intervenes = False

    def __init__(self, *args, windowed: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.windowed = bool(windowed)
        self.W = int(self.window) if (self.windowed and self.window is not None) else DEFAULT_W_OBS
        self._batches: list[tuple[int, np.ndarray]] = []   # (t, batch) in the pool
        self._pending: tuple[int, np.ndarray] | None = None
        self._t = 0
        # fitted quantities, all with a leading resample axis R = 1 + N_BOOT
        self._mean: np.ndarray | None = None      # (R, d)
        self._beta1: np.ndarray | None = None     # (R, d, d): [r, i, j] = slope of X_j on X_i

    def observe(self, X_obs, t):
        self._t = int(t)
        self._pending = (self._t, np.array(X_obs, dtype=np.float64, copy=True))

    def pool(self) -> np.ndarray:
        """The observational rows currently in the pool, (n, d)."""
        return np.concatenate([X for _, X in self._batches], axis=0)

    def refit(self):
        # step 5: append the batch, drop batches that fell out of the window, refit all pairs
        self._batches.append(self._pending)
        self._pending = None
        if self.windowed:
            # "last W episodes" includes the current one: keep t in (self._t - W, self._t]
            self._batches = [(t, X) for t, X in self._batches if t > self._t - self.W]
        X = self.pool()
        n = X.shape[0]
        w = bootstrap_weights(self.rng, n)
        mean, cov = weighted_moments(X, w)
        # OLS of X_j on [1, X_i]:  beta1 = cov(X_i, X_j) / var(X_i),  beta0 = mean_j - beta1 mean_i.
        # Guards: n >= 4 rows and var(X_i) >= 1e-8; otherwise beta1 = 0, i.e. the marginal mean.
        var = np.einsum("rii->ri", cov)                                   # (R, d)
        ok = (var >= MIN_VAR) & (n >= MIN_ROWS)
        safe_var = np.where(ok, var, 1.0)
        self._beta1 = np.where(ok[:, :, None], cov / safe_var[:, :, None], 0.0)
        self._mean = mean

    def answer(self, query_i, query_v):
        qi = np.asarray(query_i, dtype=np.int64)
        qv = np.asarray(query_v, dtype=np.float64)
        # pred[r, q, j] = mean_j + beta1[i, j] (v - mean_i)  ==  beta0 + beta1 v
        slope = self._beta1[:, qi, :]                                     # (R, Q, d)
        shift = (qv[None, :] - self._mean[:, qi])[:, :, None]             # (R, Q, 1)
        pred = self._mean[:, None, :] + slope * shift                     # (R, Q, d)
        return {"point": pred[0], "quantiles": quantiles_over_resamples(pred)}


# --------------------------------------------------------------------------- int-pairwise

class IntPairwiseAgent(Agent):
    """`int-pairwise`: per-pair OLS of X_j on [1, v] over the pool rows where X_i was the
    clamped target.  Consistent for E[X_j | do(X_i = v)] with no graph at all; the strong
    non-factored competitor (SPEC "Agents" table, row 4).

    Pool: interventional rows of the last W episodes (W from constants, default 5), the current
    episode included.  Intervention policy: the shared round-robin floor (run by the protocol)
    plus uniform-random targets for the free budget with the per-variable value cycle - the
    same policy as `mech-random`.

    Observational batches are received for ONE purpose only: the marginal-mean fallback when a
    pair's fit is guarded out.  Decision (SPEC does not say): the observational pool for that
    fallback is windowed with the same W, so the fallback adapts to shifts on the same time
    scale as the agent's own estimator."""

    intervenes = True
    uses_floor = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = int(self.window) if self.window is not None else DEFAULT_W_INT
        self._obs: list[tuple[int, np.ndarray]] = []          # (t, batch) observational pool
        self._pending: tuple[int, np.ndarray] | None = None
        self._rows: list[tuple[int, int, float, np.ndarray]] = []   # (t, i, v, row)
        self._t = 0

    # -- data intake -----------------------------------------------------------------------
    def observe(self, X_obs, t):
        self._t = int(t)
        self._pending = (self._t, np.array(X_obs, dtype=np.float64, copy=True))

    def refit(self):
        # step 5: append the observational batch (fallback pool only) and trim both pools.
        # The OLS fits themselves are computed in answer(): the interventional pool grows
        # during step 6 and nothing in step 6 reads the fit (the free policy is random and
        # score_before_reveal is None for baselines), so answer-time is the first moment the
        # "posterior" is needed and it is then a deterministic function of the pools.
        self._obs.append(self._pending)
        self._pending = None
        lo = self._t - self.W                          # keep episodes t with t > lo
        self._obs = [(t, X) for t, X in self._obs if t > lo]
        self._rows = [r for r in self._rows if r[0] > lo]

    def choose_free_interventions(self, n_free, t):
        # uniform-random targets from the agent's stream, values from the per-variable cycle
        targets = self.rng.integers(0, self.d, size=int(n_free))
        return [(int(i), self.cycle.next_value(int(i))) for i in targets]

    def receive_intervention(self, i, v, row):
        self._rows.append((self._t, int(i), float(v), np.array(row, dtype=np.float64, copy=True)))

    # -- pools -----------------------------------------------------------------------------
    def obs_pool(self) -> np.ndarray:
        return np.concatenate([X for _, X in self._obs], axis=0)

    def int_pool(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(targets (n,) int, values (n,) float, rows (n, d)) of the windowed interventional pool."""
        if not self._rows:
            return (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, self.d)))
        tgt = np.array([r[1] for r in self._rows], dtype=np.int64)
        val = np.array([r[2] for r in self._rows], dtype=np.float64)
        X = np.stack([r[3] for r in self._rows])
        return tgt, val, X

    # -- the fit ---------------------------------------------------------------------------
    def _clamped_ols(self, tgt, val, X, w):
        """OLS of X_j on [1, v] within the rows clamped on i, for every weight row of w (R, n).

        Returns beta0, beta1 of shape (R, d, d) indexed [r, i, j], and ok (R, d): whether the
        fit for target i passed the guards (n_i >= 4, >= 2 distinct v, var(v) >= 1e-8)."""
        d = self.d
        n = len(tgt)
        R = w.shape[0]
        onehot = (tgt[:, None] == np.arange(d)[None, :]).astype(np.float64)   # (n, d)
        n_i = w @ onehot                                                       # (R, d)
        S_v = w @ (onehot * val[:, None])                                      # sum of v
        S_vv = w @ (onehot * val[:, None] ** 2)                                # sum of v^2
        S_x = (w @ (onehot[:, :, None] * X[:, None, :]).reshape(n, d * d)).reshape(R, d, d)
        S_vx = (w @ (onehot[:, :, None] * (val[:, None] * X)[:, None, :]).reshape(n, d * d)
                ).reshape(R, d, d)
        # number of distinct clamp values present (with positive weight) per target
        levels = np.unique(val)
        G = (onehot[:, :, None] * (val[:, None] == levels[None, :])[:, None, :]
             ).reshape(n, d * len(levels))
        present = ((w > 0).astype(np.float64) @ G).reshape(R, d, len(levels)) > 0
        n_distinct = present.sum(axis=-1)                                      # (R, d)

        safe_n = np.where(n_i > 0, n_i, 1.0)
        mean_v = S_v / safe_n
        var_v = S_vv / safe_n - mean_v ** 2
        mean_x = S_x / safe_n[:, :, None]
        cov_vx = S_vx / safe_n[:, :, None] - mean_v[:, :, None] * mean_x
        ok = (n_i >= MIN_ROWS) & (n_distinct >= 2) & (var_v >= MIN_VAR)
        safe_var = np.where(ok, var_v, 1.0)
        beta1 = np.where(ok[:, :, None], cov_vx / safe_var[:, :, None], 0.0)
        beta0 = mean_x - beta1 * mean_v[:, :, None]
        return beta0, beta1, ok

    def answer(self, query_i, query_v):
        qi = np.asarray(query_i, dtype=np.int64)
        qv = np.asarray(query_v, dtype=np.float64)
        Q = len(qi)

        # fallback pool: observational rows, with their own shared row resample
        X_obs = self.obs_pool()
        w_obs = bootstrap_weights(self.rng, X_obs.shape[0])
        mean_obs = (w_obs @ X_obs) / w_obs.sum(axis=1)[:, None]              # (R, d)

        # interventional pool: one shared row resample across all pairs
        tgt, val, X = self.int_pool()
        n = len(tgt)
        if n > 0:
            w = bootstrap_weights(self.rng, n)
            beta0, beta1, ok = self._clamped_ols(tgt, val, X, w)
            pred_ols = beta0[:, qi, :] + beta1[:, qi, :] * qv[None, :, None]  # (R, Q, d)
            use_ols = ok[:, qi][:, :, None]                                   # (R, Q, 1)
        else:
            pred_ols = np.zeros((mean_obs.shape[0], Q, self.d))
            use_ols = np.zeros((mean_obs.shape[0], Q, 1), dtype=bool)

        pred = np.where(use_ols, pred_ols, mean_obs[:, None, :])
        return {"point": pred[0], "quantiles": quantiles_over_resamples(pred)}
