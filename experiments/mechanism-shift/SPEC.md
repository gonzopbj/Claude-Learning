# Mechanism-Shift Testbed — experimental specification (v1, pre-critique)

## Purpose

A previous discussion produced six requirements for a system that learns continually
and whose mistakes are detectable, bounded, recoverable, and not repeated:

1. An error signal from outside itself (the world as verifier; interventions as queries).
2. Calibrated uncertainty, so it can abstain instead of guessing.
3. A representation factored into mechanisms, not correlations (sparse mechanism shift).
4. Counterfactual / causal credit assignment: attribute an error to the mechanism that changed.
5. Memory that does not overwrite: learning the new must not destroy the old.
6. An improvement objective that cannot be gamed.

This testbed is the smallest instance that contains all six difficulties at once, with a
known ground truth so every claim is falsifiable. It is **not** an AGI system. It is the
experiment a careful researcher runs before claiming anything. The deliverable is numbers
with confidence intervals, plus the cases where the proposed system fails.

## World

- Linear-Gaussian structural causal model over `d = 6` observed variables.
- Random DAG per seed (Erdős–Rényi, expected in-degree ≈ 1.5, fixed topological order for
  generation only; the agent is never told the order or the graph).
- Weights drawn uniformly from ±[0.5, 1.5]; noise standard deviations from [0.5, 1.5].
- Time is divided into `T = 40` episodes. At the start of episode t ≥ 5, with probability
  `p_shift = 0.35`, exactly **one** mechanism (the structural equation of one variable, chosen
  uniformly) is resampled: new parent weights (same parents) and new noise scale. This is the
  sparse-mechanism-shift assumption made literal. The graph skeleton is fixed within a seed.
- A **hidden-confounder variant** (`hidden=True`): the same generator with `d = 7`, but one
  variable that has at least two children is removed from the agent's view. This is the
  pre-registered failure case.

## What the agent sees each episode

- `N_obs = 200` observational samples from the current SCM.
- An intervention budget `B = 50` samples. The agent chooses, per sample, a target variable
  `i` and a value `v ∈ {−2, −1, 1, 2}`; the world returns a full sample from the mutilated
  SCM with `X_i := v`. Random-target agents draw `i` uniformly.
- Nothing else. No graph, no order, no shift signal.

## What the agent is asked each episode (evaluation, held out)

`Q = 200` interventional queries `do(X_i = v)` with `i` uniform and `v ~ U(−2, 2)`.
For each query the agent returns, for every variable `j ≠ i`:
- a point prediction of `E[X_j | do(X_i = v)]`,
- a central 90 % predictive interval for that mean,
- a boolean `abstain` decision.

Ground truth is computed exactly from the current SCM.

## Agents

All agents receive **identical data streams** (same seeds, same observational samples; the
interventional samples differ only when the agent chooses targets differently). Only the
oracle sees the SCM.

| Name | Structure | Update rule | Uncertainty | Intervention choice | Tests requirement |
|---|---|---|---|---|---|
| `obs-window` | none: one ridge regression per variable on all others, observational data only | sliding window of last `W = 3` episodes | bootstrap over the window | none (ignores budget) | baseline for 1, 3, 5 |
| `obs-cumulative` | none | all data ever seen | bootstrap | none | forgetting vs. staleness trade-off |
| `int-monolithic` | none: ridge on all others, but trained on observational **and** interventional data | sliding window `W = 3` | bootstrap | random | isolates value of factorization (3) |
| `mech-full` | learned DAG | per-mechanism Bayesian linear regression; **only** mechanisms whose residual test fires are reset | posterior predictive, propagated through the mutilated graph by Monte Carlo | active: target the intervention that maximally reduces posterior variance of the currently least-certain mechanism | the proposed system |
| `mech-no-detect` | learned DAG | all mechanisms updated with exponential forgetting (no per-mechanism reset) | same | active | ablates 4 |
| `mech-random` | learned DAG | as `mech-full` | same | random target | ablates active selection |
| `oracle-structure` | true DAG given | as `mech-full` | same | active | separates structure error from parameter error |
| `oracle` | true SCM | none | exact | none | ceiling |

### `mech-full` in detail

**Structure learning** (re-estimated every episode from a window of interventional data plus
current observational data):
1. Ancestor detection: for each ordered pair `(i, j)`, two-sample test of `X_j` under
   `do(X_i)` samples versus observational samples. Reject at level α = 0.01 with a
   Benjamini–Hochberg correction across pairs ⇒ `i` is an ancestor of `j`.
2. Topological order from the ancestor matrix (Kahn's algorithm on the transitive
   reduction; ties broken by index).
3. Parents of `j`: among ancestors of `j`, keep those whose coefficient in a ridge regression
   of `X_j` on all its ancestors is significant at α = 0.05 (regression fit on samples where
   `X_j` itself was **not** intervened on, which is the standard interventional-data trick).

**Mechanism models**: for each `j`, a conjugate Bayesian linear regression
`X_j ~ N(w_j · X_pa(j) + b_j, σ_j²)` with a Normal–Inverse-Gamma prior. Posterior updated
with all non-`j`-intervened samples since the mechanism's last reset.

**Shift detection and credit assignment** (requirement 4): at the start of each episode, for
each `j`, compute the posterior-predictive log-likelihood of the new batch under mechanism `j`
and compare it with a Page–Hinkley / CUSUM statistic on the standardized residuals. If the
statistic exceeds threshold `λ` (set once on a validation seed set and never touched again),
**reset only mechanism `j`** to its prior. All other posteriors are untouched (requirement 5).

**Answering `do(X_i = v)`**: mutilate the learned graph (remove edges into `i`), fix `X_i = v`,
draw `S = 200` joint samples of the mechanism weights from their posteriors, propagate means in
topological order. Point estimate = mean over draws; interval = 5th–95th percentile;
`abstain = True` when the interval width exceeds `τ` (also fixed once on validation seeds).

**Active intervention** (requirement 1): rank variables by the total posterior variance of
the mechanisms downstream of them; spend the budget on the top-ranked variable, alternating
values. If a shift was detected in mechanism `j` this episode, spend half the budget on
parents of `j` (this is the "run the experiment that resolves the flagged confusion" rule).

## Metrics (per episode; mean and 95 % CI over `S_seeds = 20` seeds)

1. **Interventional MSE**: mean over queries and downstream variables of
   `(prediction − truth)²`. Reported for all queries and for non-abstained queries.
2. **Recovery time**: after each shift, number of episodes until the agent's MSE returns to
   within 1.5× its mean over the three episodes preceding the shift. Report median and IQR.
3. **Forgetting**: after a shift in mechanism `j`, MSE restricted to queries whose true answer
   does not depend on mechanism `j` (no path through `j`). A system that satisfies
   requirement 5 shows no bump here.
4. **Calibration**: empirical coverage of the nominal 90 % intervals, per episode, split into
   "shift episodes" and "stable episodes". Also coverage at 50 % and 99 % to detect
   systematic over- or under-dispersion.
5. **Selective risk**: risk–coverage curve obtained by sweeping the abstention threshold;
   area under it (AURC). And **mistake-detection recall**: among predictions with error above
   the 90th percentile of oracle-structure error, the fraction the agent abstained on.
6. **Structure recovery**: structural Hamming distance between learned and true DAG per
   episode, for `mech-*` agents.
7. **Value of interventions**: `mech-full` versus `mech-random` versus `int-monolithic` as a
   function of budget `B ∈ {10, 25, 50, 100}` at fixed `N_obs`.

## Pre-registered predictions (written before any code is run)

- **P1.** `obs-*` agents have interventional MSE bounded away from zero for the lifetime of
  the run wherever a query's target and outcome share a common ancestor, because they answer
  `do` with `see`. Their error does not shrink with more data.
- **P2.** After a shift, `mech-full` recovers in ≤ 2 episodes; `obs-window` in about `W`
  episodes; `obs-cumulative` never fully recovers within the run.
- **P3.** `mech-full` coverage is within ±5 points of nominal in stable episodes and drops in
  the single episode of a shift before detection fires. `int-monolithic` coverage is below
  nominal throughout because bootstrap-over-window ignores structural error.
- **P4.** `mech-no-detect` shows a forgetting bump on queries unrelated to the shifted
  mechanism; `mech-full` does not.
- **P5.** `mech-full` beats `mech-random` at `B = 10` and `B = 25`; the gap closes by
  `B = 100`.
- **P6 (the failure case).** With a hidden confounder, `mech-full` learns a wrong edge between
  the confounded children, its intervention predictions on that pair are biased, **and its
  intervals do not widen to cover the bias**. Coverage on that pair falls well below nominal
  and abstention does not rescue it. This is the honest limit: requirement 2 cannot be met
  by uncertainty over the wrong model class.

## Compute budget

Everything in numpy/scipy on CPU. Target: full sweep (8 agents × 20 seeds × 40 episodes,
plus the budget sweep and the hidden variant) in under 15 minutes on 4 cores.

## Deliverables

```
experiments/mechanism-shift/
  SPEC.md            this file (revised after critique)
  world.py           SCM generator, shifts, interventions, exact interventional means
  agents/            one file per agent family
  metrics.py
  run.py             CLI: --agents --seeds --episodes --budget --hidden --out
  tests/             property tests: generator correctness, no ground-truth leakage into agents,
                     metric correctness on hand-computed cases
  results/           JSON per run, PNG figures
  REPORT.md          tables + figures + which predictions held
```

## Non-goals

No neural networks, no language, no images. The question is whether the six requirements,
implemented in the simplest possible form, produce the behaviors claimed for them. If they do
not at `d = 6`, they will not at scale.
