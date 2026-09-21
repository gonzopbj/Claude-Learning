# Mechanism-Shift Testbed — experimental specification (v2, post-critique)

## Purpose

A previous discussion produced six requirements for a system that learns continually
and whose mistakes are detectable, bounded, recoverable, and not repeated:

1. An error signal from outside itself (the world as verifier; interventions as queries).
2. Calibrated uncertainty, so it can abstain instead of guessing.
3. A representation factored into mechanisms, not correlations (sparse mechanism shift).
4. Counterfactual / causal credit assignment: attribute an error to the mechanism that changed.
5. Memory that does not overwrite: learning the new must not destroy the old.
6. An improvement objective that cannot be gamed.

This testbed is the smallest instance in which each of the six can come back **false**, with
a known ground truth so every claim is checkable. It is **not** an AGI system. It is the
experiment a careful researcher runs before claiming anything. The deliverable is numbers
with confidence intervals, plus the cases where the proposed system fails.

Scope of what is actually tested, requirement by requirement:

- 1 is tested two ways: interventions as the data source for structure, and a proper scoring
  rule on the agent's *own requested* interventional samples, scored before they are revealed
  (metric 9). Only the second is a check rather than a heuristic.
- 2 is tested by coverage at three nominal levels *with* interval width, stratified by whether
  the learned structure is right, and by selective risk at matched abstention rates.
- 3 is tested by the budget sweep against a structure-free interventional learner
  (`int-pairwise`), which is consistent in this world without any graph. The claim for
  factorization is therefore sample efficiency and localized updating, not correctness.
- 4 is measured directly (metric 8: reset-event confusion, detection delay) and is made
  non-trivial by two world knobs (heteroscedastic noise, small and noise-only shifts) under
  which the per-mechanism residual no longer isolates the shifted mechanism for free.
- 5 is ablated by `mech-reset-all`, the agent that genuinely overwrites.
- 6 is tested in exactly one sense: two internal improvement signals are logged side by side
  (posterior-variance reduction and self log-score), and a deliberately overconfident agent
  is pre-registered to win on the first and lose on the second and on every external metric.
  Nothing more is claimed for requirement 6.

## World

### Variables and graph

- Linear-Gaussian structural causal model over `d = 6` observed variables.
- Random DAG per seed: Erdős–Rényi in a generation order with edge probability
  `p_edge = 3 / (d − 1)` (0.6 for `d = 6`), so the expected in-degree is 1.5. Resample if the
  DAG has no edges.
- **Relabelling.** After generation, a uniformly random permutation of variable labels
  (drawn from the graph RNG stream) is applied to the adjacency, weights, intercepts, noise
  scales and every sample before anything reaches any agent. Only the oracle's view records
  the permutation. The generation order is therefore *not* recoverable from indices.
- Weight of edge `p → j`: `W[j, p] = s · u`, `u ~ U[0.5, 1.5]`, `s = ±1` with probability
  1/2 each (children in rows; `W` is strictly lower-triangular in the generation order and
  is stored permuted).
- Intercepts `b_j ~ U[−1, 1]`. Noise scales `σ_j ~ U[0.5, 1.5]`.
- Structural equation: `X_j = W[j, :] · X + b_j + σ_j · (1 + γ_h · |W[j, :] · X|) · ε_j`,
  `ε_j ~ N(0, 1)` independent. `γ_h ∈ {0, 1}` is the **heteroscedasticity knob**; the main
  run uses `γ_h = 0`. Because the noise term has conditional mean zero for either value,
  `E[X | do(X_i = v)]` is the same closed form for both (below), and ground truth stays exact.
- Variances of deep variables can reach ≈ 50 (a chain of five weights of 1.5). No
  standardization is applied to the data; all error metrics are normalized (Metrics).

### Exact interventional means

`E[X | do(X_i = v)] = (I − W_i)⁻¹ (b_i + v e_i)`, where `W_i` is `W` with row `i` zeroed and
`b_i` is `b` with `b_i := 0`. The total-effect matrix `Θ = (I − W)⁻¹` satisfies
`Θ[k, i] = ((I − W_i)⁻¹)[k, i]` for all `k` (no directed path from `i` uses an edge into `i`),
so `Θ` is cached per episode and only the intercept term needs the mutilated solve. The
prediction is affine in `v`: `truth(i, k, v) = c_k(i) + Θ[k, i] · v` with
`c(i) = (I − W_i)⁻¹ b_i`. For `k` not a descendant of `i`, `truth(i, k, v) = E[X_k]` for all `v`.

Observational moments used for normalization and for population biases:
`μ = (I − W)⁻¹ b`, `Σ_ref = (I − W)⁻¹ diag(σ²) (I − W)⁻ᵀ`. `Σ_ref` is exact for `γ_h = 0` and
is the documented **reference scale** for `γ_h = 1` (the same constant across knob levels, so
numbers stay comparable).

Hidden variant: compute everything on the full 7-node SCM and return the visible coordinates.

### Shift process

- `T = 60` episodes. The shift schedule is **pre-drawn per seed** from the shift RNG stream
  before any agent runs: exactly `K = 8` shift episodes in `[7, 55]` with minimum gap 5
  between consecutive shift episodes (rejection-sample the set). This gives every shift at
  least 4 clean post-shift episodes and every shift after the first at least 2 clean pre-shift
  episodes. The marginal shift rate (≈ 0.16 per episode after episode 6) is documentation,
  not a parameter.
- At the start of a shift episode, exactly **one** mechanism is changed. **Shift-type knob**
  `shift_type ∈ {large, small, noise-only}`; main run uses `large`:
  - `large`: mechanism `j` uniform among variables with ≥ 1 parent; resample all of `j`'s
    incoming weights (same parents, same sign-magnitude rule), `b_j`, and `σ_j`.
  - `small`: `j` uniform among variables with ≥ 1 parent; `w_new = w + 0.25 · N(0, 1)` per
    incoming weight, redrawn if `|w_new| < 0.25` (keeps the skeleton identifiable); `b_j`
    and `σ_j` unchanged.
  - `noise-only`: `j` uniform among all `d` variables; only `σ_j` is resampled. No
    interventional mean changes; the ideal response is *no* reset.
- The graph skeleton is fixed within a seed. Ancestry is therefore invariant across the run.
- For every shift the world logs `(seed, t, j, shift_type, m, δ)` where
  `δ = ‖Θ_new − Θ_old‖_F` and the **effective magnitude**
  `m = E_q[(truth_after − truth_before)²] / Var_ref(X_k)` averaged over the query distribution
  (`i` uniform, `v ~ U(−2, 2)`, all `k ≠ i`), computed exactly from the two SCMs. Shift-
  conditional metrics are reported for shifts with `m ≥ 0.05`; the count excluded is reported.
  For `noise-only`, `m = 0` by construction and shift-conditional metrics use `δ_σ = |σ_new − σ_old|`.

### Hidden-confounder variant (`hidden = True`)

Generate the 6-node visible SCM exactly as above (same seed ⇒ same visible graph as the main
run). Choose an existing edge `A → B` uniformly at random from the visible edges. Add a
seventh variable `H` with no parents, `H → A` and `H → B`, weights from the same rule,
`b_H ~ U[−1, 1]`, `σ_H ~ U[0.5, 1.5]`. `H` is removed from every agent's view (samples,
queries, budget). Queries never target `H`. The shift schedule is as in the main run, except
that the 4th shift of the 8 is a `large`-type shift of mechanism `H` (`b_H`, `σ_H`
resampled), so that credit misassignment under a hidden shift is exercised once per seed.

The world computes per seed the **population confounding bias**
`c = β_A − W[B, A]`, where `β` is the population OLS coefficient vector of `X_B` on
`[1, X_pa_vis(B)]` from the 7-node observational covariance and `pa_vis(B)` are `B`'s visible
true parents (which include `A`). It also computes the implied nMSE on `do(A) → B` queries,
`F_conf = c² · E[v²] / Var_ref(X_B) = c² · (4/3) / Var_ref(X_B)`.

The latent projection of the 7-node graph onto the visible variables has directed part equal
to the visible 6-node DAG plus a bidirected `A ↔ B`; SHD in this variant is computed against
the directed part only.

### RNG streams (common random numbers)

All randomness comes from `numpy.random.default_rng` with explicit keys; no agent's choices
can affect what any other agent sees:

| Purpose | Key |
|---|---|
| graph, weights, intercepts, noise scales, relabelling permutation, hidden edge choice | `[seed, 0]` |
| shift schedule and all shift parameters (drawn up front for all `K` shifts) | `[seed, 1]` |
| observational noise for episode `t` (`N_obs × d'` array) | `[seed, t, 2]` |
| interventional noise for episode `t` (`B × d'` array, drawn up front) | `[seed, t, 3]` |
| query set for episode `t` | `[seed, t, 4]` |
| agent-internal randomness (bootstrap, posterior draws, tie-breaks, random targets) | `[seed, agent_id, 5]` |

(`d' = 7` in the hidden variant.) Interventional sample `k` of episode `t` uses row `k` of
the pre-drawn noise array **whatever target and value the agent chooses**: agents that pick
different targets still share the exogenous noise. This restores the paired design and
reduces the variance of every between-agent difference.

## Episode protocol

Every agent runs the same loop. Steps are executed in this order and nothing else happens.

1. World applies the pre-drawn shift for episode `t`, if any. The current SCM is now fixed
   for the whole episode; ground truth for step 8 is this SCM.
2. World draws the observational batch (`N_obs = 200` rows), the interventional noise array
   (`B × d'`), and the query set (`Q = 200`) from their keyed streams.
3. Agent receives the observational batch.
4. **Detection** (agents with a detector): for each mechanism with ≥ `N_min = 100` rows in
   its buffer, compute the shift statistic on the new observational batch under the current
   posterior (defined below). Apply the agent's reset rule (per-mechanism, global, oracle, or
   none). A reset truncates the buffer of the affected mechanism(s) to empty and sets its
   detector state to zero.
5. Agent appends the observational batch to every mechanism buffer and **refits** all
   mechanism posteriors from their buffers (posteriors are always a deterministic function
   of buffer, parent set and prior; there is no in-place updating).
6. **Interventions**, in two rounds:
   - Round 1 (round-robin floor, identical for every mech-* agent and `int-pairwise`):
     `F = ceil(B / (2d))` samples on each variable, values cycled per variable through
     `[2, −2, 1, −1]` with the cycle position persisting across episodes. Rows are appended
     to buffers and posteriors are refit.
   - Round 2 (free budget `B − dF`): allocated by the agent's rule (active, random, or none).
     In episodes 1–3 every agent uses round-robin for the free budget too (warm-up).
   For every interventional sample, the agent first records its self log-score for that
   sample (metric 9) under the posterior at that moment, then receives the sample and
   appends it to the buffers of all mechanisms other than the target.
7. **Structure re-estimation** (learned-structure agents): slope tests, order extraction,
   parent selection (below). If mechanism `j`'s parent set changed, refit `j` from its
   buffer under the new parent set, set its detector state to zero, and log a
   *structure-induced refit* event `(seed, t, j)`.
8. Agent answers the `Q` queries; predictions are scored against the current SCM.

Budget arithmetic at `d = 6`: `B = 10 → F = 1` (6 floor, 4 free); `B = 25 → F = 3` (18, 7);
`B = 50 → F = 5` (30, 20); `B = 100 → F = 9` (54, 46).

## What the agent sees each episode

- `N_obs = 200` observational rows from the current SCM.
- `B = 50` interventional rows chosen as in step 6: target `i`, value `v ∈ {−2, −1, 1, 2}`,
  full row from the mutilated SCM with `X_i := v`.
- Nothing else. No graph, no order, no shift signal, no permutation. Non-oracle agents
  receive a `WorldView` object exposing only `observe()` and `intervene(i, v)`.

## What the agent is asked each episode (evaluation, held out)

`Q = 200` queries `do(X_i = v)` with `i` uniform over visible variables and `v ~ U(−2, 2)`.
For each query and every visible `j ≠ i` the agent returns:

- a point prediction of `E[X_j | do(X_i = v)]`,
- quantiles of its belief about that mean at levels `{0.005, 0.05, 0.25, 0.75, 0.95, 0.995}`
  (the raw width `q_0.95 − q_0.05` is the agent's uncertainty score; the evaluator, not the
  agent, normalizes it by `sd_ref(X_j)` for cross-agent comparison),
- a boolean `abstain = (q_0.95 − q_0.05 > τ_agent)`, using the agent's own raw width and its
  own tuned `τ_agent`.

Ground truth is computed exactly from the current SCM.

## Agents

| Name | Structure | Update rule | Uncertainty | Intervention choice | Tests requirement |
|---|---|---|---|---|---|
| `marginal-mean` | none | predicts the cumulative observational sample mean of `X_j`, ignoring `(i, v)` | bootstrap of the mean | none | floor reference (nMSE), zero compute |
| `obs-window` | none: per-pair OLS of `X_j` on `[1, X_i]`, observational rows only ("see" for "do") | sliding window of last `W` episodes (`W` tuned) | nonparametric bootstrap of rows, 200 resamples | none (ignores budget) | baseline for 1, 3, 5 |
| `obs-cumulative` | none, as above | all observational rows ever seen | bootstrap | none | forgetting vs. staleness trade-off |
| `int-pairwise` | none: per-pair OLS of `X_j` on `[1, v]` over rows where `X_i` was clamped; consistent for `E[X_j | do(X_i = v)]` with no graph | sliding window `W` (tuned) | bootstrap | floor + uniform random (identical policy to `mech-random`) | the strong non-factored competitor; isolates factorization (3) |
| `mech-full` | learned DAG | per-mechanism NIG regression; **only** mechanisms whose detector fires are reset | posterior draws propagated through the mutilated learned graph | floor + active | the proposed system |
| `mech-random` | learned DAG | as `mech-full` | same | floor + uniform random | ablates active selection |
| `mech-full-nofloor` | learned DAG | as `mech-full` | same | all `B` on the active rule (no floor; the v1 rule) | ablates the exploration floor |
| `mech-reset-all` | learned DAG | when **any** mechanism's detector fires, **all** mechanism buffers are truncated | same | floor + active | ablates 5 (the agent that overwrites) |
| `mech-no-detect` | learned DAG | no detector; exponential forgetting `γ` on every mechanism's sufficient statistics | same | floor + active | adaptation-speed ablation (secondary) |
| `mech-oracle-detect` | learned DAG | no detector; the world tells it the shifted mechanism at step 4 of each shift episode; that buffer is truncated | same | floor + active | oracle for 4 only; decomposes detection vs structure vs estimation |
| `mech-overconfident` | learned DAG | as `mech-full` but prior `V0 = 0.1 I`, `a0 = 50`, `b0 = 50` | same | floor + active | probe for 6 (gameable objective) |
| `oracle-structure` | true DAG given (visible sub-DAG in the hidden variant) | as `mech-full` | same | floor + active | separates structure error from parameter error |
| `oracle` | true SCM | none | exact; width 0 | none | ceiling |
| `mech-full-diag` (hidden variant only) | learned DAG | as `mech-full`, but mechanism `B`'s buffer holds only rows where `A` was clamped (uses oracle knowledge of `(A, B)`) | same | floor + active | diagnostic for P6, not an agent claim |

All agents receive identical observational batches, query sets and shift schedules, and
share interventional noise rows (RNG streams above). `mech-full-nofloor` runs in the main
run and in the budget sweep at `B ∈ {10, 25}` only.

### Baseline details

- **Per-pair OLS with intercept.** For `obs-*`: regress `X_j` on `[1, X_i]` over the agent's
  observational pool; predict `β̂_0 + β̂_1 v`. For `int-pairwise`: regress `X_j` on `[1, v]`
  over pool rows where the target was `i`. Bootstrap: 200 nonparametric resamples of the
  pool's rows (one resample of row indices shared across all pairs, drawn from the agent's
  stream), refit, take the six quantiles of the resampled prediction. Guards: a fit needs
  `n ≥ 4` rows and (for `int-pairwise`) ≥ 2 distinct `v`; otherwise, and whenever the
  regressor variance is `< 1e−8`, the pair falls back to the `marginal-mean` rule (sample
  mean of `X_j` over the agent's observational pool; `int-pairwise` also receives the
  observational batch for this purpose only). No ridge penalty: every baseline fit has two
  parameters and at least four rows.
- `marginal-mean`: never abstains; excluded from selective-risk comparisons.
- `oracle`: exact point, all quantiles equal to the truth, never abstains.

### Mechanism models (all `mech-*` and `oracle-structure`)

Mechanism `j` is `X_j ~ N(x' w, σ²)` with `x = [1, X_pa(j)]` (an intercept is **always**
fitted; zero means are never assumed) and a Normal–Inverse-Gamma prior:
`w | σ² ~ N(0, σ² V0)`, `σ² ~ IG(a0, b0)`, with `V0 = 10 I`, `a0 = 2`, `b0 = 1` (prior mean
of `σ²` is 1, matching the noise-scale range). `mech-overconfident` uses `V0 = 0.1 I`,
`a0 = 50`, `b0 = 50` (prior mean 1, prior sd ≈ 0.14).

Buffer of mechanism `j`: all rows received since `j`'s last reset in which `j` was **not**
the intervention target (rows where a parent of `j` was clamped are valid for `j`). With
weights `ω_r` per row (1 for every agent except `mech-no-detect`, where
`ω_r = γ^(t − t_r)`, `t_r` the episode of row `r`):
`S_xx = Σ ω x x'`, `S_xy = Σ ω x y`, `S_yy = Σ ω y²`, `n_eff = Σ ω`;
`V_n = (V0⁻¹ + S_xx)⁻¹` (via `np.linalg.solve`), `m_n = V_n S_xy`, `a_n = a0 + n_eff / 2`,
`b_n = b0 + ½ (S_yy − m_n' V_n⁻¹ m_n)`.
Posterior draws: `σ²_s = 1 / rng.gamma(shape = a_n, scale = 1 / b_n)`,
`w_s = m_n + sqrt(σ²_s) · chol(V_n) z_s`, `z_s ~ N(0, I)`.
Posterior predictive for a row `x`: Student-t with `ν = 2 a_n`, location `x' m_n`, scale²
`(b_n / a_n)(1 + x' V_n x)`.
Degenerate cases: an empty buffer gives the prior; the intercept-only model (no learned
parents) is the ordinary NIG on a column of ones.

`mech-no-detect` uses `γ` tuned on validation (grid `{0.5, 0.7, 0.9}`); its buffers are never
truncated.

### Structure learning (learned-DAG agents; re-run at step 7 every episode)

Structure pool: all interventional rows from the last `W_struct = 10` episodes (including the
current one). A shorter pool than "since episode 1" is used because a `large` shift can flip
a weight's sign, and a pooled slope over a sign flip loses power; the pool bounds the mixing.

1. **Ancestor slope test.** For each ordered pair `(i, j)`, `i ≠ j`: take the pool rows where
   `X_i` was the clamped target; regress `X_j` on `[1, v]` by OLS; compute the two-sided
   t-test p-value for slope = 0 with `n − 2` degrees of freedom. Require `n ≥ 4` and ≥ 2
   distinct `v`, otherwise `p := 1`. Apply Benjamini–Hochberg at `α = 0.01` across the
   `d(d − 1)` pairs; rejected pairs are the estimated ancestor relation `Â`. This has full
   power under the symmetric value cycle, needs no observational rows, and is robust to
   mixed regimes (a mixture of slopes of the same sign is nonzero; a mixture of zero slopes
   is zero). If BH rejects nothing, `Â` is empty and every mechanism is intercept-only.
2. **Order extraction (robust to cycles and non-transitivity in `Â`).** Draw one random
   permutation `π` from the agent's stream at episode 1 and keep it for the run. Repeat `d`
   times: among the not-yet-placed nodes, pick the one with the fewest not-yet-placed
   estimated ancestors; break ties by `π`; append it. This always terminates and coincides
   with a topological sort when `Â` is consistent. Then drop every `(i, j) ∈ Â` with `i`
   placed after `j`. The surviving relation is acyclic and the learned `W` is strictly
   lower-triangular in the learned order, so `I − W` is unit-triangular and invertible.
3. **Parent selection.** Candidate parents of `j` = surviving ancestors of `j`. Fit OLS of
   `X_j` on `[1, candidates]` over **mechanism `j`'s own buffer** (rows since `j`'s last
   reset, `j` not clamped). Keep candidates with two-sided t-test `p < 0.05` (no BH; at most
   5 regressors). Guards: require `n ≥ p + 5` and full column rank (drop constant columns
   first); otherwise keep `j`'s previous parent set. Property: in population, regressing on
   a superset of the true parents gives zero coefficients on the extras, so ancestors that are
   not parents are excluded as data accumulates.

The topological-order leak in v1 is closed twice: by the relabelling permutation in the world
and by the agent-private tie-break `π`.

### Shift detection (agents with a detector)

For mechanism `j` with current design `x_r = [1, X_pa(j)]`, `p = |pa(j)| + 1`, on the `N_obs`
rows of the new observational batch, under the posterior from `j`'s buffer *before* the batch
is appended:

- `ℓ_old = Σ_r log t_ν(y_r; x_r' m_n, (b_n / a_n)(1 + x_r' V_n x_r))`, `ν = 2 a_n`;
- `ℓ_new = Σ_r log N(y_r; x_r' ŵ, σ̂²)`, with `ŵ` the OLS fit on the batch alone and
  `σ̂² = RSS / N_obs`;
- `GLR_t = ℓ_new − ℓ_old` (≈ `(p + 2) / 2` in expectation under no change);
- Page–Hinkley: `g_t = max(0, g_{t−1} + GLR_t − δ)` with drift `δ = p + 2`; **fire** when
  `g_t > λ`.
- On fire: truncate the buffer (per the agent's reset rule), set `g := 0`.
- Guards: detection is skipped and `g` held at 0 for any mechanism with fewer than
  `N_min = 100` rows since its last reset (this covers episode 1). `g_j := 0` whenever `j`'s
  parent set changes (step 7).

Reset rules: `mech-full`, `mech-random`, `mech-full-nofloor`, `mech-overconfident`,
`oracle-structure`: truncate only `j`. `mech-reset-all`: if any `g_j > λ`, truncate all `d`
buffers and zero all `g`. `mech-oracle-detect`: no statistic; truncate exactly the shifted
mechanism in each shift episode (including `noise-only` shifts, so its self-inflicted regret
on those is the same as `mech-full`'s would be with perfect detection).

Known behavior to expect, not a bug: under a wrong parent set (missing parent), an upstream
shift changes the omitted-variable term and mechanism `j` fires spuriously. This is precisely
what `oracle-structure` isolates. Under `γ_h = 1`, an upstream shift changes the *scale* of
downstream residuals without changing the downstream mechanism; the detector, which assumes a
constant scale given the parents, is expected to misattribute (P7).

### Answering `do(X_i = v)` (vectorized, closed form)

Per episode: draw `S = 1000` joint posterior samples (all mechanisms independently):
`W_s` of shape `(S, d, d)` (row `j` holds `w_s` for `pa(j)`, zeros elsewhere), `b_s` of shape
`(S, d)`. For every possible target `i`, form the mutilated `W_{s,i}` (row `i` zeroed) and
compute `M_{s,i} = (I − W_{s,i})⁻¹` with one `np.linalg.inv` call on an array of shape
`(d, S, d, d)`. For query `(i, v)`: `μ_s = M_{s,i} (b_s with b_i := 0) + v · M_{s,i}[:, i]`;
all `Q` queries in one `einsum`. Point = mean over `s`; the six quantiles over `s`;
`abstain = (q_0.95 − q_0.05 > τ_agent)`. Store per prediction: normalized error, normalized
width90, three coverage booleans (50/90/99), abstain, and the query indices — not the draws.

This is a Monte Carlo over *parameter* uncertainty propagated through the *learned* graph. It
contains no term for the graph being wrong. Calibration is therefore stratified by structural
correctness (Metric 4) so that misses can be attributed.

### Active intervention rule (requirement 1 as a heuristic)

`score(i) = Σ_{k ≠ i} Var_s[μ_{s,k}(i, v = 2)] / v̂_k`, where the variance is over the same
`S` draws used for answering (reuse the previous episode's `M_{s,i}` or recompute; six
candidates cost one `einsum`), and `v̂_k` is the agent's own sample variance of `X_k` over the
last 3 episodes of observational rows (an agent-side estimate, never `Σ_ref`). The free budget
is spent entirely on `argmax_i score(i)`, values cycled `[2, −2, 1, −1]`; ties (including the
all-zero case when the learned graph is empty) are broken by the agent's stream. There is no
"half the budget on parents of `j`" clause: after a reset, `j`'s posterior is wide, so queries
into `j` have high answer variance and the rule targets `j`'s ancestors automatically.
`mech-random` and `int-pairwise`: free budget on uniform random targets with the same value
cycle. `oracle-structure` uses the active rule with the true DAG.

### Self log-score (requirement 6 as a hypothesis)

Before interventional sample `k` with target `i`, value `v` is revealed, the agent computes,
under its current posterior, `Σ_{j ≠ i} log t_ν_j(x_j; x_pa(j)' m_j, s_j²)` on the realized
row (the factorized posterior predictive log density; parents' realized values are taken
from the same row). Per episode the agent logs (a) `self_score_t` = mean over the `B`
samples and (b) `var_obj_t = Σ_i score(i)` before and after the episode's updates. Both are
written to the raw output for every mech-* agent. Neither is used by any agent to change its
behavior; they are the two candidate "improvement objectives" that P8 compares.

## Tuning and calibration

- Evaluation seeds: `0–19` (hidden variant: `0–39`). Validation seeds: `1000–1009`, disjoint.
  No tuning on hidden or knob variants; `run.py` refuses to run evaluation seeds without a
  `constants.json` and logs its SHA-256 in every results file.
- For each agent independently, minimize mean nMSE (Metric 1) over validation episodes over
  its grid: `obs-window` `W ∈ {1, 2, 3, 5}`; `int-pairwise` `W ∈ {3, 5, 8, 12}`;
  `mech-no-detect` `γ ∈ {0.5, 0.7, 0.9}`.
- `λ` (one value for every detector agent): run `mech-full` on validation seeds with the shift
  schedule **disabled**; collect all `g_t`; set `λ` to the smallest value giving ≤ 1 false
  reset per 100 mechanism-episodes. Additionally run `mech-full` on validation seeds *with*
  shifts at `λ ∈ {½, 1, 2, 4, 8} × λ*` and report the detection ROC (recall on `m ≥ 0.05`
  shifts vs. false-reset rate) as an exploratory figure.
- `τ_agent`: the 90th percentile of raw width90 over that agent's predictions in validation
  *stable* episodes (target abstention rate 10 % per agent, matched by construction).
  `marginal-mean` and `oracle` never abstain.
- Fixed constants (not tuned): `V0, a0, b0, N_min, δ, W_struct, α_BH = 0.01, α_parent = 0.05,
  S = 1000, bootstrap 200`.
- Validation costs about half of one evaluation sweep.

## Metrics

Definitions used throughout: **shift episode** = the episode `t` at whose start a mechanism
was resampled; **post-shift** = `t + 1`; **stable** = all other episodes. Per-seed scalars are
computed first; CIs are over seeds (Analysis plan). `Var_ref(X_j)` is the diagonal of `Σ_ref`
for the current SCM.

1. **Normalized interventional MSE (nMSE)**: `(prediction − truth)² / Var_ref(X_j)`, averaged
   over all queries and all `j ≠ i` (primary). Secondary table: split by `j` a true
   descendant of `i` vs not (the non-descendant half is the false-effect rate; truth is
   `E[X_j]` there and anti-causal predictions show up). Always report the `marginal-mean`
   floor and the realized abstention rate next to any non-abstained number.
2. **Post-shift regret and recovery**, for shifts with `m ≥ 0.05` (or `δ_σ` for `noise-only`):
   - *Regret*: `Σ_{u = t}^{t+3} (nMSE_agent(u) − nMSE_oracle-structure(u))` over the shift episode
     and the three following (no threshold; captures delay and magnitude). Raw cumulative
     nMSE over the same window is exploratory.
   - *Recovery time*: pre-shift level `L` = mean nMSE over the 2 episodes before the shift;
     recovered at the first episode `u ≥ t` with `nMSE(u) ≤ max(1.5 L, L + 0.02)`;
     right-censored at the next shift or `T`. Report the Kaplan–Meier median with a
     seed-level cluster bootstrap CI and the fraction recovered within 1, 2, 3 episodes.
3. **Forgetting**: after a shift in mechanism `j`, nMSE restricted to **`j`-independent**
   queries. Query `(i, k)` is `j`-dependent iff `k = j` or (`j ∈ desc(i)` and `j ∈ anc(k)`)
   in the **true** DAG; if `j = i` the query is independent (the intervention cuts `j`'s
   mechanism). Reachability comes from the transitive closure of the true adjacency. The
   forgetting bump is `nMSE_indep(t) − mean nMSE_indep(t − 2, t − 1)`. Test case: chain
   `0 → 1 → 2`, shift in 1: `(0, 2)` and `(0, 1)` dependent; `(1, 2)` and `(2, 0)` independent.
4. **Calibration**: empirical coverage of the 50 %, 90 % and 99 % intervals, computed on
   **all** predictions (never the non-abstained subset), each reported next to the mean
   normalized width at that level. Split by stable / shift / post-shift episodes. For
   learned-structure agents, additionally stratify by (a) `SHD = 0` vs `SHD > 0` at that
   `(seed, t)` and (b) whether the set of directed paths `i → j` in the learned graph equals
   the true set for that query; `oracle-structure` is the parameter-only reference on the
   same queries.
5. **Selective risk**: every agent's uncertainty score is its normalized width90. Report the
   risk–coverage curve, the risk at fixed coverages `{0.5, 0.8, 0.9, 1.0}` obtained by
   thresholding each agent's own score (comparable across agents by construction), and AURC
   (coverage grid of 100 points in `(0, 1]`, trapezoid integral of `risk(c)`), alongside the
   oracle-ordering AURC (sort by true error) and random-ordering AURC (overall nMSE) as
   references. A **mistake** is absolute: normalized squared error `> 0.25`. Report mistake
   recall `P(abstain | mistake)` and precision `P(mistake | abstain)` at the matched 10 %
   abstention rate.
6. **Structure recovery**: SHD between the learned and true DAG per episode for learned-
   structure agents; against the visible sub-DAG in the hidden variant.
7. **Value of interventions**: `mech-full`, `mech-random`, `mech-full-nofloor`,
   `int-pairwise` as a function of `B ∈ {10, 25, 50, 100}` at fixed `N_obs`; nMSE primary,
   SHD secondary (so a gap can be attributed to structure or to parameters at `SHD = 0`).
8. **Credit assignment (direct)**: from the logged reset events `(seed, t, j)` (detector-
   induced; structure-induced refits logged separately and not counted here) and true shifts
   `(seed, t, j, m)`: detection delay (episodes from shift to first reset of the shifted
   mechanism, right-censored at the next shift); attribution precision (fraction of reset
   events matching a shift with `m ≥ 0.05` in the same or previous episode); false-reset
   rate per mechanism-episode in stable periods; misattribution rate (a reset in a shift
   episode or the one after, on a non-shifted mechanism). Cross-tabulated as
   correct / wrong-mechanism / false-alarm / miss per shift.
9. **Internal objectives**: per episode, `self_score_t` and the decrease in `var_obj_t`,
   for every mech-* agent.

## Analysis plan

- One primary scalar per seed per pre-registered prediction (stated with each prediction).
  Percentile bootstrap over the 20 per-seed scalars (2000 resamples), paired across agents
  where the scalar is a difference. One-sided bootstrap p-value against the stated threshold;
  Holm correction across the primaries at `α = 0.05`; unadjusted 95 % CIs reported alongside.
  A prediction "held" iff its Holm-adjusted test rejects in the predicted direction.
- Any event-pooled statistic (recovery, detection delay, attribution) uses a seed-level
  cluster bootstrap.
- All per-episode curves and every table not tied to a primary are labelled exploratory.
- Time 2 seeds of the full sweep first; if the projected 4-core wall time is under 7 minutes,
  raise evaluation seeds to `0–39` for all runs within the 15-minute budget.

## Pre-registered predictions (written before any code is run)

- **P1 (see is not do; quantitative).** In `world.py` compute per seed and episode the
  population see-for-do bias of `obs-window`'s estimand for each ordered pair:
  `β_1 = Σ_ref[j, i] / Σ_ref[i, i]`, `β_0 = μ_j − β_1 μ_i`,
  `bias(v) = (β_0 − c_j(i)) + (β_1 − Θ[j, i]) v`, and the asymptotic floor
  `F_obs = mean_{i ≠ j} E_v[bias(v)²] / Var_ref(X_j)` with `E_v[v²] = 4/3`. Primary scalar:
  the per-seed mean over stable episodes at least `W` episodes after the last shift of
  `(nMSE_obs-window − F_obs)`. Prediction: it lies in `[−0.1 F_obs − 0.01, 0.1 F_obs + 0.01]`
  in ≥ 80 % of seeds (the error is the population bias, not noise, and does not shrink), and
  `mech-full`'s nMSE is below `obs-window`'s by episode 5 in ≥ 90 % of seeds. Secondary:
  `obs-*` nMSE on non-descendant pairs is bounded away from zero wherever `j` is an ancestor
  of `i` or shares a common ancestor with `i` (anti-causal predictions); on descendant pairs
  with no back-door path it converges to the truth. Note that this is a check that the
  testbed does what it says, not a discovery.
- **P2 (recovery).** Primary scalar per seed: mean over `m ≥ 0.05` shifts of
  `regret(mech-full) − 2 · regret(mech-oracle-detect)`. Prediction: `≤ 0` (detection costs at
  most as much again as relearning). Secondary: `mech-full` recovers within 2 episodes for
  ≥ 90 % of `m ≥ 0.05` shifts; `obs-window` recovers at about `W` episodes; `obs-cumulative`'s
  recovery is censored at the next shift in ≥ 50 % of events; `obs-window`'s regret grows
  with the number of true descendants of `j` while `mech-full`'s does not (rank correlation,
  exploratory).
- **P3 (calibration).** (a) In stable episodes with `SHD = 0`, `mech-full`'s 90 % coverage is
  within ±5 points of nominal (primary scalar: per-seed coverage in that stratum minus 0.90;
  prediction: the CI of its mean lies inside `[−0.05, 0.05]`). (b) In shift episodes with
  `m ≥ 0.05` where the detector did **not** fire on the shifted mechanism at step 4,
  `mech-full`'s 90 % coverage on `j`-dependent queries is below 50 % **and** its abstention
  rate on those queries does not exceed its stable-episode rate by more than 2 points: a
  Bayesian model without a misspecification term is confidently wrong about what it has not
  detected. (c) `int-pairwise`'s coverage is near nominal (±5) in stable episodes, and at
  `B = 10` its nMSE at episode 60 is at least 3× `mech-full`'s, because each pair sees only
  ≈ `B / d` rows per episode.
- **P4 (memory that does not overwrite).** After a shift with `m ≥ 0.05` in mechanism `j`,
  `mech-reset-all`'s forgetting bump (Metric 3) in the shift episode is at least `0.05`;
  `mech-full`'s is below `0.01`. Primary scalar: per-seed mean over such shifts of
  `bump(mech-reset-all) − bump(mech-full)`; prediction `≥ 0.04`. `obs-window` and
  `int-pairwise` (windowed, monolithic pools) also show a bump; `mech-no-detect` shows no bump
  but a recovery time of about `1 / (1 − γ)` episodes (secondary).
- **P5 (value of structure and of active selection).** (a) Primary scalar: paired per-seed
  difference in mean nMSE over episodes 20–60 at `B = 10`, `int-pairwise − mech-full`;
  prediction `> 0`. At `B = 100` the same difference is within `±0.02`: structure buys sample
  efficiency, not correctness. (b) `mech-full − mech-random` at `B = 10` and `B = 25` is `< 0`
  (paired); the gap closes by `B = 100`. (c) `mech-full-nofloor` has worse SHD than
  `mech-random` at `B = 10` (per-seed mean SHD over episodes 10–60; paired difference `> 0`),
  and `mech-full` (with floor) has SHD no worse than `mech-random`.
- **P6 (the failure case: hidden confounder).** Three separable claims on the constructed
  pair `(A, B)`:
  (i) `mech-full`'s SHD to the visible DAG is 0 by episode 10 in ≥ 80 % of seeds: no wrong
  edge appears, because interventional ancestor tests are immune to latent confounding
  (`do(A)` moves `B` through the real edge only; `do(B)` does not move `A`).
  (ii) The bias is omitted-variable bias in mechanism `B`'s coefficient on `A`, diluted by
  the `do(A)` rows in `B`'s buffer: the signed error of `mech-full`'s point prediction on
  `do(A) → B` queries at `v = 2` over episodes 40–60 has the sign of `c` and magnitude in
  `[0.3 |c| · 2, 1.0 |c| · 2]` in ≥ 80 % of seeds.
  (iii) `mech-full`'s 90 % coverage on `do(A) → B` queries over episodes 40–60 is below 50 %
  (primary scalar: per-seed coverage on that pair; prediction: CI upper bound `< 0.5`), its
  abstention rate on that pair does not exceed its global rate by more than 2 points, and
  `obs-*` and `oracle-structure` (visible DAG) are biased on the same pair.
  (iv) The scheduled shift in `H` fires the detectors of both `A` and `B` in the same episode
  in ≥ 50 % of seeds (credit misassigned to two visible mechanisms; there is no correct
  attribution available).
  Diagnostic (oracle knowledge, not an agent claim): `mech-full-diag`, which fits mechanism
  `B` only on rows where `A` was clamped, recovers coverage on `(A, B)` to within ±10 points
  of nominal, which pins the cause on the observational rows. This is the honest limit:
  requirement 2 cannot be met by uncertainty over the wrong model class, and requirement 4
  cannot attribute a shift in a variable the model does not have.
- **P7 (credit assignment, direct).** Main run: `mech-full` attributes ≥ 80 % of `m ≥ 0.05`
  shifts to the correct mechanism with delay 0 (primary scalar: per-seed attribution
  accuracy; prediction: CI lower bound `≥ 0.8`), with false-reset rate `≤ 2 %` per
  mechanism-episode in stable periods. Under `γ_h = 1`, `mech-full`'s attribution accuracy
  drops by at least 15 points and its misattribution rate rises (an upstream shift changes
  downstream residual scale), while `mech-oracle-detect`'s nMSE is unaffected by `γ_h` to
  within `0.02`, which pins the failure on the detector rather than the estimator.
- **P8 (a gameable and a non-gameable objective).** Primary scalar: per-seed difference
  `self_score(mech-full) − self_score(mech-overconfident)` averaged over episodes 10–60;
  prediction `> 0`. Simultaneously `mech-overconfident` has a larger per-episode decrease in
  `var_obj` in episodes 1–10, abstains on `< 2 %` of predictions, and has worse 90 % coverage
  (by ≥ 20 points) and worse AURC than `mech-full`. That is the smallest falsifiable form of
  "variance reduction is gameable; a proper score on world-generated outcomes is not".
- **P9 (shift types).** `noise-only`: `mech-full`'s regret (Metric 2, against
  `oracle-structure`) over the 4 post-shift episodes is positive and at least `0.02` on
  average (primary scalar per seed; the reset discards correct weights — the price of the
  reset rule), and `mech-oracle-detect` pays the same price. `small`: detection recall on
  `m ≥ 0.05` shifts falls below 60 %, and undetected shifts produce coverage on `j`-dependent
  queries below 70 % for at least 2 episodes with no rise in abstention.

## Compute budget

Everything in numpy/scipy on CPU, parallelized over `(agent, seed)` with
`multiprocessing.Pool(workers = 4)`. Estimated cost per agent-episode: mech-type ≈ 30 ms
(30 slope tests, 6 refits, `S = 1000` draws with `d` batched inverse arrays, one einsum over
`Q`), baselines ≈ 5 ms, oracle ≈ 1 ms.

| Run | Agent-episodes | Single-core estimate |
|---|---|---|
| main: 13 agents × 20 seeds × 60 episodes at `B = 50` | 15,600 | ≈ 5.5 min |
| budget sweep: `{mech-full, mech-random, int-pairwise}` × `B ∈ {10, 25, 100}`, plus `mech-full-nofloor` at `{10, 25}`, × 20 × 60 | 13,200 | ≈ 5.5 min |
| hidden: `{mech-full, mech-random, int-pairwise, obs-cumulative, oracle-structure, mech-oracle-detect, oracle, mech-full-diag}` × 40 seeds × 60 | 19,200 | ≈ 6.5 min |
| knobs: 5 configs (`γ_h × shift_type` minus the main) × `{mech-full, mech-oracle-detect, oracle-structure, int-pairwise}` × 20 × 60 | 24,000 | ≈ 9.5 min |
| validation and `λ` calibration (10 seeds, all agents; ROC grid for `mech-full`) | ≈ 11,000 | ≈ 4 min |

≈ 31 core-minutes, ≈ 8 minutes on 4 cores. Target remains **under 15 minutes on 4 cores**.
If the 2-seed timing run projects an overrun, cut in this order: knob-grid seeds to 10; hidden
seeds to 20; drop `obs-cumulative` and `mech-no-detect`; `T` to 50. Per-(agent, seed) wall
time is written to `config.json` so a regression is visible.

## Deliverables

```
experiments/mechanism-shift/
  SPEC.md                         this file
  world.py                        SCM generator, relabelling, shift schedule and types, hidden
                                  variant, exact interventional means, Σ_ref, m, δ, F_obs, c,
                                  WorldView (observe / intervene only), keyed RNG streams
  agents/                         one file per agent family (baselines.py, mech.py, oracle.py)
  metrics.py                      all nine metrics, j-dependence mask, KM recovery, AURC
  calibrate.py                    writes constants.json (W, γ, λ, τ per agent) from seeds 1000-1009
  run.py                          CLI: --agents all|a,b,c --seeds 0-19 --episodes 60 --budget 50
                                  --n-obs 200 --shift-type large|small|noise-only --hetero 0|1
                                  --hidden --workers 4 --constants constants.json --out results/<name>
  report.py                       reads a results dir, writes metrics.json, figures/, REPORT.md tables
  tests/                          property tests (list below)
  results/<run-name>/config.json  all parameters, git hash, constants.json hash, per-(agent, seed) wall time
  results/<run-name>/raw/         <agent>__seed<s>.npz (np.savez_compressed): float32 err[T,Q,d-1],
                                  width90[T,Q,d-1]; bool cov50/cov90/cov99/abstain[T,Q,d-1];
                                  int query_i[T,Q], shift_j[T] (-1 if none), reset events,
                                  structure-refit events; learned adjacency per episode;
                                  self_score[T], var_obj[T]; world W, b, σ per episode
                                  (≈ 10 MB per agent-seed; git-ignored)
  results/<run-name>/metrics.json per-seed scalars and bootstrap CIs
  results/<run-name>/figures/     PNG
  REPORT.md                       tables + figures + which predictions held (Holm-adjusted)
```

### Tests (`tests/`)

1. Exact interventional mean agrees with `10⁵` mutilated-SCM samples within 4 standard
   errors, for `γ_h ∈ {0, 1}`; affine in `v`; equals `E[X_k]` for non-descendants for all `v`;
   unchanged for every `(i, k)` with `j` not on a directed `i → k` path after a shift in `j`.
2. Two dummy agents with different intervention policies observe byte-identical
   observational batches, query sets, shift schedules and interventional noise rows.
3. Leakage: `WorldView` has no attribute exposing `W`, `b`, `σ`, the permutation or the
   schedule; `agents/` never imports world internals (grep assertion). Running `mech-full`
   on a relabelled copy of the same world yields identical predictions up to relabelling.
4. NIG: posterior mean matches OLS to `1e−3` at `n = 10⁴`; batch update equals two sequential
   updates; prior-predictive check: 2000 draws of `(w, σ²)` from the prior, `n = 30` rows
   each, empirical 90 % coverage of `w_1` in `[0.87, 0.93]` (catches a wrong Gamma
   parameterization or a dropped `1 + x' V x`).
5. Structure: with 2000 interventional rows per target on 20 random DAGs, `SHD = 0` in ≥ 19;
   recovered ancestor relation equals the true transitive closure in ≥ 95 % of 50 graphs
   with `N = 50` rows per target. Order extraction returns a permutation and an acyclic edge
   set on a fully symmetric "ancestor" matrix. All-ones p-values yield an empty graph and no
   exception.
6. Detector: with no shifts, ≤ 1 false reset per 100 mechanism-episodes at `λ*`; a weight
   change of 1.0 on a unit-variance parent fires within the same episode in ≥ 90 % of 100
   trials; only mechanism `j` fires when structure is correct and `γ_h = 0`; no re-fire in
   the episode after a reset.
7. Vectorized propagation matches a naive per-draw per-query topological loop on `d = 4`,
   `S = 3`, `Q = 2` to `1e−10`.
8. Metrics on hand-computed cases: recovery time on a fixed nMSE series (no-bump, censored,
   immediate); the `j`-dependence mask on the 3-node chain; coverage on a fixed interval
   list; AURC on 3 predictions; attribution confusion on a fixed event log.
9. Hidden generator produces the `H → A → B`, `H → B` motif with `A → B` a visible edge;
   queries and budgets never target `H`; `c` and `F_conf` match a `10⁵`-sample OLS.
10. Shift schedule: exactly 8 shifts in `[7, 55]`, minimum gap 5, identical across agents;
    `m` matches a Monte Carlo estimate.
11. Baselines: `obs-window` prediction equals the Gaussian conditional
    `μ_j + Σ_ji / Σ_ii (v − μ_i)` on the same rows; `int-pairwise` recovers `Θ[j, i]` to
    within 3 standard errors from 2000 clamped rows.
12. Smoke: one `mech-full` seed for 3 episodes completes in under 2 s; the full 13-agent set
    on one seed for 3 episodes raises no exception at `B = 10` (small-`n` guards).

## Non-goals

No neural networks, no language, no images. No nonlinear mechanisms in this version: a mild
edge nonlinearity would change the estimator and the detector at once and would need Monte
Carlo ground truth; the heteroscedastic knob is the one departure from the agent's home turf
that keeps the truth exact. The question is whether the six requirements, implemented in the
simplest possible form, produce the behaviors claimed for them. If they do not at `d = 6`,
they will not at scale.

## Changes from v1

Blocking issues (all three critics):

- **Ancestor test** replaced by a slope test of `X_j` on the clamped value `v` within
  `do(X_i)` rows (BH at 0.01, small-`n` guards). The v1 two-sample test had no power under the
  symmetric value cycle in a zero-mean world and fired spuriously across shifts.
- **Episode protocol** pinned as a numbered list (shift → batch → detect → refit →
  interventions in two rounds → structure → answer), with warm-up episodes 1–3 and the first
  shift at episode ≥ 7. P2, P3 and recovery now have one meaning.
- **Recovery time** redefined: shift schedule pre-drawn (8 shifts, min gap 5, `T = 60`),
  magnitude filter `m ≥ 0.05`, threshold `max(1.5 L, L + 0.02)`, right-censoring with
  Kaplan–Meier, plus post-shift regret against `oracle-structure` as the threshold-free
  companion.
- **Identical data streams** made real with keyed RNG streams and common random numbers for
  interventional noise; a test enforces it.
- **Baselines** made answerable: `obs-*` are per-pair OLS on `X_i` (see-for-do);
  `int-monolithic` replaced by `int-pairwise`, the consistent structure-free interventional
  regressor, so the factorization claim is about sample efficiency and is falsifiable.
- **Requirement 5 ablation** is now `mech-reset-all`; `mech-no-detect` cannot produce an
  overwrite bump and is demoted to a secondary adaptation-speed ablation. P4 rewritten.
- **Requirement 4** measured directly (Metric 8) and made non-trivial by the heteroscedastic
  and shift-type knobs; `mech-oracle-detect` added to decompose detection vs structure vs
  estimation. P7 added.
- **Requirement 6** exercised minimally: self log-score vs variance objective,
  `mech-overconfident` probe, P8 added; Purpose now says exactly what is and is not tested.
- **Detector** defined exactly (NIG posterior predictive vs batch ML fit, Page–Hinkley with
  drift `p + 2`, `N_min` guard, zero on parent-set change) instead of being named twice.
- **P6** re-derived: interventional ancestor tests are immune to latent confounding, so the
  failure is omitted-variable bias on a confounded existing edge; the hidden generator now
  guarantees the `H → A → B, H → B` motif; three separable claims plus a diagnostic and a
  scheduled hidden shift.

Major issues:

- Variable labels randomly permuted and tie-breaks randomized (topological-order leak).
- Order extraction robust to cyclic / non-transitive ancestor estimates; learned `W` always
  invertible; parent selection is OLS with t-tests on the mechanism's own buffer.
- Active rule defined as normalized answer-variance over posterior draws, with a round-robin
  floor shared by all interventional agents and a `nofloor` ablation; the "half the budget on
  parents" clause dropped.
- Every agent gets tuned constants from a stated validation set and objective; `τ` set per
  agent to a 10 % abstention rate; `constants.json` hashed into results.
- Metrics normalized by `Var_ref(X_j)`, averaged over all `j ≠ i` with a descendant split and
  a `marginal-mean` floor; selective risk at fixed coverages with an absolute mistake
  threshold; calibration on all predictions with widths and structural strata; six quantiles
  returned so 50/99 % coverage is computable.
- Analysis plan: one per-seed scalar per prediction, paired percentile bootstrap, Holm across
  primaries, cluster bootstrap for event-pooled statistics, everything else exploratory.
- P1 made quantitative against the population floor `F_obs`; reverse-causal case included.
- Exact interventional mean, `W` orientation, intercepts, NIG hyperparameters and Gamma
  convention, vectorized propagation, results layout, CLI, and a concrete test list written
  out.

Minor issues incorporated: `p_edge = 3/(d − 1)` and sign-magnitude weights; explicit
stable/shift/post-shift definitions; budget-sweep and hidden-variant scope; small-`n` guards
at `B = 10`; compute table for the enlarged design.

Rejected or altered (with reason):

- Sub-episode (every-50-samples) recovery checkpoints: conflicts with the batch episode
  protocol and multiplies evaluation cost in post-shift episodes by ≈ 4; post-shift regret
  covers the same question at episode granularity.
- `int-monolithic` as a Gaussian conditional on pooled observational + interventional rows:
  its estimand mixes regimes and it is neither the honest non-factored competitor
  (`int-pairwise` is) nor a clean "see" baseline; may be added as exploratory.
- 40 % uniform exploration fraction and 80/20 active/random split: replaced by the
  deterministic round-robin floor `ceil(B / 2d)`, which is identical across agents and keeps
  the paired design; the `nofloor` ablation tests the same question.
- Ridge `α = 1` on standardized regressors for all fits: unnecessary since every baseline fit
  is a 2-parameter OLS with `n ≥ 4`, and the NIG prior `V0` already regularizes the mech fits.
- Structure pool "since episode 1": replaced by a 10-episode pool because a `large` shift can
  flip a weight's sign, which a pooled slope test would average toward zero.
- Shift-conditional metrics stratified by `δ` median: `m` (query-distribution nMSE change) is
  used as the primary magnitude because it is on the metric's own scale; `δ` is logged.
