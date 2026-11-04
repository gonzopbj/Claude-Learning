# Failure Notebook

Every time a method or a model surprises you by breaking, it goes here. Both tracks.

By month twelve this file is worth more than your notes on the methods themselves. Knowing the conditions under which something fails is what separates someone who can use a tool from someone who can be trusted with it.

Keep entries short. The value is in having many, not polished ones.

---

<!-- Causal template:

## YYYY-MM-DD — [what broke]
**Setup:** DGP, sample size, estimand
**Expected:** what I predicted
**Observed:** what happened
**Why:** the actual mechanism
**Generalizes to:** where else this bites

-->

<!-- ML template:

## YYYY-MM-DD — [what broke]
**Setup:** architecture, data, hyperparameters
**Symptom:** what the loss curve / gradients looked like
**Cause:** the actual mechanism
**Signature:** how I'd recognize this failure next time from the curve alone

-->

*The five entries below come from `experiments/mechanism-shift/`, a demonstration run by
the tutor ahead of the curriculum. Numbers are in `results/report/REPORT.md` there.*

## 2026-09-21 — Calibrated parameter uncertainty is confidently wrong about structure
**Setup:** Bayesian SCM (`mech-full`) with learned DAG, 6 variables, linear-Gaussian, 20 seeds,
posterior draws propagated through the learned graph; 90 % predictive intervals.
**Expected:** coverage near 0.90 in stable episodes.
**Observed:** 0.91 when the learned graph was exactly right; 0.28 on queries whose directed
paths in the learned graph differed from the truth. With a hidden confounder on one edge,
coverage on that pair was 0.11 and the abstention rate barely moved.
**Why:** the Monte Carlo over posterior weights contains no term for the graph being wrong.
Uncertainty over the wrong model class is not uncertainty about the answer.
**Generalizes to:** every Bayesian or ensemble method whose model class excludes the truth;
"calibrated" always means "conditional on the model being right".

## 2026-09-21 — Overwriting all memory cost nothing, because relearning was cheap
**Setup:** `mech-reset-all` (every mechanism reset whenever any detector fires) versus
`mech-full` (only the flagged mechanism reset). Pre-registered forgetting bump ≥ 0.04.
**Expected:** a visible error bump on queries unrelated to the shifted mechanism.
**Observed:** bump 0.000 for both; identical post-shift regret. The cost showed up only in
interval width (0.12 vs 0.07) and abstention (12 % vs 9.5 %).
**Why:** 250 rows arrive every episode and each mechanism has at most six parameters, so a
fully reset model is back to precision within the same episode. The memory requirement only
bites when relearning is expensive relative to the data rate.
**Generalizes to:** continual-learning benchmarks that are too data-rich to show forgetting;
check the relearning cost before concluding a method "does not forget".

## 2026-09-21 — Variance-based active intervention bought nothing over random targets
**Setup:** `mech-full` picks the intervention target with the highest posterior answer
variance; `mech-random` picks uniformly. Both share a round-robin exploration floor.
**Expected:** active beats random at budgets 10 and 25.
**Observed:** paired differences of 0.004, −0.004, −0.002, −0.001 across budgets 10 to 100,
all with confidence intervals containing zero. Removing the exploration floor, on the other
hand, tripled the error and doubled the structural Hamming distance.
**Why:** in a six-variable linear world the round-robin floor already covers every target
every episode; the marginal information from concentrating the free budget is small, and the
score ranks by variance of a wrong graph's answers when the graph is wrong.
**Generalizes to:** active learning claims that never compare against random with the same
exploration floor.

## 2026-09-21 — The "gameable objective" probe did not bite
**Setup:** `mech-overconfident` uses a tight prior (V0 = 0.1 I, a0 = b0 = 50) to fake a
variance-reduction objective; predicted to win on posterior-variance reduction and lose on
self log-score, coverage and selective risk.
**Expected:** a 20-point coverage gap and a positive log-score difference.
**Observed:** coverage 0.824 vs 0.819, log-score difference 0.012 with a CI through zero.
**Why:** the prior is swamped after one episode of 250 rows. A manipulation that the data
erase in one step cannot demonstrate anything about objectives.
**Generalizes to:** any test of reward hacking where the hack is weaker than the signal;
design the probe to persist, for instance by shrinking the likelihood rather than the prior.

## 2026-09-21 — Residual-based shift detector: perfect under homoscedastic noise, 10 % false alarms under heteroscedastic noise
**Setup:** Page–Hinkley on per-mechanism generalized likelihood ratios, threshold tuned to a
1 % false-reset rate on validation seeds with constant noise scale.
**Expected:** attribution accuracy to drop by 15 points when noise scale depends on parents.
**Observed:** attribution accuracy stayed at 0.95, but false resets rose from 1.3 % to 10 %
per mechanism-episode and attribution precision fell from 0.44 to 0.17. The oracle-structure
agent suffered the same, so the failure is the detector, not the graph.
**Why:** the detector's null model assumes a constant residual scale given the parents. An
upstream shift changes the downstream residual scale without changing the downstream
mechanism, and the detector cannot tell the two apart.
**Generalizes to:** change-point detection whose null model is narrower than the world;
misattribution shows up as false alarms before it shows up as misses.

## 2026-11-04 — A soft attention gate looked empty and constrained nothing
**Setup:** JEPA predictor (`experiments/jepa-causal-predictor`), attention p_ij ∝ exp(l_ij)·σ(α_ij)
with an L1 penalty on σ(α), state regime with a known 3-edge graph.
**Expected:** σ(α) above 0.5 on true edges, near 0 elsewhere.
**Observed:** every off-diagonal σ(α) was between 0.07 and 0.17, so 0 edges above threshold. Yet
the ranking was perfect (AUROC 0.996), and the Jacobian showed full-strength true edges.
**Why:** exp(l)·σ(α) = exp(l + log σ(α)). The gate is an additive logit bias that the learned
logit cancels. The penalty shrinks σ(α) at no cost to the loss.
**Generalizes to:** any finite multiplicative gate inside a softmax. Only hard (sampled or
thresholded) masks constrain. Read soft gates as rankings, never as graphs.

## 2026-11-04 — A hard Gumbel gate collapsed to the empty graph, including the action edge
**Setup:** same predictor, straight-through Gumbel-sigmoid mask, penalty λ·mean σ(α).
**Expected:** a monotone λ trade-off between edges kept and prediction loss.
**Observed:** at λ ≥ 0.02 (state) and λ = 0.005 (pixels) the gate pruned *every* edge,
including action → object 0. Validation loss doubled, and pixel planning became independent of
the model: two seeds gave byte-identical trajectories. At λ = 0.005 in the state regime it found
the exact graph in 5/5 seeds.
**Why:** an edge that is weak early in training, before the encoder or predictor carries the
signal, is pruned. A pruned edge gets no gradient through the mask, so it never comes back.
**Generalizes to:** any learned hard mask trained jointly with its inputs. Warm-start dense or
anneal λ, and always check that the edges you need (here, the action) survive.

## 2026-11-04 — End-to-end pixel JEPA encoded the distractor and dropped the controllable objects
**Setup:** CNN on two stacked 32×32 frames, latent MSE + SIGReg, no stop-gradient (LeWM-style).
**Expected:** a weak but object-related representation; chunks not aligned (pre-registered).
**Observed:** the slow distractor's position was decoded at R² 0.98. Fast action-driven objects
were at 0.45–0.6, velocities at 0.04, and an MLP probe did no better. Prediction was no better
than copying the last frame, and planning was at the random floor. An inverse-dynamics loss
recovered only the directly pushed object (0.57 → 0.92).
**Why:** the JEPA loss rewards predictable features. The slowest thing on screen is the most
predictable, and the thing you control is the least predictable without the action.
**Generalizes to:** "the model ignores the distractor" is not a free property of
joint-embedding world models. Check what the encoder kept before evaluating anything
downstream of it.
