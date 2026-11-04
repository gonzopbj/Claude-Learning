# Causally structured JEPA predictor: pilot specification

Written before the main runs (2026-11-04). It was revised after an adversarial design critique
(`docs/design_critique.md`) and a literature check (`docs/literature_review.md`). The
predictions below were fixed before any result except one smoke run.

## Question

In a JEPA world model, does a learnable sparse K×K adjacency that gates attention between latent
factor tokens:

1. recover the true interaction graph,
2. improve robustness under distribution shift, and
3. improve planning,

compared with dense attention? And does this survive when the factorization has to be learned
from pixels?

## Why this is a pilot, not the study

- It runs on CPU only (4 cores) and needs a few hours of total compute.
- The model is tiny (d = 16 per factor, one attention layer).
- The point is to find out cheaply whether the idea has a signal worth a GPU-scale study, and
  where it breaks.

## World (`env.py`)

- **Objects.** Five point masses in a box with reflecting walls.
- **Graph.**

  | Edge | Mechanism |
  |---|---|
  | action → 0 | push |
  | 0 → 1 | spring follower |
  | 1 → 2 | spring follower |
  | 3 → 2 | spring |

  Object 3 is an autonomous noisy oscillator. Object 4 is a distractor: an Ornstein–Uhlenbeck
  random walk with no edges.
- **Identifiability pre-check (`check_edges.py`).** Each true edge explains 5–11 % of the
  child's one-step Δ-variance in a linear fit. The weakest is 3 → 2 at about 5 %. Non-edges
  explain less than 0.2 %.
- **No collisions between objects,** so the graph is global and not state-dependent.

## Model (`model.py`)

- **Shape.** A LeWorldModel-style JEPA: encoder → K = 5 factor tokens of d = 16 → one-layer
  transformer predictor over 5 factor tokens + 1 action token → next latent.
- **Loss.** Multi-step (3-step) latent rollout MSE + 0.09 · SIGReg. There is no stop-gradient
  and no EMA, as in LeWM. SIGReg is exactly LeWM's formula (17 knots on [0, 3], Gaussian window).
- **Compute deviations from LeWM's SIGReg:**
  - 64 or 256 slices instead of 1024.
  - One random time step per batch instead of the mean over all steps.
- **One attention layer** so that the gate is the one-step dependency graph. There is no A^L
  leakage.
- **Self-loops are pinned to 1 and never penalized.** Without this, the softmax renormalization
  would make the gate's scale unidentifiable.

### Predictor variants

| Variant | What it is |
|---|---|
| `dense` | Baseline. |
| `dense_ent` | Dense plus an attention-entropy penalty. Sparsity without structure. |
| `soft` | p_ij ∝ exp(l_ij) · σ(α_ij), with L1 on σ(α). |
| `hard` | Straight-through Gumbel-sigmoid mask, with penalty on σ(α). Evaluated at σ > 0.5. |
| `mult` | The literal proposal: softmax(l_ij · σ(α_ij)). Predicted to not mask: a gate of 0 gives logit 0, which is uniform weight, not zero weight. |
| `oracle` | The true graph, fixed. Upper bound. |
| `random` | A random graph with the same edge count, fixed and seeded. Control. |

### Regimes

- **State regime.** A per-object encoder, so factor k sees only object k and alignment is given.
  This is the main experiment: 5 seeds, every variant, paired (same data, same evaluation
  episodes).
- **Pixel regime.** Two stacked 32×32 RGB frames go into a CNN, and the flat embedding is split
  into 5 chunks. Nothing ties a chunk to an object. It is run two ways:
  - **End-to-end per variant** (dense, hard, oracle). This tests whether sparsity of the
    transition graph makes chunks align with objects. SIGReg alone is rotation-invariant, and
    mechanism-sparsity identifiability results (Lachapelle et al.) suggest a sparse graph could
    break that symmetry. This is the most novel question here.
  - **Frozen encoder.** One dense end-to-end encoder is trained and frozen, then every predictor
    variant is trained on the same latents. This makes the variants comparable in a single latent
    space.

## Metrics (`evaluate.py`)

- **Alignment.**
  - The R² matrix of each factor (alone) → each object's position, with Hungarian matching.
  - Rule fixed in advance: if any matched R² is below 0.8, graph recovery is reported as
    *undefined*, not as a negative result.
- **Prediction error.**
  - Primary: decoded-latent error, ‖P(ẑ_{t+h}) − P(z_{t+h})‖ in position units. P is one fixed
    ridge probe per run, fit on that run's training latents.
  - It is normalized by the copy baseline ‖P(z_t) − P(z_{t+h})‖, at h = 1 and h = 5.
  - Objects included: 0–3 (the distractor is excluded). For do(1): objects 0, 2 and 3.
- **Shifts.**

  | Shift | What changes | Role |
  |---|---|---|
  | `physics` | Stiffer springs, less damping, heavier masses | Mechanism change on existing edges. Negative control: masking cannot help. |
  | `distractor` | Distractor 3× faster | Input shift on a node with no children. Tests reliance on the distractor. |
  | `do1` | Object 1 clamped at a random point | Hard intervention. Object 1 is re-encoded every step during rollouts. Tests modularity and the 0 → 2 grandparent shortcut. |

- **Graph recovery.** Off-diagonal object → object entries (20 entries, 3 true):
  - Threshold-free: AUROC and AUPRC.
  - SHD at a fixed σ > 0.5 for gates.
  - SHD with top-3 ("oracle-k") for model-agnostic scores: the Jacobian ‖∂ẑ_i/∂z_j‖ normalized by
    the self term, and mean attention weights.
  - The action column is reported separately.
- **Planning.**
  - The task: bring object 2 to within 0.1 of a goal position. Object 2 is reachable only
    through the chain action → 0 → 1 → 2.
  - Goals come from a 30-step random rollout, so they are reachable.
  - The planner is CEM (128 samples, 4 iterations, horizon 15) with replanning every 5 steps
    over 75 steps.
  - The cost is the probe-decoded position of object 2, the same for every variant.
  - Brackets: CEM on the true simulator (oracle), about 0.68 success; random actions, about
    0.09.
  - Planning is run in-distribution and under the distractor shift, with 40 episodes each.
- **λ_A selection** is done on validation prediction loss only, never on graph recovery. The
  rule, fixed in advance: take the largest λ_A in {0.005, 0.02, 0.1} whose validation prediction
  loss is within 5 % of the best. Selection uses seed 100, which is not one of the evaluation
  seeds.
- **Smallest effect worth caring about:** 10 % relative prediction error, or 10 points of
  planning success.

## Pre-registered predictions

| # | Prediction |
|---|---|
| P1 | In-distribution, h = 1: `dense`, `soft`, `hard` and `oracle` are within 10 % of each other. `random` is more than 20 % worse. |
| P2 | `soft` and `hard` gates recover the graph: AUROC ≥ 0.9 and SHD (σ > 0.5) ≤ 1 in at least 4 of 5 seeds. |
| P3 | The `mult` gate does not mask. Its Jacobian dependence on non-parents stays near dense's, whatever its gate values look like. |
| P4 | The dense model's own Jacobian already recovers the graph (AUROC ≥ 0.9), so in the state regime the gate is not needed to *read off* the graph. Mean dense attention weights recover it worse than the Jacobian (this bears on the open question in `PROGRESS.md` about attention weights as causal edges). |
| P5 | Distractor shift: dense's non-distractor error rises more than 20 % relative to in-distribution; `oracle`, `soft` and `hard` rise less than 5 %. |
| P6 | do(1): `oracle`, `soft` and `hard` degrade less than `dense` on objects 0, 2 and 3. |
| P7 | Physics shift (negative control): `dense` and `oracle` degrade by the same amount, within 10 %. |
| P8 | Planning in-distribution: no dense-vs-gated difference larger than 10 points. Under the distractor shift, gated beats dense by more than 10 points. |
| P9 | `random` masks that cut the action → 0 → 1 → 2 chain fail at planning (near the random-action floor). |
| P10 | Pixel, end-to-end: dense chunks do not align (min matched R² < 0.8). |
| P11 | Pixel, end-to-end: the oracle mask raises mean matched R² by more than 0.1 over dense (sparsity breaks the rotation symmetry). This is the genuinely uncertain one, roughly a coin flip. |
| P12 | Pixel: the learned hard gate's graph recovery is undefined or near chance, unless P11 holds. |
