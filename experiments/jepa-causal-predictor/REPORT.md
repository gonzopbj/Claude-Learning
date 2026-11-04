# Report: a causally structured JEPA predictor (Idea 2), CPU pilot

**Date:** 2026-11-04

**Status:** The pilot is complete. Every number below comes from `results/summary.md`, which
`aggregate.py` regenerates from `runs/*/eval.json`. The design, metrics and predictions P1–P12
were fixed in `SPEC.md` before the main grid ran. The low-data arm (P13) was added after the main
results and is labelled exploratory.

---

## Bottom line

**Don't pursue Idea 2 as stated.** That is, "a learnable sparse adjacency in the JEPA predictor
improves planning and robustness". There are two reasons:

1. **When the factors are given (state regime), the gate is redundant.**
   - With fully observed Markov state, the best one-step predictor depends only on the true
     parents. A dense one-layer transformer finds this by itself. Its Jacobian on non-edges is
     about 1 % of the self term, and both its Jacobian and its raw attention weights recover the
     graph perfectly (AUROC 1.00, 5/5 seeds).
   - So the gate has nothing to fix. In-distribution error, all three shifts, and planning were
     all within noise of dense. The one exception was the hard gate, which was *worse* at
     planning: −9.5 points, 95 % CI [−16, −3].
   - The hard gate does give a crisp, correct graph readout: SHD 0, the action edge correct, 5/5
     seeds. But you could already read that graph off the dense model.
2. **When the factors must be learned (pixel regime), the predictor isn't the bottleneck. The
   representation is.**
   - End-to-end JEPA training at this scale learned slow features. It encoded the slow distractor
     almost perfectly (R² 0.98). It mostly dropped the fast, controllable objects, and velocities
     were absent (R² ≈ 0.04).
   - Prediction was no better than copying the last frame, and planning was at the random floor.
   - Neither the oracle mask nor an inverse-dynamics loss made the latent chunks align with
     objects (min matched R² ≈ 0.2 against a pre-set bar of 0.8). The learned hard gate collapsed
     to an empty graph and cut the action edge.

The literature check (`docs/literature_review.md`) had already put the predictor at about 70–80 %
done by **SPARTAN** (NeurIPS 2025), with S2-SSM and Causal-JEPA nearby. A planning-benchmark
version would read as incremental.

**What *might* be worth researching:**

- The identifiability question underneath the idea: whether mechanism sparsity plus SIGReg makes
  learned factors align with objects.
- Settings where data cannot determine the parents: few-shot, partially observed, confounded.

This pilot gives no positive evidence for the identifiability question. It does show the
precondition you would need to solve first: an encoder that keeps the controllable, fast
objects. Details are in §6.

---

## 1. What was built

The code is in this directory, about 1,500 lines of Python, CPU only.

**Environment (`env.py`).**
- Five point masses in a box: action → 0 → 1 → 2 ← 3. Object 3 is a noisy anchored oscillator
  and object 4 is a no-edge Ornstein–Uhlenbeck distractor.
- `check_edges.py` confirmed that each true edge explains 5–11 % of the child's Δ-variance, and
  non-edges less than 0.2 %.
- Observations are either object states or two stacked 32×32 RGB frames.

**Model (`model.py`).**
- A LeWorldModel-style JEPA:
  - encoder → 5 factor tokens (d = 16) + action token,
  - a one-layer transformer predictor,
  - loss = multi-step latent MSE + 0.09·SIGReg (LeWM's exact Epps–Pulley form), with no
    stop-gradient and no EMA.
- Self-loops are pinned to 1.
- Predictor variants:

  | Variant | Description |
  |---|---|
  | `dense` | Baseline |
  | `dense_ent` | Attention-entropy penalty |
  | `soft` | Multiplicative gate inside the exponent, renormalised |
  | `hard` | Straight-through Gumbel-sigmoid mask |
  | `mult` | The literal `softmax(l·σ(A))` |
  | `oracle` | The true graph |
  | `random` | Random graph with the same edge count |

**Evaluation (`evaluate.py`).**
- Factor→object alignment: per-factor R², with Hungarian matching.
- Prediction error: probe-decoded and normalised by the copy baseline, at h = 1 and h = 5.
- Three shifts:
  - physics: a negative control,
  - distractor 3× faster,
  - do(object 1 := random point).
- Graph recovery: AUROC, AUPRC and SHD for the gate, the Jacobian and the attention weights.
- CEM planning: bring object 2, reachable only through the chain, to a goal.
  - On the true simulator, CEM reaches 0.68 success.
  - Random actions reach 0.09.

**Scale.**
- State regime: 7 variants, 3–5 seeds, paired.
- Pixel regime: dense, hard and oracle × 2 seeds, plus 2 inverse-dynamics runs.
- λ_A was selected on validation *prediction* loss with a held-out seed, never on graph recovery.
- In total: 45 main runs (including the 6 λ-sweep runs) + 9 low-data runs. Training took about
  7.7 CPU-hours for the 45 runs; with evaluation and planning, about 15 CPU-hours.

## 2. Pre-registered predictions: verdicts

State regime: n = 5 seeds (3 for `dense_ent`/`mult`). "Paired Δ" is the per-seed difference
from dense, with a t-based 95 % CI.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | ID h=1 error: dense/soft/hard/oracle within 10 %; random >20 % worse | 0.341 / 0.328 / 0.358 / 0.333 (× copy baseline); random 0.447 (+31 %); at h=5 random is 0.527 vs 0.200 | **Holds** |
| P2 | soft and hard gates recover the graph: AUROC ≥ 0.9 and SHD ≤ 1 in ≥ 4/5 seeds | hard: AUROC 1.00, SHD 0, exactly 3 edges, action edge correct, 5/5. soft: AUROC 0.996 but **0 edges** above 0.5, SHD 3 | **Half**: hard yes, soft no |
| P3 | `mult` doesn't mask: non-edge Jacobian stays near dense's | mult 0.005 vs dense 0.010. But dense's non-edge dependence is already ≈ 0, so this can't discriminate. mult's gate *looks* right (SHD 0.67) | **Untestable here** |
| P4 | Dense Jacobian recovers the graph (AUROC ≥ 0.9); attention weights do worse | Jacobian AUROC 1.00, SHD 0. **Attention AUROC is also 1.00** | **Half**: Jacobian yes; attention isn't worse here |
| P5 | Distractor shift: dense error +20 %, gated < 5 % | dense +4 %, oracle +6 %, hard +4 %, soft +10 % | **Fails**: dense isn't hurt, because it already ignores the distractor |
| P6 | do(1): gated degrades less than dense | dense ×0.997, oracle ×0.974 (paired Δ −0.023, CI [−0.034, −0.011]), soft ×0.992, hard ×1.009 | **Fails**: oracle is statistically better but 2 %, below the pre-set 10 % bar |
| P7 | Physics shift: dense ≈ oracle within 10 % (negative control) | ×1.155 vs ×1.178 | **Holds** |
| P8 | Planning: no ID difference > 10 pts; gated beats dense by > 10 pts under the distractor shift | ID: soft +1.5, oracle −0.5, mult +5.8, hard −9.5 [−16, −3]. Distractor shift: none better; hard −9 [−16, −2] | **ID half holds, shift half fails** |
| P9 | Random masks that cut the chain fail at planning | All 5 random masks cut it; success 0.10 ± 0.08 vs random-action floor 0.09 | **Holds** |
| P10 | Pixel: dense chunks don't align (min R² < 0.8) | min matched R² 0.23 (both seeds) | **Holds** |
| P11 | Pixel: the oracle mask raises mean matched R² by > 0.1 | oracle 0.33 vs dense 0.36. With inverse dynamics: 0.47 vs 0.49 | **Fails** |
| P12 | Pixel: the hard gate's graph recovery is undefined or at chance | Undefined (alignment < 0.8), and the gate collapsed to 0 edges | **Holds** (trivially) |

The predictions about *mechanics* held: P1, P7, P9 and P10. The ones encoding the *hope* failed:
P5, P6, P8-shift and P11.

## 3. What happened, and why

### 3.1 The dense model is already sparse (the key fact)

`figures/graph_auroc_state.png` and the graph table show the following.

- The dense predictor's normalised Jacobian on all 17 non-edges averages 0.010. The grandparent
  path 0→2 is 0.010 and the distractor→2 path is 0.009.
- This isn't luck. With the full state observed, x_{t+1} ⫫ non-parents | parents, so the loss
  minimiser is parent-only, and 50,000 transitions are enough to find it.
- A sparsity prior is a prior: it helps exactly where the likelihood is flat. **In a fully
  observed, Markov, well-sampled world, nothing is flat, so the prior can't do anything.**

This explains every null:

- **Distractor shift (P5).** Dense puts about 0 weight on the distractor, so changing it changes
  nothing.
- **do(1) (P6).** The grandparent shortcut 0→2 that the gate was supposed to remove was never
  learned.
- **Planning (P8).** The models' dynamics are the same, so their plans are the same.
- **Attention weights read off the graph (P4).** In a single layer with aligned tokens, attention
  mass and influence coincide. This partly answers the open question in `PROGRESS.md` about
  attention weights as causal edges. It's true *in this degenerate case*. The interesting failure
  (multi-layer mixing, value-norm effects) wasn't probed.

The design critique (`docs/design_critique.md`, point 6) predicted this in advance: "on Markov,
fully observed data, a converged dense model has no reason to use the distractor". It proposed
building in a real spurious correlation: couple the distractor to object 1 in training and break
the coupling at test. **I didn't implement that in the main grid,** and it is the most important
missing experiment (§6). The pilot confirms the critique's point.

### 3.2 The soft gate is decorative

- At every λ the soft gate's values looked "empty": off-diagonal σ ≈ 0.07–0.17, so zero edges
  above 0.5. Yet its graph ranking was perfect (AUROC 0.996) and its Jacobian showed full-strength
  true edges.
- **The reason:** p_ij ∝ exp(l_ij)·σ(α_ij) = exp(l_ij + log σ(α_ij)). The attention logit l
  can add back whatever log σ(α) takes away. Any finite soft gate inside a softmax is an additive
  logit bias that the network can cancel. The L1 penalty pushes σ(α) down at no cost to
  prediction, because l compensates.
- So a soft gate constrains nothing, and its values can't be thresholded. Only the ranking means
  anything.
- The literature review predicted this for the multiplicative form and noted that SPARTAN uses
  hard masks for this reason. It applies just as much to the "correct" exp·σ form.
- **`mult`** (the literal proposal, softmax(l·σ(α))) came out similar: AUROC 1.0 and SHD 0.67,
  so its gate is informative. But since dense already uses no non-edges, its masking can't be
  checked here (P3).

### 3.3 The hard gate works, narrowly, and costs performance

- On the λ-selection seed:
  - At λ_A ≥ 0.02 the hard gate **collapsed to the empty graph**. It cut the action edge too,
    and validation loss doubled.
  - At 0.005 it kept 8 edges (5 spurious, all unused per the Jacobian).
- On the five evaluation seeds, at 0.005, it found exactly the true 3 edges plus the action edge
  every time (`figures/gates_state.png`). The working window spans less than one decade of λ.
- It cost:
  - +17 % h=5 error (0.234 vs 0.200),
  - −9.5 planning points.
- The likely cause is the straight-through estimator: training sees stochastic masks and
  evaluation sees a deterministic one, so the predictor learns to hedge against dropped edges.
- **In pixels** at the same λ it collapsed to 0 edges in both seeds and cut the action column. As
  a result, planning became independent of the model: both seeds produced byte-identical
  trajectories at 0.025 success.
- **Lesson:** with a hard gate, an edge that is weak *early in training*, before the encoder
  carries the signal, gets pruned and never comes back. Annealing λ, or warm-starting dense, is
  required. SPARTAN-style methods presumably do this. The proposal didn't mention it.

### 3.4 The pixel regime fails at the encoder, before the predictor matters

Pixel results:

| Metric | Value |
|---|---|
| Linear probe position R², objects 0–3 | 0.45–0.60 |
| Linear probe position R², distractor | 0.98 |
| Velocity R² | ≈ 0.04 |
| MLP probe R² (object 0 / 1 / 2) | 0.39 / 0.20 / −0.02 |
| Decoded prediction error | 1.00 × copy |
| Planning success | 0.03–0.15 (floor 0.09) |

Because the MLP probe is also low, the information is *absent*, not just nonlinearly encoded.

- **Interpretation:** JEPA rewards features that are easy to predict. The slow, smooth
  distractor is the most predictable thing on screen. The fast, action-driven objects are the
  least predictable, so the encoder drops them.
- This is a known slow-feature / predictability bias of joint-embedding world models, and
  it hits hardest exactly where planning needs information.
- **Inverse dynamics** (EB-JEPA's remedy, λ = 1) helped only the object that is directly
  controlled:
  - object 0 position R² went from 0.57 to 0.92, and its velocity R² from 0.04 to 0.53;
  - the oracle-mask run gave it its own chunk (R² 0.88);
  - downstream objects stayed at 0.53–0.69;
  - prediction improved to 0.88× copy;
  - planning stayed at the floor (n = 1 seed).

  That makes sense: the action identifies the object it touches, and nothing forces the encoder
  to represent the followers.
- **The oracle mask didn't induce alignment (P11 failed), with or without inverse dynamics.** So
  even a *perfect* sparse graph, imposed rather than learned, didn't break SIGReg's symmetry
  here. One caveat: the factors it should align to are barely present in the representation at
  all. A real test of "sparsity identifies factors" needs an encoder that carries all the objects
  first. The pilot didn't reach that precondition.
- The frozen-encoder stage was skipped by the pre-set rule. With alignment < 0.8, any graph read
  from the chunks is undefined.

## 4. Low-data arm (exploratory, P13)

LOWDATA_PLACEHOLDER

## 5. Caveats: what this pilot can't tell you

- **Tiny scale.** d = 16, one layer, 32×32 frames, CPU, 4000 steps. The pixel failure could be a
  scale failure: LeWM uses ViT encoders, 224 px and GPU-days. I'd still expect the
  slow-feature bias to persist at scale without a remedy, but this pilot doesn't show that.
- **Global graph, no contacts.** The world was built so that one global graph is correct. This is
  the setting most *favourable* to the idea, which makes the null stronger. But SPARTAN's
  argument is about *local*, contact-dependent graphs, which weren't tested.
- **One layer.** This was chosen so that gate = graph. With several layers, dense attention can
  route information through intermediate tokens, and sparsity may matter more for robustness.
  The same is true with partial observability (one frame of positions, no velocities, so history
  is needed).
- **Few pixel seeds** (2, and 1 for inverse dynamics). The pixel conclusions rest on large,
  consistent effects (R² 0.2 vs a 0.8 bar; dense, hard and oracle all at the floor), not on
  small differences.
- **Deviations from LeWM:**
  - SIGReg with 64 or 256 slices instead of 1024, on one random time step per batch.
  - A CNN rather than a ViT.
  - A latent split into chunks rather than slots.
- **Probes are linear ridge,** with an MLP check on one run.
- **Data-version mismatch, found while writing this report.**
  - The cached train/val/test datasets in `data/` were generated before I lowered object 3's
    drive noise (`drive_3`) from 0.4 to 0.15 to make the planning task feasible.
  - So all models were trained, and prediction-scored, in the 0.4 world. Planning episodes ran
    in the 0.15 world, which is a mild shift: object 3 is less noisy than in training.
  - Every variant faced the identical shift with the identical episodes, so the paired planning
    comparisons stand. Absolute planning numbers carry this extra shift, and the oracle-CEM
    bracket (0.68) was measured in the 0.15 world.
  - The exploratory coupled-world arm generated fresh data at 0.15 and is self-consistent.

## 6. If you want to research something here

In order of how much I'd bet on them:

1. **Data-limited or confounded state worlds: where a structural prior *can* matter.** This is
   cheap, needs no pixels, and fits the M10 invariance convergence in the curriculum. Create the
   conditions where likelihood alone can't find the parents:
   - few episodes (§4);
   - a non-parent correlated with a parent in training and decorrelated at test (an ICP-style
     environment shift);
   - observation noise on a parent, so a correlated non-parent carries real information.

   The interesting result would be a *trade-off*: the gate costs ID accuracy and buys shift
   robustness. That trade-off is exactly the invariance-vs-prediction story. **But be
   realistic:** this is SPARTAN's and CDL's territory, so novelty would come from the analysis,
   not the architecture.
2. **The identifiability question** (sparsity × SIGReg → aligned factors). This is the only
   angle with no clear prior work, and it links to the month-9 ICA material. It needs a
   precondition first: an encoder that represents every object. Options:
   - slot attention, or the Causal-JEPA route with frozen slots (which sidesteps the question),
   - stronger inverse-dynamics or multi-step action-conditioning losses,
   - a richer state space at the pixel level.

   Expect this to be a hard, research-grade project with a real chance of a negative answer. The
   pilot's P11 failure is weak evidence against it, given the confound above.
3. **Not recommended:** benchmark planning gains on PushT / Two-Room. That has the highest
   compute cost and the lowest novelty, and the state-regime result says to expect none
   in-distribution.

**Cheapest next step:** take the existing state-regime code and add the confounded-non-parent
environment from (1). That's about a day of work and an hour of CPU. If dense doesn't pick up the
spurious edge there either, drop the idea.

## 7. Files

| Path | Contents |
|---|---|
| `SPEC.md` | Design and pre-registered predictions (written before the main runs) |
| `docs/literature_review.md` | Novelty check against SPARTAN, S2-SSM, Causal-JEPA, Lachapelle et al. |
| `docs/design_critique.md` | Adversarial review of the design, run before the experiments |
| `results/summary.md` | All tables, regenerated by `python3 aggregate.py` |
| `figures/` | Learned gates, prediction error, graph AUROC, planning, pixel alignment |
| `runs/<name>/` | Per-run config, training log and `eval.json` (weights are git-ignored) |

To reproduce one run:

```bash
./run_job.sh 1 --regime state --variant hard --lam_A 0.005 --seed 0
```

This trains for about 12 minutes on one CPU thread, then evaluates with planning.
