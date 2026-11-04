> Adversarial review of the design, written by a planning subagent before any code ran. Most points were adopted; see SPEC.md.

## Adversarial critique of the causal-JEPA pilot design (most important first)

**P0: problems that can make the headline result meaningless**

1. **Soft gate scale is not identifiable under softmax.** In `logits + log σ(A)`, softmax normalizes each row. Lowering every entry of a row by the same amount changes nothing, so the L1 penalty on σ(A) can push all gates down together at no cost to the loss. A then looks "sparse" without masking anything. The same problem hits the action column. Fixes, pick one:
   - Pin the diagonal: self-loop always on with bias 0 and excluded from the penalty. Off-diagonal gates are then relative to self.
   - Or gate the values multiplicatively (sigmoid gate on messages, no renormalization).

   You are right about the literal `logits*A` version. A=0 gives logit 0, which means uniform attention, not masking. Run it once as a negative demo and don't spend seeds on it.

2. **An end-to-end encoder can route around the mask (pixel regime).** If the encoder trains jointly with a masked predictor, it can copy parent information into the child's token. A can then be sparse while the model still uses every object, so "A recovers the graph" and "A constrains computation" come apart. Splitting a flat embedding into K chunks makes this almost certain, and SIGReg's isotropy does nothing to tie chunks to objects. Fixes:
   - Two-stage protocol for predictor comparisons: train one shared encoder, freeze it, then train all predictor variants on the same frozen latents. This also removes most of the latent-geometry confound.
   - Report end-to-end results separately.
   - In the pixel regime, if Hungarian-matched probe R² stays below a pre-set bar (say 0.8 for each object), report graph recovery as **undefined, not negative**.
   - Realistically, on CPU, run pixel as a single smoke test and put the seeds into the state regime.

3. **Stacked masked layers.** With L layers the effective reach is roughly A^L, and the residual stream adds hidden self-loops. A layer-1 gate of 0→2 can be learned as 0 while 0 still reaches 2 through 1. Use L=1 (with a per-token MLP) for any graph-recovery claim. If you use L=2, share A across layers and score recovery against the true graph, not against A².

4. **Is the edge signal strong enough to see in one step?** Edges act through velocity: v_i(t+1) depends on p_j(t). At the object level (one token holds the full position and velocity), one step does reveal j→i, so keep the graph at object granularity, not positions-only. The real risk is size: with small dt and heavy damping, the parent term (k·Δp·dt) is tiny next to the inertial self-term. MSE ignores it and the sparsity penalty prunes true edges. Fixes:
   - Pre-check with the simulator: for each true edge, how much of the child's one-step Δ-variance is lost when the parent is removed? Tune dt and stiffness until each edge explains at least about 10%.
   - Make sure parents are actually excited, using a persistently exciting random action policy.
   - Avoid object-object collisions, or the true graph becomes dense and state-dependent.

**P1: confounds and metric pitfalls**

5. **Comparable prediction error.** Raw latent MSE is not comparable across latent spaces. In order of preference:
   - (a) Frozen shared encoder (point 2), so all variants share one latent space.
   - (b) Decode predictions to true state with a probe fit on the encoder's own latents, and report state-space error.
   - (c) Normalized MSE, error / Var(z_{t+1}) per dimension, plus the ratio against a copy baseline ‖z_{t+1} − z_t‖².
   - For OOD, also normalize by the copy baseline computed on the OOD data, because a 3× faster distractor raises everyone's error trivially.

6. **"Dense uses the distractor" is assumed, not tested.** On Markov, fully observed data, a converged dense model has no reason to use the distractor. Make the claim testable:
   - Measure dependence directly: Jacobian norm ‖∂ẑ_i/∂z_distractor‖ and ‖∂ẑ_2/∂z_0‖ for every variant.
   - Build in a real spurious correlation: in training, couple the distractor's random walk to object 1 (shared noise or a lag). Break the coupling at test time.
   - Cleaner still is the grandparent shortcut. Along 0→1→2, z_0 is collinear with z_1, so dense attention may predict object 2 partly from object 0. The do(1) clamp exposes this: after clamping, object 2 must not respond to object 0.
   - Keep distractor ×3 as is. It tests input shift on keys and queries, which masking blocks mechanically.

7. **Defining the intervention.** Clamping position also forces velocity to 0, which no training state has. Specify that the clamped object is re-encoded (teacher-forced) at every rollout step. Score descendants (object 2 should follow the clamped 1) and non-descendants (0, 3, distractor) separately.

8. **Stiffness ×1.5 is a mechanism change on an existing edge.** Masking cannot help here; expect every variant to degrade by the same amount. Label it a negative control, not a test of the hypothesis.

9. **Thresholding.**
   - Use AUROC and AUPRC (threshold-free) as the primary metrics, off-diagonal only.
   - Exclude self-loops and fix action→0 (or report the action column separately).
   - SHD: σ > 0.5 for the hard gate; for the soft gate, either a threshold fixed in advance or top-k with k = the true edge count. Label top-k "oracle-k".
   - Choose λ on validation prediction error, never on graph recovery (that would leak the answer).

10. **Controls.**
    - The random mask must keep self-loops and include several draws. Report how often a draw cuts the action→2 path; those draws fail planning trivially.
    - Add "dense + attention-entropy penalty" so you can separate the effect of sparsity from the effect of the correct structure.
    - Match parameter count and give dense the same tuning budget (weight-decay sweep).

**P2: planning and statistics**

11. **Planning will be dominated by things other than predictor structure.** Likely culprits:
    - Latent quality, CEM settings (samples, iterations, horizon), and lag: the 0→1→2 spring chain needs a long horizon.
    - The goal cost. A full-latent distance to a goal observation also penalizes objects 0, 1 and 3, so the task is ill-posed. Score only object 2's factor, or a probe-decoded object-2 position, and use the same cost for every variant.

    Fixes:
    - Add bracketing baselines: CEM with the true simulator (upper bound) and random actions (lower bound).
    - Use identical start/goal episode sets for every variant, with at least 50 episodes per seed.

12. **Seeds.** 3 seeds only detects large effects. Use paired designs (same data, same frozen encoder, same eval episodes, same seeds for each variant) and report per-seed differences. Run 5 seeds in the state regime, where it is cheap. Bootstrapping over episodes is not a substitute for seed variance. Write down the minimum effect size you care about before running.

13. **Compute triage.** Pixel × 6 variants × 3 seeds × CEM will not fit in a few CPU hours. Run in this order:
    1. Simulator identifiability check.
    2. State regime, frozen-free (per-object encoders are already modular), L=1, five variants (drop the literal version).
    3. Do(1) and distractor tests.
    4. Planning.
    5. A single pixel run.

**Stop-gradient question:** for the state regime it hardly matters. In the pixel regime, use the frozen two-stage protocol for comparisons. If you do run end-to-end, use LeWM's no-stop-gradient setup with SIGReg. Adding EMA confounds the comparison with an extra variable.

## Predictions I would bet on (state regime, L=1, diagonal pinned)

- **Oracle mask vs dense, in-distribution:** essentially tied (within noise).
- **Oracle mask vs dense, out of distribution:** clearly better on distractor ×3 and on the do(1) non-descendant / grandparent-shortcut test. High confidence.
- **Anchored soft or hard gate, graph recovery:** AUROC above 0.85 once the edge-signal check passes. Moderate-high confidence.
- **Unanchored soft gate:** an A that looks sparse but is uninformative. High confidence.
- **Literal `logits*A`:** behaves like dense. High confidence.
- **Stiffness ×1.5:** no difference between variants. High confidence.
- **Planning:**
  - In-distribution: no significant dense vs learned-mask difference with 3–5 seeds. High confidence.
  - Random-mask draws that cut the chain fail planning.
  - Any planning gain shows up only under distractor ×3.
- **Pixel regime:** factor-object alignment fails or is partial with chunk splitting on CPU, and graph recovery is near chance. I would not bet on the pixel regime improving planning or robustness.

### Critical files for implementation

None yet: the design hasn't been built and the working directory (`/home/user/Claude-Learning/experiments/mechanism-shift`) holds no relevant code.