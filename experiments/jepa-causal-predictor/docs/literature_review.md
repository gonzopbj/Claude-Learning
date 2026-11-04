> Produced by a research subagent on 2026-11-04. Paper numbers marked unverified came from search snippets (arxiv was not reachable from the sandbox); repo details were read from source.

## Novelty check: causally structured JEPA predictor (as of 2026-11-04)

**Access notes.** I cloned and read three repos from source: `lucas-maes/le-wm`, `facebookresearch/eb_jepa`, `rbalestr-lab/lejepa`, plus `galilai-group/cjepa`. They are in `/tmp/claude-0/-home-user/bb95c317-b18b-51b9-befa-4f1fa6cada8d/scratchpad/lewm/`. I could not reach arxiv.org, le-wm.github.io, huggingface.co, neurips.cc, mlanthology.org or themoonlight.io: these hosts are outside this session's allowed network domains. So paper tables (LeWM success rates, SPARTAN numbers) come from search snippets, not the PDFs.

### 1. Technical details

**a. SIGReg** (exact code, `le-wm/module.py`, matching `lejepa/MINIMAL.md`)
- Grid: `t = linspace(0, 3, 17)`. Trapezoid weights are `2·dt` with `dt` at the two ends, where `dt = 3/16`. These are multiplied by the window `exp(-t²/2)`. The target is `phi = exp(-t²/2)`, the characteristic function of N(0,1). Only t ≥ 0 is used; this is valid by symmetry.
- Slices: `A ~ randn(D, 1024)`, with columns normalised to unit length. A fresh draw is made every call.
- Statistic: for `x_t = (z @ A)·t`, `err = (mean_N cos(x_t) − phi)² + (mean_N sin(x_t))²`. Then `stat = (err @ weights) · N`, where N is the batch size, averaged over slices and over time steps.
- LeWM applies it per time step to `emb` of shape (T, B, D). That is the projector output, not the predictor output.
- LeWM loss (`train.py`): `pred_loss = MSE(pred_emb, emb[:, 1:])` plus `0.09 · sigreg(emb)`. This uses plain addition; there is no stop-gradient and no EMA.
- The minimal LeJEPA image recipe is different: `λ·sigreg + (1−λ)·inv_loss` with λ = 0.02.

**b. LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv 2603.19312; configs from the repo)
- **Encoder:** ViT-tiny, patch size 14, 224 px input, trained from scratch. It outputs the CLS token (192-d), passed through a projector MLP (192→2048→192, BatchNorm1d, GELU).
- **The whole latent is one 192-d vector per frame.** The predictor's tokens are time steps, not factors. Your K-factor split is therefore a real architectural change, not a re-reading of existing tokens.
- **Predictor:** `ARPredictor`, a 6-layer transformer with 16 heads, dim_head 64, MLP 2048, dropout 0.1, causal attention over time, learned positional embedding. History is 3 frames, and it predicts 1 step ahead.
- **Action conditioning:** AdaLN-zero in every block. The action embedder is a Conv1d plus MLP over `frameskip(5) × action_dim`.
- **Second projector:** a `pred_proj` MLP with BatchNorm sits after the predictor.
- **Training:** AdamW, learning rate 5e-5, weight decay 1e-3, batch size 128, 100 epochs, bf16, gradient clipping 1.0.
- **Planning:** CEM with 300 samples, 30 iterations, top-30 elites, horizon 5 with frameskip 5, and receding horizon 5. The cost is MSE to the goal embedding at the last step only. The eval budget is 50 steps over 50 episodes.
- **Environments:** PushT, TwoRoom, OGBench Cube, DMC Reacher.
- **Success numbers (unverified, from search snippets only):** about 96% on PushT versus about 92% for DINO-WM. The snippets also say planning is about 48× faster. Check these against the paper.

**c. EB-JEPA** (`examples/ac_video_jepa`, arXiv 2602.03604)
- Environment: Two Rooms from PLDM (a dot moving through a door in a wall), in fixed-wall and random-wall variants.
- Model: Impala CNN encoder producing one global vector per image, and a 2-layer GRU predictor that takes the raw 2D action.
- **It does not use SIGReg.** It uses VICReg variance and covariance terms, time-similarity and an inverse-dynamics (IDM) loss, with coefficients (8, 16, 12, 1). A SIGReg/BCS loss does exist in `eb_jepa/losses.py`.
- Planning: MPPI or CEM, 200 samples, 20 elites, 20 iterations, horizon 90. The cost sums the distance to the goal over all intermediate predicted states.
- Results: 97 ± 2% success on random wall (MPPI) and 96% (CEM). Removing IDM drops success to 1%.
- **Implication:** with a GRU and a single vector, EB-JEPA is a poorer host for factor-attention than LeWM.

### 2. Closest prior work

- **SPARTAN** (Lei, Schölkopf, Posner, NeurIPS 2025, arXiv 2411.06890): transformer world model over object-factored tokens. It samples a hard Bernoulli adjacency from the sigmoid of the attention logits, masks attention with it, penalises edges, and reports better few-shot adaptation and distractor robustness. **This is nearly your predictor.** Differences: it assumes a given factored representation, uses a local (state-dependent) graph rather than a global learnable A, and is not trained as a JEPA with SIGReg.
- **Baumgartner, Lei, Watson, Posner, CLeaR 2026** (PMLR v323): sparsity-regularised transformer plus theory that local causal structure is often *necessary* for identifiability. Differences: variational inference, synthetic domains, no JEPA.
- **S2-SSM** (Petri et al., arXiv 2505.02074): slot SSM with sparse cross-attention, measuring SHD on Interventional Pong. It reportedly beats dense attention on SHD with similar MSE. Difference: SSM instead of JEPA, one toy dataset.
- **Causal-JEPA** (Nam, Le Lidec, Maes, LeCun, Balestriero, ICML 2026; same group as LeWM): JEPA over frozen VideoSAUR/SAVi slots with object-level masking. Its attention mask is a fixed block-causal mask (`dinowm_causal.py`), with no learned adjacency. A summary of the paper says it does not extract or evaluate relational structure against ground truth. Difference: no learned A, no SHD, pretrained slot encoder rather than end-to-end.
- **Lachapelle et al., mechanism sparsity** (CLeaR 2022, arXiv 2107.10098; 2401.04890): a VAE with a learned binary mask over latent-to-latent transitions. It proves that **sparsity of the transition graph identifies the latents**. Difference: VAE rather than JEPA. In theory, it is the core of what your A penalty would be doing.
- **FIOC-WM** (Feng, Lippe, Magliacane, NeurIPS 2025): object-centric latents plus an explicit interaction structure from pretrained encoders, used for hierarchical RL. Not a JEPA.
- **Orthogonal JEPA / JEPA-Anything** (arXiv 2608.20065, 2609.20800): split the JEPA latent into orthogonal factors. No interaction graph. Unverified beyond abstracts.
- **"When Does LeJEPA Learn a World Model?"** (Klindt, LeCun, Balestriero, arXiv 2605.26379): SIGReg gives identifiability only up to a global rotation, and only for Gaussian latents. Unverified beyond summaries.
- **From memory, unverified:**
  - NRI (Kipf 2018): VAE that infers edge types over given trajectories.
  - C-SWM (Kipf 2020): contrastive slots with a dense GNN, no learned sparsity.
  - CDL (Wang 2022): conditional-independence-tested causal mask on state variables.
  - GRADER: graph-based causal dynamics on state variables.
  - OOCDM: object-oriented class-level causal graphs on state variables.
  - SlotFormer: dense transformer over slots.
  - RIM / Neural Production Systems: top-k sparse module communication, but not an explicit, SHD-evaluable graph.
  - VCD (Lei 2022): earlier work from the SPARTAN group.

### 3. Novelty verdict

About **70–80% is already done.**
- The predictor (adjacency-masked attention between factor tokens with a sparsity penalty, evaluated on SHD, OOD dynamics shifts and distractors) is SPARTAN, with S2-SSM and Baumgartner et al. as close variants.
- "JEPA plus object factors" is Causal-JEPA.

What I found no evidence of:
1. A learned sparse interaction graph inside an **end-to-end-from-pixels SIGReg JEPA**, where factors are *not* given by a pretrained slot encoder.
2. Any study of the interaction between SIGReg and a sparse graph.

**Most defensible angle: whether graph sparsity breaks the rotation symmetry left by SIGReg.** This is an identifiability question, not a benchmark-number question.
- An isotropic Gaussian is rotation-invariant, so SIGReg alone cannot make the K blocks align with true factors (Klindt et al.).
- Mechanism sparsity (Lachapelle) is exactly the kind of assumption that could break that symmetry.
- **Test:** train the factored LeWM in a synthetic environment with a known graph and measure block-to-factor alignment as well as SHD, with and without the A penalty.

This also connects to the month-9 ICA convergence in this project: the Gaussian is the non-identifiable case. Planning success on PushT on its own would very likely be read as incremental to SPARTAN plus LeWM.

### 4. Known failure modes to plan for

- **Factor misalignment:** blocks of a single CLS vector have no reason to align with objects, and SIGReg's rotation invariance actively mixes them.
  - If factors are entangled, A is meaningless and SHD is undefined; you would need a matching step, for example Hungarian matching on probe R².
  - C-JEPA avoids this only by using pretrained slots.
- **Mask collapse:**
  - A soft additive or multiplicative A inside the softmax gets renormalised away, so attention can route through any non-zero edge. SPARTAN uses *hard* sampled masks for this reason.
  - Too strong a penalty empties A (each factor predicts only itself); too weak leaves it dense.
  - The self-edge must be excluded from the penalty or forced on.
- **Multi-layer leakage:** with 6 layers, information can travel over 2-hop paths, so the effective graph is roughly a power of A, not A. Use a single interaction layer, or apply the same A in every layer and evaluate the transitive closure.
- **No in-distribution gain:** expect equal MSE and planning success in-distribution. Reported gains appear under dynamics shifts, few-shot adaptation and distractors (SPARTAN, S2-SSM). The S2-SSM review also notes an unspecified binarisation threshold for SHD; fix and report yours.
- **Global vs local graph:** a single global A cannot express contact interactions, which are state-dependent. SPARTAN's main argument for local graphs is that a global A in PushT-like scenes would come out dense.
- **Collapse confounds:** removing the IDM term in EB-JEPA drops success to 1%. Factor tokens could shortcut via the AdaLN action pathway. Check that the factors actually change with each action.