# Causal AI + Machine Learning — Year One

**Calibration:** ~10 focused hours/week. Second-year data science undergrad. Strong on pandas/sklearn, near-zero on deep learning. GPU access available. Degree coursework covers none of this, so the plan carries everything. Goal is deep understanding; artifacts secondary.

That's roughly 500 hours across two fields. Here is what it honestly buys you.

**On the causal side:** genuine competence. You'll read any paper in the field, implement the core algorithms from scratch, and — more valuably — recognize when a published result is an artifact of its benchmark. This is the anchor and it gets the larger share.

**On the ML side:** proficiency, not mastery. You'll have built a transformer, a VAE, and a small world model from scratch, which means you'll understand what these systems actually do rather than what their papers say they do. You will not be positioned to push the deep learning state of the art. Nobody is at 200 hours.

That combination — deep in one field, literate in the adjacent one — is worth more at your stage than depth in either alone. Causal representation learning exists because Schölkopf knew both causality and kernel methods well. The transferable idea comes from the second field, and you cannot carry a tool you have only read about.

---

## The two-track design

Six hours causal, four hours ML, every week, in parallel. They merge in the final quarter.

Parallel rather than sequential, because the tracks make each other easier:

| You learn | Which makes this obvious |
|---|---|
| Regularization and overfitting (ML, M4) | Why double ML needs orthogonality and cross-fitting (causal, M6) |
| Architectural inductive bias (ML, M5) | That causal assumptions are inductive bias by another name |
| ICA and latent identifiability (ML, M9) | Why LiNGAM works at all (causal, M9) — they are the same theorem |
| d-separation and confounding (causal, M2) | Why your model latched onto a spurious feature |

Studied apart, each is harder than it needs to be. The convergences below are marked **⇄** and they are not decorative — they are the reason for the structure.

---

## Month-by-month at a glance

| # | Causal track (6h/wk) | ML track (4h/wk) | |
|---|---|---|---|
| 1 | Structural causal models | Backprop from scratch | |
| 2 | d-separation, Markov equivalence | Training dynamics | |
| 3 | do-calculus, backdoor, frontdoor | Optimization & the generalization puzzle | |
| 4 | Potential outcomes | Regularization, overfitting, double descent | |
| 5 | IPW, g-computation | Architecture as inductive bias | ⇄ |
| 6 | AIPW, double ML + asymptotics | Self-supervised learning | ⇄ |
| 7 | Heterogeneous effects | Transformers from scratch | |
| 8 | PC algorithm from scratch | Latent variable models, VAEs | ⇄ |
| 9 | LiNGAM, ANM, the benchmark critique | ICA and identifiability | ⇄⇄ |
| 10 | Invariance, ICM, ICP | RL and model-based RL | ⇄ |
| 11 | **Merged:** causal representation learning + world models / JEPA | | ⇄⇄⇄ |
| 12 | **Merged:** project and synthesis | | |

---

## Weekly rhythm

| Block | Hours | |
|---|---|---|
| Causal theory | 3 | Reading, derivations, on paper |
| Causal code | 3 | Implementing, and breaking things |
| ML | 4 | Almost entirely implementation |

One paper per week from month 3. Notes weekly.

The ML track is more code-heavy on purpose: deep learning understanding comes from watching training fail in specific ways, not from reading about it. The causal track is more paper-heavy on purpose, because its difficulties are conceptual rather than empirical.

**Protect the causal theory block.** It is the first thing sacrificed when a model won't converge, and it is the part that compounds.

---

## Two rules that outrank the syllabus

**Simulate everything (causal).** Causal inference has a property most of statistics lacks: you can generate data from a known ground truth and check whether your estimator recovers it. Never accept a method you have not watched succeed *and fail* on synthetic data where you control the answer.

**Implement from scratch, then use the library (both).** For every core algorithm — d-separation, backdoor search, IPW, AIPW, PC, ANM, backprop, attention, the ELBO — write it yourself first, then check against the reference implementation. The order is not negotiable. Reading working code produces recognition; writing broken code produces understanding.

---

## Resources

**Causal**
1. Pearl, Glymour & Jewell, *Causal Inference in Statistics: A Primer* — months 1–3. Written for exactly your level.
2. Hernán & Robins, *What If* (free PDF, Harvard) — months 4–7. The best treatment of potential outcomes in existence.
3. Peters, Janzing & Schölkopf, *Elements of Causal Inference* (free PDF, MIT Press) — months 8–11.
4. Pearl, *Causality* — a reference, not a linear read.
5. Brady Neal, *Introduction to Causal Inference* (free video + notes) — companion for months 1–5.

**ML**
1. **Karpathy, *Neural Networks: Zero to Hero*** (free video series) — months 1–2 and again for month 7. micrograd → makemore → nanoGPT. From-scratch philosophy, matches this plan exactly. Your primary DL text.
2. Prince, *Understanding Deep Learning* (free PDF) — the modern reference. Read alongside.
3. Sutton & Barto, *Reinforcement Learning* (free) — month 10, chapters 1–6 and 8.
4. Kingma & Welling, *An Introduction to Variational Autoencoders* — month 8.
5. Goodfellow, Bengio & Courville, *Deep Learning* — reference only, dated in parts.

**Stack:** `numpy`, `scipy`, `pandas`, `matplotlib`, `networkx`, `scikit-learn`, `pytorch`, then `causal-learn`, `dowhy`, `econml`, `doubleml`.

---

# Phase 1 — Foundations (Months 1–3)

## Month 1

### Causal — Structural causal models

**Why now:** you need a formal object. "Causation" as intuition has no rules you can check; an SCM is a thing you can manipulate.

- The three-level hierarchy: association, intervention, counterfactual. Which questions live where, and why a level cannot be climbed with data alone.
- Confounding, selection bias, Simpson's paradox, Berkson's paradox. Work Simpson's until it stops feeling paradoxical — the resolution is that the right answer depends on causal structure, not on the data.
- Conditional independence as an object with its own algebra. It is not transitive.
- SCMs: $X_i := f_i(\mathrm{PA}_i, U_i)$. The `:=` is assignment, not equality. The asymmetry is the point.
- **The key asymmetry:** SCM → graph → distribution is a function. Distribution → graph is not. Everything hard in causal discovery descends from this.

**Build:** an SCM simulator in pure numpy. Then chain, fork, and collider — compute $\mathrm{Corr}(X,Y)$ and $\mathrm{Corr}(X,Y \mid Z)$ for each. Predict all six numbers before running.

### ML — Backprop from scratch

**Why now:** every later confusion in deep learning traces back to not knowing what a gradient actually is here. Build it once and it never confuses you again.

- Karpathy's micrograd, followed along, then rebuilt from memory.
- Scalar autograd: the computation graph, the chain rule as graph traversal, backward passes.
- Then an MLP in pure numpy — forward, loss, backward, update — no autograd at all.
- Then the same in PyTorch, and verify the gradients match your numpy version to machine precision.

**Deliverable:** a working MLP trained on something small, written twice, agreeing with itself.

---

## Month 2

### Causal — d-separation and Markov equivalence

- d-separation: blocked at chains and forks when conditioned on, blocked at colliders **unless** the collider or a descendant is conditioned on. Learn this cold.
- Markov condition (graph independencies ⇒ distributional ones) and faithfulness (the converse, an *assumption*). Construct an SCM where two causal paths cancel exactly and faithfulness fails.
- Markov equivalence classes: same skeleton, same v-structures. CPDAGs.
- Consequence: observational data alone can never distinguish $X \to Y$ from $Y \to X$ in the linear-Gaussian bivariate case. This limitation drives half the field.

**Build:** d-separation yourself on a networkx DiGraph, checked against `networkx.d_separated`. Then enumerate all implied CIs for a DAG and test each on simulated data — note the gap between the population claim and finite-sample reality.

**Milestone:** d-separation by hand on an 8-node DAG, under a minute.

### ML — Training dynamics

**Why now:** most people's mental model of training is "call `.fit()` and hope." Yours should be mechanistic.

- Initialization: why bad init kills a network, and what Xavier/He actually fix.
- Activation and gradient distributions across layers. Instrument them and *look* at them.
- Vanishing and exploding gradients, seen directly rather than described.
- Batch norm and layer norm: what they do, and the honest fact that the original explanation was wrong.
- Learning rate as the single most important hyperparameter.

**Build:** deliberately break a network five ways — bad init, no normalization, learning rate too high, too low, dead ReLUs. Diagnose each from the loss curve and gradient histograms alone. Write down the signature of each failure.

---

## Month 3

### Causal — do-calculus

**Why now:** the payoff. You can now ask *when is the effect of an action computable from data I merely observed?*

- The `do` operator as graph surgery: intervening on $X$ deletes edges into $X$, because intervention overrides the structural equation.
- Truncated factorization / g-formula. Derive it from the SCM definition.
- **Backdoor criterion** — and construct a violation of each clause to see why each is there.
- **Frontdoor criterion** — estimating an effect despite an unobserved confounder, via a mediator. This is where the field stops feeling like reformatted regression.
- The three rules of do-calculus, and its completeness: if a quantity is identifiable, do-calculus derives it.
- What *not* to condition on: colliders, mediators, M-bias. The "control for everything" instinct is actively harmful. Demonstrate it.

**Build:** a backdoor adjustment set finder. Then simulate a frontdoor scenario with unmeasured confounding, show naive regression is biased and the frontdoor estimator is not. Then simulate M-bias — a case where adjusting for a *pre-treatment* variable introduces bias.

**Milestone:** given any DAG and query, produce an identifying formula or a proof of non-identifiability.

### ML — Optimization and the generalization puzzle

- SGD, momentum, Adam. What each fixes and what each breaks.
- Learning rate schedules, warmup.
- The loss landscape: sharp vs. flat minima, and appropriate skepticism about the visualizations.
- **The puzzle:** modern networks have more parameters than data points and generalize anyway. Classical statistical learning theory says they should not. Read about double descent. Sit with the fact that this is not fully resolved.

**Build:** implement SGD, momentum, and Adam from scratch. Reproduce a double descent curve. Watch test error go down, up, and down again.

---

# Phase 2 — Estimation and Representation (Months 4–6)

## Month 4

### Causal — Potential outcomes

**Why now:** there are two languages for causality — Pearl's graphs and Neyman–Rubin potential outcomes. Formally equivalent, but the estimation literature is written in the second.

- $Y_i(1)$, $Y_i(0)$. The fundamental problem: you only ever observe one.
- Consistency, SUTVA, ignorability, positivity.
- ATE, ATT, CATE — when each is the right question.
- Why randomization works, stated precisely: treatment independent of potential outcomes.
- **The translation:** unconfoundedness ⟺ the backdoor criterion holds. Do this explicitly until automatic.

**Reading:** *What If*, Part I.

**Build:** the same SCM, simulated as an RCT and as an observational study. Quantify bias against confounding strength.

### ML — Regularization and overfitting

**Why now:** this is the prerequisite for month 6, the hardest month of the year. Build it properly here and month 6 becomes tractable.

- L1/L2, dropout, early stopping, data augmentation. What each does to the hypothesis space.
- The bias-variance decomposition, and where it stops being the right frame for deep nets.
- Cross-validation done correctly, and the many ways it leaks.
- **The idea to carry forward:** every regularized estimator is *deliberately biased*. You accept bias to reduce variance. Fine when you want prediction. A disaster when you want an unbiased causal effect — and month 6 is the fix.

**Build:** measure the bias a regularized model introduces into a coefficient whose true value you know. Vary regularization strength. Plot bias against it. **Save this plot** — you will use it in month 6.

---

## Month 5 ⇄

### Causal — Classical estimators

- Outcome regression / g-computation.
- Stratification and matching. Why matching is intuitive and statistically awkward.
- The propensity score $e(x) = P(T = 1 \mid X = x)$ and the balancing property — conditioning on a *scalar* suffices. Remarkable, and it does not rescue you from needing the right $X$.
- IPW: reweighting into a pseudo-population. Derive unbiasedness.
- Diagnostics: overlap, standardized mean differences, effective sample size. Positivity violations and the pathology of extreme weights.

**Reading:** *What If*, Part II.

**Build:** IPW and g-computation from scratch, no libraries. Then deliberately misspecify the propensity model, then the outcome model, and watch each estimator break in its own way. Then real data — LaLonde/NSW or IHDP.

**Milestone:** demonstrate by simulation exactly when IPW blows up.

### ML — Architecture as inductive bias ⇄

- Convolutions: weight sharing and translation equivariance as an *assumption about the world* baked into the architecture.
- Why CNNs beat MLPs on images with fewer parameters. It is not capacity — it is a better prior.
- Pooling, receptive fields, residual connections.
- **⇄ The convergence:** a causal graph is an inductive bias. A convolution is an inductive bias. Both are structural assumptions that buy sample efficiency when correct and bias when wrong. Once you see this, "which assumptions am I making?" becomes the same question in both fields.

**Build:** a CNN from scratch, then in PyTorch. Compare against an MLP with matched parameter count. Then break the assumption — train on rotated images and watch the translation prior fail to help.

---

## Month 6 ⇄ — *the hard month*

### Causal — Double machine learning

**Budget extra time. This is the intellectual center of modern causal estimation.**

- AIPW / doubly robust estimation. Derive it. Prove double robustness: consistent if *either* nuisance model is correct.
- **Why naive ML plugins fail** — and you already have both halves. Regularization bias: your ML model is biased by design (you plotted this in month 4). Overfitting bias: the same data used to fit nuisances and estimate the effect.
- **Cross-fitting** fixes the second.
- **Neyman orthogonality** fixes the first: construct a moment condition whose derivative with respect to the nuisance parameters vanishes, so first-order nuisance error does not propagate.
- The headline result: nuisances converging at $n^{-1/4}$ suffice for $\sqrt{n}$-consistent, asymptotically normal effect estimates. Understand where the rate comes from — it is a product of two errors.
- Influence functions, gently. The concept, not the full semiparametric theory.

**Math side-track, run in parallel:** consistency, $\sqrt n$ rates, LLN, CLT, the delta method, plug-in bias. Any intermediate mathematical statistics text. Since your degree is not covering this, it has to happen here. Do not skip it as a detour — it gates everything after.

**Build:** AIPW with cross-fitting from scratch. Compare naive ML plugin, plugin without cross-fitting, and full DML across many simulations. Plot the sampling distribution of each. Watch the naive versions produce confidence intervals with badly wrong coverage.

**Milestone:** explain to a skeptical engineer why they cannot fit XGBoost to observational data and read off the treatment coefficient.

### ML — Self-supervised learning ⇄

- Why labels are the bottleneck, and what pretext tasks buy you.
- Contrastive methods: SimCLR, InfoNCE. Why negatives matter.
- Non-contrastive: BYOL, and the fact that it works without negatives — which was surprising and remains partly unexplained.
- Masked prediction: BERT, MAE. Predicting missing parts as a route to representation.
- **⇄ The connection:** SSL learns representations by exploiting known structure in the data. Causal discovery learns structure by exploiting known constraints on representations. Same coin.

**Build:** SimCLR on CIFAR-10, small scale. Evaluate with linear probing. Then check whether the representation is robust to a nuisance factor you introduced deliberately. It probably is not, and that failure is month 10's whole subject.

---

# Phase 3 — Discovery and Latent Structure (Months 7–9)

## Month 7

### Causal — Heterogeneous effects *(compressed)*

- CATE: $\tau(x) = E[Y(1) - Y(0) \mid X = x]$.
- Meta-learners: S, T, X, DR, R. What each assumes, when each is biased. The S-learner's regularization can shrink the treatment effect toward zero — and you will know exactly why by now.
- Causal forests: honest splitting, splitting on effect heterogeneity rather than outcome variance.
- **The evaluation problem:** you never observe individual treatment effects, so you cannot compute test error. Not a technicality — a structural limitation. Uplift curves and their limits.

**Build:** T- and X-learners yourself; EconML for causal forests. Vary heterogeneity and confounding. You should end the month noticeably less confident in CATE methods than the literature's tone suggests. That is the correct calibration.

### ML — Transformers from scratch

**Why now:** you have the foundations, and attention is not hard once you do.

- Attention as learned, content-dependent routing. Queries, keys, values.
- Multi-head attention, positional encoding, the residual stream.
- Karpathy's nanoGPT, built along, then rebuilt from memory.
- Tokenization, and how much LLM weirdness is downstream of it.
- Scaling laws, conceptually.

**Build:** a small GPT trained on a corpus you choose. Then instrument it — look at attention patterns, find a head that does something interpretable. Use the GPU here; this is what it is for.

---

## Month 8 ⇄

### Causal — Causal discovery

**Why now:** so far the graph was given. Where does it come from?

- CI testing: partial correlation, $\chi^2$, kernel-based (HSIC, KCIT). The core problem: as conditioning sets grow, power collapses.
- **PC algorithm:** start complete, remove edges by CI tests of increasing order, orient v-structures, propagate Meek's rules. Output is a CPDAG.
- Order-dependence and PC-stable.
- **FCI** and latent confounders. PAGs. "There is a hidden common cause" as an inferable conclusion.
- **GES:** greedy search over equivalence classes with a decomposable score.
- Multiple testing and error accumulation — why discovery output on real data is far less reliable than papers imply.

**Build: implement PC from scratch.** The single most clarifying implementation exercise in the field. Then compare against `causal-learn`, vary sample size, and watch performance degrade faster than you expect.

### ML — Latent variable models ⇄

- Autoencoders, then the leap to variational: the ELBO, the reparameterization trick, KL as a regularizer.
- Posterior collapse and why it happens.
- β-VAE and disentanglement — what people hoped for.
- **⇄ The connection:** a VAE posits latent variables that generate observations. An SCM posits latent noise variables that generate observations. These are the same shape of model asked different questions. The VAE asks "can I compress?"; the SCM asks "what happens if I intervene?" Month 9 shows why the second question is much harder to answer from data.

**Build:** a VAE from scratch — derive the ELBO on paper first. Train it, inspect the latent space, deliberately induce posterior collapse, then fix it.

---

## Month 9 ⇄⇄ — *the convergence month*

### Causal — Functional models and the honest critique

- Why asymmetry is recoverable from bivariate data if you restrict the function class. Additive noise models: if $Y = f(X) + N$ with $N \perp X$, the reverse direction generically admits no such model.
- **LiNGAM:** linear, non-Gaussian, acyclic. Identifiability via ICA. Understand precisely why non-Gaussianity breaks the symmetry.
- Post-nonlinear models, IGCI.
- **NOTEARS** and the continuous-optimization wave: acyclicity as the smooth constraint $\mathrm{tr}(e^{W \circ W}) - d = 0$, then gradient descent.
- **Then the critique.** Reisach et al., "Beware of the Simulated DAG": much of that literature's performance came from *varsortability* — marginal variance increasing along the topological order in the synthetic data generators. The algorithms were partly reading off an artifact. Standardize and the results largely evaporate.
- Time series: Granger causality and its real assumptions, PCMCI.

### ML — ICA and identifiability ⇄⇄

- Linear ICA: recovering independent sources from mixtures. Why non-Gaussianity is what makes it possible — the same theorem underlying LiNGAM, arrived at from the other direction.
- Nonlinear ICA and its **unidentifiability** without extra structure.
- **Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"** — the impossibility result. Unsupervised disentanglement requires inductive bias or supervision. Full stop.
- iVAE and identifiability via auxiliary variables.

**Build (shared across both tracks):** implement FastICA. Use it to build LiNGAM. Then run it on a VAE's latent space. Watch the same mathematics do causal and representational work. Then reproduce the varsortability result: run NOTEARS on synthetic data, standardize, rerun, watch the score drop.

**Milestone — the most important in this plan:** write a short, blunt, evidenced critique of a published benchmark. Not for publication. Research taste is the ability to see through a favorable evaluation, and it is built by doing this once, deliberately, carefully.

---

# Phase 4 — Convergence (Months 10–12)

## Month 10 ⇄

### Causal — Invariance and the ICM principle

- **Independent Causal Mechanisms:** the causal factorization decomposes into modules that do not inform each other. Interventions change one and leave the rest intact. This is the deep reason causal models transfer and statistical ones do not.
- Sparse mechanism shift.
- **ICP:** the causal parents of $Y$ are the feature set whose predictive relationship is invariant across environments. Elegant, conservative, exponential.
- **IRM and what happened to it.** Read the original, then the failure analyses (Rosenfeld et al.; Gulrajani & Lopez-Paz's DomainBed). A beautiful idea that did not deliver empirically. Witnessing this properly is worth more than most successful papers.
- Distribution shift taxonomy mapped onto causal structure.

### ML — Reinforcement learning ⇄

- MDPs, value functions, Bellman equations.
- Model-free: Q-learning, policy gradients, actor-critic.
- **Model-based RL:** learn the dynamics, plan in the model.
- **⇄ The convergence:** a learned dynamics model is a conditional density. A causal model is a family of densities indexed by interventions. Model-based RL works while the policy stays near the data distribution and degrades as it departs — which is precisely a statement about intervention. The agent's own exploration *is* interventional data, and it is mostly unexploited.

**Reading:** Sutton & Barto, chapters 1–6 and 8.

**Build:** ICP on multi-environment synthetic data. Then the colored-MNIST setup where a spurious feature is predictive in training and anti-predictive at test — confirm ERM fails, try IRM, see for yourself how fragile the improvement is. Then a simple model-based agent in a gym environment, and change one environment parameter it never saw.

---

## Month 11 ⇄⇄⇄ — *fully merged, 10h/week*

### Causal representation learning and world models

**Why now:** everything before assumed the causal variables were given. The frontier: what if they must be *learned* from pixels?

- What interventions buy you: with interventional data, latent causal variables become identifiable under conditions. This is why embodiment matters, and it is the strongest argument for causality in robotics.
- Schölkopf et al., "Toward Causal Representation Learning" — the agenda-setting survey. You now have every prerequisite for it.
- **World models:** latent dynamics, the Dreamer lineage, planning in latent space.
- **JEPA:** predicting in representation space rather than pixel space. LeCun's argument for why generative pixel prediction is the wrong objective. Read the argument, then read the critiques.
- Throughout, ask: is this model learning causal structure, or an excellent conditional density that will break under intervention? Usually the second. Be able to say exactly what would have to change.

**Build:** train a small latent-dynamics world model on a control task. Then intervene on the environment in a way absent from training — change a mass, a friction coefficient, an object's color. Measure the degradation and characterize it. Write up what a causal version would need and why it is hard.

---

## Month 12 — Project and synthesis

Pick one and go deep for three weeks.

**A — LLMs and causal reasoning.** Do LLMs reason causally or retrieve memorized causal claims? The "causal parrots" line and the counterarguments. LLMs as priors over graph structure — surprisingly effective, surprisingly unreliable. Build an LLM-assisted discovery pipeline and evaluate it honestly against a classical baseline on data the model cannot have memorized. That last clause is the entire difficulty.

**B — Embodied causality.** Interventions as an agent's native operation. Build an agent that performs *targeted* interventions to identify structure, versus one acting randomly. Measure sample efficiency.

**C — Causal representation learning.** Reproduce an identifiability result, then break its assumptions one at a time and document where it fails.

**Final two weeks, regardless of lane:** write 15–20 pages, for yourself, explaining the whole territory — SCMs through world models — without looking anything up. The gaps you hit while writing are your real gaps. This is the highest-value fortnight in the plan.

---

## Ongoing habits

**One paper per week** from month 3. Three passes: abstract + figures + conclusion; then method; then details only if earned. Most do not earn the third. For each, three sentences: what they claim, what must be true for it to hold, what they did not test.

**A failure notebook.** Every time a method surprises you by breaking, record the setup and the mechanism. By month twelve this is worth more than your notes on the methods themselves.

**Monthly re-derivation.** Close everything. Re-derive the backdoor adjustment formula, the AIPW estimator, backprop for a two-layer net, and the ELBO. If you cannot, you have lost it — cheap to recover now, expensive later.

**Email an author** of a paper you replicated, with one specific sharp question. Most academics answer. This is how the research world opens up, and it costs nothing.

---

## Where you'll be

The plan is deliberately front-loaded with fundamentals because the frontier moves and the fundamentals do not. do-calculus will be identical in twenty years. Backprop will be identical in twenty years. Whatever is state-of-the-art in world models right now will not be, and the people who adapt fastest are the ones who understood the invariants.

Month 12 is not the finish line. What you should have: fluency in both causal languages, working from-scratch implementations of everything central in both fields, calibrated skepticism about published claims, and a clear sense of which open problem you would want to attack.

That last item is the actual output. Everything else is scaffolding for it.
