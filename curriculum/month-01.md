# Month 1 — Structural Causal Models · Backprop from Scratch

> Also the structural template for months 2–12.

**This month's split:** 6h/week causal, 4h/week ML. No convergence point yet — the two tracks are laying independent foundations. The first real convergence is month 5.

---

# Causal Track — Structural Causal Models

## Why this month exists

You cannot reason carefully about causation using intuition, because intuition has no rules you can check. An SCM is a mathematical object you can manipulate, and once you have it, questions that felt philosophical become questions with answers.

The thing to come away with: **a causal model contains strictly more information than a probability distribution**, and you can see exactly where the extra information lives.

## Week 1 — The problem

**Read:** Pearl, Glymour & Jewell, *Primer*, Ch. 1.

- The three-level hierarchy: association, intervention, counterfactual. What lives at each level, and why a level cannot be climbed with data alone.
- Confounding. Selection bias. Simpson's paradox. Berkson's paradox.
- Conditional independence with its own algebra. Review the graphoid axioms. Notice it is not transitive.

**Do:** work Simpson's paradox until it stops being paradoxical. The resolution is that whether to aggregate depends on the causal structure, not on the numbers. The same table supports opposite conclusions under different structures — construct both.

**Sit with this:** two datasets are numerically identical. Under what circumstances should you act differently on them?

## Week 2 — Building the object

**Read:** *Primer*, Ch. 2.1–2.3.

- Structural equations: $X_i := f_i(\mathrm{PA}_i, U_i)$. The `:=` is assignment, not equality.
- Exogenous noise, and where the randomness comes from.
- The graph induced by an SCM. The distribution induced by an SCM.
- **The key asymmetry:** SCM → graph → distribution is a function. Distribution → graph is not. Many SCMs induce the same distribution. Everything difficult in causal discovery descends from this.

**Exercise C1 — SCM simulator** *(from scratch, no libraries)*

Takes structural equations plus a noise specification, samples from them.

- Arbitrary DAG, arbitrary functional forms
- Samples in topological order
- At least Gaussian and uniform exogenous noise
- Returns a DataFrame

*Ask Claude for tests, not the implementation.*

## Week 3 — The three structures

Every DAG is built from three patterns:

| | Structure | Marginal | Conditioned on $Z$ |
|---|---|---|---|
| Chain | $X \to Z \to Y$ | dependent | independent |
| Fork | $X \leftarrow Z \to Y$ | dependent | independent |
| Collider | $X \to Z \leftarrow Y$ | **independent** | **dependent** |

The collider row breaks people's intuition and matters most in practice.

**Exercise C2 — Correlation study.** For each structure compute $\mathrm{Corr}(X,Y)$ and $\mathrm{Corr}(X,Y \mid Z)$. **Predict all six numbers before running.** Write them down first.

**Exercise C3 — Collider bias in the wild.** Find a real example in a domain you know. Not a textbook one. Three sentences: the variables, why the collider arises, who is fooled by it.

*Seed: why does the correlation between talent and attractiveness among successful actors tell you nothing about the correlation in the general population?*

## Week 4 — Consolidation

**Exercise C4 — Adversarial simulation.** Construct an SCM where naively regressing $Y$ on $X$ gives a coefficient with the **wrong sign** relative to the true causal effect. Then construct one where the sign is wrong because of a *collider* you conditioned on. The second is harder and is the point.

**Exercise C5 — Write it down.** Half a page from memory, no notes: what an SCM is, what it gives you that a joint distribution does not, why the collider case is counterintuitive.

---

# ML Track — Backprop from Scratch

## Why this month exists

Every later confusion in deep learning traces back to not really knowing what a gradient is in this setting. People who skip this stage spend years with a vague sense that training is alchemy. Build it once, mechanically, and it never confuses you again.

You have a GPU. You will not use it this month. That is deliberate.

## Weeks 1–2 — micrograd

**Watch and follow:** Karpathy, *Neural Networks: Zero to Hero*, part 1 (micrograd).

- The computation graph as an explicit data structure
- The chain rule as graph traversal, not as a formula
- Forward pass, backward pass, `.backward()` demystified
- Why the topological order matters (you have seen this word already this month, in the SCM sampler — the connection is not deep, but it is not nothing either)

**Exercise M1 — Rebuild micrograd from memory.** Follow along once. Then close the video and rebuild it. Scalar `Value` class, operator overloading, autograd. It will not work the first time. That is the exercise.

## Week 3 — MLP in pure numpy

**Exercise M2 — No autograd at all.**

Forward pass, loss, backward pass written out by hand, parameter update. Train it on something small — a 2D classification toy problem is fine.

- Derive the gradient of the loss with respect to each weight matrix on paper first
- Then implement
- Then check with finite differences: perturb a weight, recompute the loss, compare to your analytic gradient

The finite-difference check is not optional. It is how you find out you were wrong.

## Week 4 — PyTorch, and verification

**Exercise M3 — The same network, three ways.**

Rewrite it in PyTorch. Then verify all three implementations agree:

- micrograd gradients vs. numpy gradients vs. PyTorch gradients
- Match to machine precision on a fixed input and fixed weights

When they disagree — and they will — the disagreement is the most instructive thing that happens this month.

**Then:** read the PyTorch autograd docs and notice that you already know what everything does.

---

## Milestones

**Causal:** given three variables and a described mechanism, state without hesitation whether conditioning on the third helps or hurts, and justify it structurally.

**ML:** derive and implement backprop for a two-layer network without reference to anything, and verify it against finite differences.

Run `/checkpoint` when you think both are met.

---

## Failure modes to watch for

**Causal — memorizing the three-structure table instead of understanding it.** The check: can you explain *why* conditioning on a collider creates dependence? (Explaining away: if you know the effect occurred and one cause was absent, the other becomes more likely.)

**Causal — treating faithfulness as automatic.** A graph implies independencies; a distribution *having* them is a separate claim. You will meet this properly in month 2, but start noticing it now.

**ML — following the video without rebuilding.** Following along produces the feeling of understanding and none of the substance. The rebuild-from-memory step is the whole exercise; the follow-along is just preparation for it.

**ML — reaching for the GPU.** Nothing this month needs one. Scale is a distraction from mechanism.

**Both — moving on because the reading is finished.** The reading is not the month. The implementations are.

---

## Looking ahead

**Causal M2** generalizes the three structures into d-separation, which handles arbitrary graphs and conditioning sets. If the collider case is solid, month 2 is mostly bookkeeping. If it is not, month 2 will be miserable.

**ML M2** is training dynamics — why networks fail to train, diagnosed from gradient histograms. It assumes you can instrument a network you fully understand, which is exactly what you will have built.
