# Progress

> Claude reads this at the start of every session and updates it at the end.
> Keep it honest. Overstating progress here only costs you later.

---

## Current position

**Month:** 1 · **Week:** 1
**Causal track:** Structural causal models — not started
**ML track:** Backprop from scratch (micrograd) — not started
**Started:** _(date)_
**Last session:** —

**Next target:** SCM simulator in numpy (C1), and follow micrograd part 1.

---

## Hours split

Target is 6 causal / 4 ML. Log roughly, weekly. Drift toward ML is the expected failure mode.

| Week | Causal | ML | Note |
|---|---|---|---|
| | | | |

---

## Completed

_Nothing yet._

<!-- Format:
### Month 1
**Causal**
- [x] C1 SCM simulator
- [ ] C2 correlation study
**ML**
- [x] M1 micrograd rebuilt from memory
- [ ] M2 numpy MLP + finite-difference check
-->

---

## Open questions

Left unresolved at the end of a session. Claude raises these first next time.

- (2026-09-20, from the attention discussion) Take X ← Z → Y with Z observed and no edge
  X → Y. Train a transformer to predict Y from (X, Z). What will the attention weight from Y's
  position onto X be, and what would it need to be for "attention weights are causal edges"
  to survive? Predict before reasoning it through.
- (2026-09-21, from the AGI discussion) Of the six requirements for a continually learning
  system, which can a purely observational learner satisfy even in principle, and which need
  the ability to act? Sort them and defend the sort.
- (2026-09-21) Before opening `experiments/mechanism-shift/results/report/REPORT.md`, predict
  which of its nine pre-registered predictions held. Five did. The four that failed are in
  `notes/failures.md`.

---

## Explorations outside the plan

Things built or read ahead of the curriculum, at the student's request. Not progress on the
milestones; listed so they are not mistaken for it.

| Date | What | Where | Student's role |
|---|---|---|---|
| 2026-09-21 | Mechanism-shift testbed: six continual-learning requirements tested on a synthetic SCM under sparse mechanism shifts, 14 agents, nine pre-registered predictions | `experiments/mechanism-shift/` | Asked for it; did not write it. To be reread in months 2, 5 and 10 when the pieces are rebuilt from scratch. |

---

## Known gaps

Prerequisites that surfaced as missing and haven't been fixed. Being listed is not a failure — sitting here for two months is.

| Gap | Surfaced in | Status |
|---|---|---|
| Asymptotic statistics (rates, CLT, delta method) | Anticipated, causal M6 | Scheduled, not started |

---

## Milestones

| Month | Causal | ML | |
|---|---|---|---|
| 1 | Conditioning on a third variable: helps or hurts, instantly | Backprop derived + implemented, verified by finite differences | ☐ ☐ |
| 2 | d-separation by hand, 8-node DAG, under a minute | Diagnose five training failures from curves alone | ☐ ☐ |
| 3 | Any DAG + query → identifying formula or non-identifiability proof | SGD/momentum/Adam from scratch; double descent reproduced | ☐ ☐ |
| 4 | Translate freely between backdoor and unconfoundedness | Quantify regularization-induced bias in a known coefficient | ☐ ☐ |
| 5 | Show by simulation exactly when IPW blows up | CNN from scratch; demonstrate its inductive bias failing | ☐ ☐ |
| 6 | Explain why XGBoost on observational data can't give a causal coefficient | SimCLR trained; representation shown non-robust to a nuisance | ☐ ☐ |
| 7 | Articulate why CATE can't be validated like supervised learning | Working GPT from scratch, with an interpreted attention head | ☐ ☐ |
| 8 | PC from scratch, matching `causal-learn` | VAE from scratch, ELBO derived on paper first | ☐ ☐ |
| 9 | Evidenced written critique of a published benchmark | FastICA driving both LiNGAM and latent analysis | ☐ ☐ |
| 10 | Build a case where ERM fails, explain it causally | Model-based agent broken by one unseen parameter change | ☐ ☐ |
| 11 | World model broken by an unseen intervention, failure characterized | ☐ | |
| 12 | 15–20 page synthesis, written from memory | ☐ | |

---

## Pace notes

Deviations and why. Useful in month 6 when deciding whether to compress or extend.

_None yet._
