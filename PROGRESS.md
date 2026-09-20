# Progress

> Claude reads this at the start of every session and updates it at the end.
> Keep it honest. Overstating progress here only costs you later.

---

## Current position

**Month:** 1 · **Week:** 1
**Causal track:** Structural causal models — not started
**ML track:** Backprop from scratch (micrograd) — not started
**Started:** _(date)_
**Last session:** 2026-09-20 (a question about attention and causality; no track work started)

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

- **2026-09-20 — "Is there an attention mechanism trained for causality?"** Answered at landscape level: every such model imports its causal information from somewhere other than the observational data (a simulator with known graphs, a given DAG used as the attention mask, an assumed unconfoundedness plus a balancing loss, or interventional data), because distribution-to-graph is not a function. Not resolvable properly until attention is built (ML M7) and CRL is reached (M11). Two follow-ups:
  - After C2: on a fork where Z causes both X and Y, a head attending from Y to X with a large weight has learned what about whether X causes Y? Answer it from the six numbers.
  - At M7 or later: log AVICI (Lorch et al. 2022) and CATT (Yang et al. 2021) in `notes/papers.md` with the three-sentence judgement. Check whether CATT's "mediator" satisfies the front-door conditions from M3.

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
