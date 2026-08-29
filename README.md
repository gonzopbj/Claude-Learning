# Causal AI + Machine Learning — Year One

A twelve-month self-directed program running two parallel tracks: causal inference (6h/week) and modern machine learning (4h/week), merging in the final quarter. Built to be worked with Claude Code as a tutor.

Full plan: [`curriculum/year-plan.md`](curriculum/year-plan.md). Start here: [`SETUP.md`](SETUP.md).

---

## Why two tracks in parallel

Because they explain each other. Once you understand regularization concretely, double machine learning's entire motivation becomes obvious rather than mysterious. Once you have implemented ICA, LiNGAM stops being a separate algorithm and becomes the same theorem pointed at a different question. Studied separately, each field is harder than it needs to be.

The convergence points are marked in the plan. Month 9 is the strongest one.

---

## How this is meant to work

The central risk of learning with an AI assistant is that it does the work for you. The output makes sense, you feel like you understood it, and three weeks later you cannot reproduce any of it. Recognition is not understanding, and the gap between them is invisible from the inside.

This is worse on the ML track than the causal one, because a working training script *looks* like understanding. It isn't.

`CLAUDE.md` is built to prevent this. Claude will refuse to write your exercises, escalate hints one level at a time, and push back when your explanation is a paraphrase rather than a mechanism. It will still help you debug — reading a stack trace with you is teaching, handing you the fixed file is not.

**This only works if you don't fight it.** You will be tempted to say "just show me the code." That stays available and is sometimes right. Make it a decision, not a reflex.

---

## The weekly rhythm

| Block | Hours | |
|---|---|---|
| Causal theory | 3 | Reading and derivations, on paper |
| Causal code | 3 | Implementing, and breaking things |
| ML | 4 | Almost entirely implementation |

**Protect the causal theory block.** Deep learning has a faster feedback loop and is more immediately satisfying, so those hours quietly migrate. `CLAUDE.md` instructs Claude to call this out when it notices, but you should notice first.

---

## Working a session

```bash
cd causal-ai
claude
```

Then `/session`. Claude reads `PROGRESS.md`, tells you where you left off on both tracks, and proposes a target.

Tell Claude when you're stopping so it updates `PROGRESS.md`.

- `/checkpoint` at the end of each month — closed-book assessment against both milestones. Failing one is information, and it's much cheaper to find a gap in month 4 than in month 9 when everything depends on it.
- `/quiz <topic>` any time.
- `/connect` when the two tracks feel unrelated. They usually aren't.

---

## Two files that matter more than they look

**`notes/failures.md`** — every time a method or a model surprises you by breaking, record the setup and the mechanism. By month twelve this outweighs your notes on the methods themselves, because knowing when something fails is what separates someone who can use a tool from someone who can be trusted with it.

**`PROGRESS.md`**, specifically Known Gaps. Be honest. Nobody else reads it.

---

## Two rules

**Simulate everything (causal).** You can generate data from a known ground truth and check whether your estimator recovers it. Most of statistics can't. Don't waste it.

**Implement, then look (both).** Write it yourself first, then read the reference implementation. Reading working code produces recognition; writing broken code produces understanding.
