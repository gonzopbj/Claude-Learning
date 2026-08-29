# Setup

One-time, ~30 minutes.

## 1. Repository

```bash
cd causal-ai
git init
git add -A
git commit -m "Initial two-track curriculum scaffold"
```

Commit at the end of every session. Not for backup — so that in month nine you can run `git log` and see what you actually did, which is a more honest record than memory.

## 2. Python environment

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import numpy, networkx, sklearn; print('core ok')"
```

## 3. PyTorch and GPU

Install the build matching your CUDA version from https://pytorch.org — do not `pip install torch` blindly, it may give you a CPU-only build.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False`, sort it out now. Debugging CUDA in month 7 while trying to understand attention is a bad use of your hours.

**On the cluster:** find out early how job submission works, whether it's Slurm or something else, and what the queue times look like. Write yourself a working submission script in month 1 while nothing depends on it. Ask Claude for help with this — it's boilerplate and teaches nothing.

**You will not need the GPU until month 5.** Months 1–4 are deliberately small-scale. Scale is a distraction from mechanism.

## 4. Causal libraries

Not needed until month 3, and `causal-learn` can be awkward to install. They're commented out in `requirements.txt`. Uncomment when you reach them.

## 5. Claude Code

Install from https://docs.claude.com/en/docs/claude-code/overview, then launch from the repository root:

```bash
cd causal-ai
claude
```

Launching from the root matters — `CLAUDE.md` loads from the working directory and above. Run `/memory` to confirm it's active.

## 6. First session

```
/session
```

Claude reads `PROGRESS.md`, sees you're at the start, and proposes a first target on each track.

---

## Directory layout

```
causal-ai/
├── CLAUDE.md              tutor instructions — loaded every session
├── README.md
├── SETUP.md
├── PROGRESS.md            living state
├── requirements.txt
├── .claude/commands/      slash commands
├── curriculum/
│   ├── year-plan.md
│   └── month-NN.md        generated as you reach each month
├── notes/
│   ├── failures.md
│   └── papers.md
├── src/
│   ├── causal/
│   └── ml/
├── tests/
└── notebooks/
```

---

## Optional: personal preferences

`~/.claude/CLAUDE.md` applies across all your projects:

```markdown
- Explain the why before the how.
- When I make an error, tell me directly. Don't soften it.
- Prefer showing me a failing case over describing one.
```

---

## Adjusting the plan

The year plan is a hypothesis, not a contract. If month 2 takes six weeks, it takes six weeks — say so and have Claude rescope in `PROGRESS.md`.

The one thing not to adjust silently is the 6/4 hour split. Deep learning is more immediately rewarding and will quietly eat the causal hours if you let it. If you want to change the balance, change it deliberately and write down why.
