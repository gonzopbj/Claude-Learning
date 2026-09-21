# Mechanism-shift testbed

**What this is.** A small, fully synthetic experiment that puts six requirements for a
continually learning system to the test on a world where the truth is known exactly:

1. an error signal from outside the system (here: interventions on the world),
2. calibrated uncertainty and the ability to abstain,
3. a representation factored into causal mechanisms,
4. credit assignment: attributing an error to the mechanism that changed,
5. memory that does not overwrite what still holds,
6. an improvement signal that cannot be gamed.

The proposed system (`mech-full`) is a Bayesian structural causal model: it learns the graph
from its own interventions, keeps one conjugate regression per mechanism, resets only the
mechanism whose residuals say it changed, and answers `do(X_i = v)` queries by propagating
posterior draws through the mutilated learned graph. It is compared against baselines that
lack one ingredient each, and against oracles that have the truth.

**What this is not.** It is not an AGI system and it does not claim to be. It is the
experiment you run before claiming anything. Read `REPORT.md` for which pre-registered
predictions held and which did not, and read the failure cases first.

**Status in the curriculum.** This was built at the student's request as a demonstration,
ahead of the curriculum. It is not one of his exercises and he did not write it. The right
way to use it is to read `SPEC.md`, predict each result before opening `REPORT.md`, and
then, in months two, five and ten, rebuild the pieces (SCM simulator, structure learning from
interventions, invariance under shift) from scratch as the plan prescribes.

## Files

| File | What |
|---|---|
| `SPEC.md` | The experimental design, revised after three independent critiques. Read first. |
| `INTERFACES.md` | The contract between world, agents, protocol and metrics. |
| `world.py` | Linear-Gaussian SCM, sparse mechanism shifts, hidden-confounder variant, exact interventional means. |
| `protocol.py` | The eight-step episode loop every agent runs; the evaluator lives here. |
| `agents/` | `mech.py` (the proposed system and its ablations), `baselines.py`, `oracle.py`. |
| `metrics.py` | The nine metrics and the bootstrap / Holm analysis helpers. |
| `calibrate.py` | Tunes every agent's constants on validation seeds, never on evaluation seeds. |
| `run.py`, `report.py`, `run_all.sh` | Run one sweep, build the report, reproduce everything. |
| `tests/` | 151 property tests, including the no-leakage tests. |
| `results/report/REPORT.md` | The results, with verdicts on the nine pre-registered predictions. |

## Reproduce

```bash
pip install numpy scipy scikit-learn matplotlib pytest
cd experiments/mechanism-shift
python -m pytest tests -q        # about 40 s
bash run_all.sh                  # about 20 minutes on 4 cores
```

Raw per-prediction arrays under `results/*/raw/` are git-ignored; everything in the report is
recomputable from the seeds.
