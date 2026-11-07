# Decisions log

Autonomous run. Each entry: decision, then why.

1. **Location: `experiments/sharp-ksat/`**, not repo-root `src/`, `tests/`, `notes/`, `REPORT.md`.
   The repo is a causal-inference/ML learning programme. Its `src/causal`, `src/ml`, `tests/`
   and `notes/` belong to the student's curriculum work, and CLAUDE.md forbids filling those with
   implementations. This task is off-curriculum. It follows the existing `experiments/<name>/`
   convention (cf. `experiments/mechanism-shift`). All task-specified paths are relative to that
   directory.
2. **Not logged in PROGRESS.md hours.** Off-curriculum. One line under "Open questions" points to it.
3. **Sources.** Anthropic paper from hexagonmath.org (PDF). OpenAI paper from the raw file in the
   public `openai/math` repo. Neither Lean repository was opened: GitHub access in this session is
   scoped to one repo, and raw access worked only for the OpenAI paper and docs. Lean scope is taken
   from each paper's own statement.
4. **Exact arithmetic via modular reduction (OpenAI).** The paper works over ℚ with exact rationals.
   I work modulo 31-bit primes (2^31 − 1, plus more primes combined by CRT until their product
   exceeds 2^(occurring vars)). The 31-bit size keeps int64 numpy products from overflowing. All
   divisions are by small integers, invertible mod p, and the count is below the modulus, so the
   residue is the count. Same result, much faster. Documented as an implementation deviation,
   not an algorithmic one.
5. **OpenAI: free parameters (r, Q, R).** The paper's (2.19) parameters are only defined for
   astronomically large n: r ≥ 2 needs √log₂n ≥ 2(log₂log₂n)². I expose r, Q, R as knobs. The
   algorithm is exact for *any* valid choice (interval I ⊆ [0,Q], R ≤ Q+1). Only cost changes.
6. **OpenAI: Tovey copying optional** (`copy=True` faithful, default `copy=False`). Copying makes
   occurrences ≤ 3 per variable, which the asymptotic analysis needs (M = O(n)). Correctness only
   needs deg H ≤ M·(R−1) with M the actual number of occurrences. Without copying the group
   enumeration runs over n variables instead of ≈3m. Both modes are tested.
7. **OpenAI: no sparsification, no §5–6 parity route, no Appendix D.** Sparsification and the
   parity route only change the asymptotics. Leaving them out keeps the result exact.
8. **OpenAI: greedy schedule balancing** instead of Lemma 4.5's derandomised conditional
   expectations. Any schedule is correct.
9. **Anthropic: one recursion level on a 2×2 or 3×3 grid**, with level 0 evaluated by tensor
   contraction instead of literal brute force over edge strings. The paper's constants (K₀ = 64,
   g = 3K⁸ ≈ 2·10^14) make even level 1 physically impossible. Level-0 contraction computes the
   same value. Brute force is also available as an option and is used in tests.
10. **Anthropic: grouping heuristic.** Lemma 1 tries all K^g groupings and keeps a "good" one. Good
    groupings are guaranteed only for g = 3K⁸. With clauses as clusters (g = m), I use a
    deterministic balanced local search that minimises the largest pair of group edge bits.
11. **Anthropic: tables as tensors, not lazy queries.** The paper evaluates W̃_γ(Θ) on demand
    (2^{≈5S/K} work per query). I materialise merged-part tensors with einsum. Same value, more
    memory. This limits the instance sizes that can be run.
12. **Timeouts.** The task asks for 10 minutes per instance and ≥10 instances per point. That is
    infeasible inside ~2 h of compute for 5 methods × 5 families. I use **60 s per instance**, 10
    instances per point, and stop scaling a method on a family once the median instance times out.
    Growth exponents are fitted only on points where all instances finished.
13. **No external exact counter.** No sharpSAT-TD, d4 or ganak binary is installed, and building
    one is out of scope. The DPLL counter (with component caching) and a tensor-network contractor
    (opt_einsum) are the practical baselines.
14. **Anthropic: occurrence chains ordered by group.** §6 says the occurrences of a variable may
    be chained "in any order". I chain them in group order, so each variable adds at most one
    edge bit per pair of consecutive groups it touches. Edge bits are then ≤ n(K−1), instead of up
    to ≈ 3m for an arbitrary order.
15. **Anthropic: swap-only grouping.** Group sizes stay balanced (they differ by at most 1). The
    first heuristic was allowed to empty groups. On a test formula it collapsed four groups into
    two, which is legal under (G1)/(G2) but leaves the 2×2 grid with no diagonals.
16. **Anthropic: `max_L = 6`.** Dense row tensors have up to 2^{4L} entries. L = 7 would need
    ≈2 GB per tensor. Instances with L > 6 are reported as `oom`, not as timeouts.
17. **Test sizes for the paper algorithms.** Instances have n ≤ 20 declared variables. For the
    OpenAI scheme at most 10 variables occur (R = 1), or at most 6 (R = 2, Hermite path). For the
    Anthropic scheme m ≤ 9. Their cost is ≈ 2^(occurring vars) × Python overhead, so hundreds of
    n = 20 dense instances would take hours. The baselines are tested on dense n ≤ 20 instances.
