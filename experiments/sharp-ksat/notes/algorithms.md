# The two October-2026 sub-exponential #k-SAT algorithms

Status of both: **unreviewed claims**. Both come with Lean developments written by AI models;
in both cases the authors themselves say no machine can check that the Lean statement means
what the paper says. Independent review (Impagliazzo, Paturi, Williams et al.) is in progress.

## Sources actually read

| Source | Where | Read? |
|---|---|---|
| Anthropic (Claude), *A 2^o(n) Time Algorithm for 3-SAT*, 26 Oct 2026, 49 pp | `hexagonmath.org/pdf/2610.00577v1` | Yes: §§1–6 and §8 in full, Appendix A (end), the Lean appendix E, skimmed B/D/G/H |
| OpenAI, *The Exponential Time Hypothesis is False*, 17 Oct, revised 25 Oct 2026, 67 pp | `github.com/openai/math`, `preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf` | Yes: §1, all of §2 (technique overview), §3.5, Appendix D.1–D.2. Skimmed §§3–6 headings |
| OpenAI Lean scope note | `openai/math/lean/docs/378.md` | Yes |
| Anthropic Lean repo | `github.com/anthropics/formal-math/tree/main/eth` (cited in the paper) | **No.** This session's GitHub access is limited to one repository, so it was not opened. The paper's Appendix E reproduces the statement file |
| OpenAI Lean source | `openai/math/lean/` | **No**, only the scope note. Same reason |
| Majdoddin, *Lowered crossover points of Anthropic's subexponential 3-SAT algorithm*, arXiv 2610.41622 | arxiv.org HTML | Abstract and main-results paragraph only |

Everything below is my paraphrase of the papers. Notation follows each paper. Where I am unsure,
it says **[unclear]**.

---

## 1. Anthropic: "rewrite-and-merge" on clique tensor networks

### Claimed bound
Theorem 1: #k-SAT in time `2^{O(min{ k·m/√log m , k²·n·log log n/√log n })}`, with an absolute
constant hidden in the O. The main text proves only the weaker Theorem 2: for every δ > 0, time
`O(2^{δn})`. **Space is not discussed as a headline**: the algorithm queries tables lazily, so it
is polynomial-ish per level. I did not find a stated space bound.

The paper itself says the bound in m does not beat 2^n until n > 10^10000 (for m = 5n), the bound
in n not until n > 10^10, and "our general approach seems unlikely to yield any general improvement
unless n > 10^7, however it is instantiated."

### The object it works on
A **network**: nodes a with local settings z_a, integer weights W_a(z_a), and pairwise factors
Ψ_{a,b} that "factor through c bits", meaning Ψ = Σ_{θ∈{0,1}^c} f(z_a,θ) f'(z_b,θ) (rank ≤ 2^c). Its
value is Z = Σ_z Π_a W_a Π_{a,b} Ψ_{a,b}. Every network can be turned into a **(K,λ)-clique**: K
nodes, every pair joined by a λ-bit string; node w holds K−1 strings, Ψ tests that the two copies
agree, W_w is an arbitrary integer table. This is a closed tensor network on the complete graph
K_K with bond dimension 2^λ.

Formula → network (§6): split the clauses into clusters. Each **occurrence** (clause, literal)
gets a bit holding the variable's proposed value. A variable's occurrences are chained by
equality **tests** (d occurrences ⇒ d−1 tests). Tests inside a cluster go into W, tests between
clusters are edge bits. Total edge bits < 3m. Note this is **linear in m, not n**.

### The key identity: eight squares (Lemma 2)
For 3-bit blocks, σ(p,q) = (−1)^{φ(p,q)}, φ = Σ_{i≤j} p_i q_j + q1 p2 p3 + p1 q2 p3 + p1 p2 q3. Then

    2·[x=v][y=u] = R(x,y,u,v) + C(x,y,u,v)
    R = [x⊕y⊕u⊕v=0] σ(x,y) σ(v,u)   (factors through 3 bits between (x,y) and (u,v))
    C = [x⊕y⊕u⊕v=0] σ(x,u) σ(v,y)   (factors through 3 bits between (x,u) and (y,v))

A "crossing" (two equality tests, each 3 bits) costs 6+6 = 12 bits summed over the two possible
bipartitions (rows-vs-columns). After the rewrite, each term costs 3 one way and 6 the other.
For λ-bit strings cut into t = ⌈λ/3⌉ blocks, expand Π_s (R_s + T·C_s) = Σ_i A_i T^i and group
by i, the number of column-choice blocks. Then Σ_i A_i = 2^t [x=v][y=u], and A_i factors through

    c_row(i) = min(2λ, 4t+3i)  bits row-wise,   c_col(i) = min(2λ, 7t−3i)  bits column-wise,

so c_row + c_col ≤ 11t ≈ (11/12)·(4λ). **This 11/12 is the entire source of the speedup.** The
extra t bits name *which* blocks took the column choice; without grouping by i the number of
terms would be 2^{t|R|} and eat the saving.

### One level (Proposition 3)
Input: a (g², λ)-clique placed on a g×g grid. Edges inside a row or column are "straight"; every
other edge is a diagonal of exactly one rectangle (2 rows × 2 columns).

```
Clique(ℓ, λ, tables W):
    if ℓ == 0: return Σ over all edge strings of Π_w W_w          # brute force, 2^{λ·K(K-1)/2}
    place K_ℓ = g² nodes on a g×g grid; t = ceil(λ/3); R = rectangles
    total = 0
    for ι : R → {0..t}:                                         # (t+1)^{|R|} terms, poly in λ
        Z_ι network: replace the two diagonals of each rectangle q by A_{ι(q)}(x_q,y_q,u_q,v_q)
        row net    = merge each grid row into one node; A factors through c_row bits
        column net = merge each grid column;              A factors through c_col bits
        pick the one with smaller total edge bits U         # ≤ (11/12 + 1/g)·U_before
        total += Network(ℓ-1, that net)
    return total / 2^{t|R|}

Network(ℓ, net on g nodes):                                       # Lemma 1
    try all K^g groupings into K = K_{ℓ-1} groups until one is "good"
        (every group ≤ 2g/K nodes, every pair of groups shares ≤ (U + gP/K)/K² bits)
    pad every group-pair edge to L bits; the group's table W̃_γ(Θ) is a sum over the
        group's internal settings (2^{≈5S/K} work per query, never stored)
    return Clique(ℓ, L, W̃)
```

Parameters: K_0 = 64, g_ℓ = 3K_ℓ^8, K_{ℓ+1} = g_ℓ² = 9K_ℓ^16, e_0 = 1/2,
e_{ℓ+1} = (11/12 + 2/K_ℓ)e_ℓ + 2/K_ℓ. The exponent per edge bit goes to 0 like (11/12)^ℓ while K
grows doubly exponentially. The top level clusters the m clauses into 3K^8 clusters, groups them, and
calls Clique. Sparsification (counting-preserving, Dell et al.) reduces to m = O(n).

### Where the speedup comes from
Not from contraction order. A single contraction tree for the clique cannot go below about 4/9 of
brute force (Appendix G, Austrin–Kaski–Kubjas). The algorithm escapes that model by expanding the
sum into a **signed sum of networks** and contracting **different terms in different orders**:
some terms merged by rows, some by columns. Signed (negative) table entries are essential.

### Constants left open / what Theorem 1 adds
- Theorem 2 uses Chebyshev balancing (g = 3K^8). Theorem 1 uses Hoeffding (g ≈ K·polylog K),
  pieces with ρ = 1/√2 from Appendix B (built from partition numbers), and lets K grow with m.
  The constant c of the Appendix B pieces is **not explicit** in the paper (the Lean constant is
  not explicit either). Majdoddin's follow-up computes one.
- For the n-bound, Appendix D replaces the clause graph by a variable graph with truncated
  inclusion–exclusion over violated clauses and sparse parity checks. **[unclear]** I only skimmed
  this and did not implement it.

### Things I don't fully understand
- Appendix B's ρ = 1/√2 pieces ("operators on a polynomial ring", partition-number bit costs). Not
  implemented, only the Lemma 2 (ρ = 11/12) pieces.
- Appendix H's account of how the decomposition was found. Read for colour only.

---

## 2. OpenAI: low-degree replacement forms + Taylor/Hermite interpolation

### Claimed bound
Theorem 1.1: for fixed k ≥ 2, `(n+m)^{O_k(1)} · 2^{O_k(n log log n / √log n)}` time and
`poly_k(n,m)` **space**; `2^{O_{k,D}(n/√log n)}` if m ≤ Dn. The main-text route (§§3–6) proves the
weaker exponent `O_k(n (log log n)² / √log n)`. That weaker bound is the one formalised in Lean.
The sharper bound is proved in Appendix D by a different route: a general **Boolean tensor
contraction theorem**.

### Core algorithm (bounded-occurrence 3-CNF, §2.1–2.7)
Count = Σ_{x∈{0,1}^n} Π_e Φ_e(x), with Φ_e = 1 − Π_{v∈e}(1 − L_{e,v}(X_v)) the multilinear
satisfaction polynomial.

1. **Replacement forms.** Replace each clause input X_v by a linear form ⟨P_{e,v}(T), X⟩ whose
   coefficients are polynomials in T of degree ≤ R, with P_{e,v}(0) = e_v. Define
   H(T) = Σ_x Π_e Φ_e(⟨P_{e,v}(T),x⟩). Then H(0) = #F and deg H ≤ M·R (M = number of occurrences).
2. **Designs at t = 1..Q.** At each nonzero point t the forms admit an (r,τ,D)-design: variables
   split into r groups of ≤ 2n/r, each clause assigned to one group, and each form depends on
   other groups **only through a few integer linear "measurements" y_h = C_h x_h** (≤ 2τn rows
   total, coefficient mass ≤ D per row). Then
   `H(t) = Σ_{ȳ} Π_g Σ_{u: C_g u = y_g} A_{t,g}(u; ȳ)`: enumerate measurement tuples, then each
   group separately. Cost ≈ (2D+1)^{2τn} · 2^{2n/r}.
3. **Taylor data + Hermite interpolation.** H has degree ~MR > Q, so take Taylor coefficients
   z^0..z^q of H(t+z), q = ⌊3nR/Q⌋, at each t. To keep the design valid while expanding in z,
   introduce a separate formal input X for the shift ΔP(z), restrict to Boolean X = 1_U, and use
   inclusion–exclusion over U ⊆ E, |E| ≤ q:
   `H(t+z) ≡ Σ_{|E|≤q} Σ_{U⊆E} (−1)^{|E|−|U|} Z_{t,E,U}(z)  (mod z^{q+1})`
   where Z_{t,E,U} sums over x with x_i = 1 on E and X = 1_U. Each Z is computed by step 2.
   Then solve the Hermite system for H(0).
4. **Constructing the forms (§2.7, §4).** Auxiliary coordinates B = ([r]×[r])^Q, D = r^{2Q}: a
   coordinate records (variable group, clause group) at each of the Q points. Π_s swaps the pair
   at position s. V0 = Π_s Π_s, W0 = ½ Σ_s (I − Π_s), M0 = V0·f(W0) with f of degree R−1
   interpolating (−1)^j on an interval I of R integers around Qp, p = (r−1)/(2r). Eigenvectors
   ψ (tensor products of e_aa, e_ab ± e_ba) with j ∉ I are the measurement rows C. Then
   `⟨P_u(T), ζ⟩ = ((I − M0)ζ)_u + Σ_w M0[u,w] · Π_{s∈E_{u,w}} (1 − T/s) · ζ_w`,
   E_{u,w} = points where w's variable group ≠ u's clause group. The first term is a function of
   the measurements. In the second, the cross-group terms vanish at those points. Schedules
   a_v, d_e ∈ [r]^Q place variables and clauses, derandomised so groups are ≤ 2n/r and label
   multiplicity h ≤ 2M/D.
5. **Parameters (2.19):** λ = ⌈log₂ n⌉, ν = ⌈log₂ λ⌉, r = ⌊√λ/ν²⌋, Q = ⌊λ/(128ν)⌋,
   R = 4⌈√λ⌉, τ = 1/(rλ), D = r^{2Q}. "For sufficiently large n": needs r ≥ 2 and 6R ≤ Q.

### Reduction for general k-CNF (§2.8)
Count-preserving sparsification (σ ~ (log n)^{−1/(2k)}) → split long clauses with OR-gates
(z = u ∨ v as 3 clauses) → Tovey copying: a variable with s occurrences becomes s copies tied by
an implication cycle, so every variable occurs ≤ 3 times. Both maps are bijections on solutions.
§2.9/§5–6 give the sharper route: parity tests from expander (Tanner) codes plus
Beigel–Tarui polynomials, counting common zeros of sparse F₂ equations.

### Where the speedup comes from
Three costs multiply: Taylor data `Σ_{j≤q} C(n,j) 2^j`, measurements `(2D+1)^{2τn}`, group
enumeration `2^{2n/r}`. Log of the product is `n·[ (R/Q) log(2Q/R) + τ log(2D+1) + 1/r ]`. All three
go to 0 together only because the M0 construction lets R ≈ √Q (binomial concentration of W) while
measurements stay a 2e^{−R²/8Q} fraction. **The algebraic trick is that the separating
substitution can be made low-degree in the interpolation variable.**

### Constants / parameters left open
- All constants in O_k(·) are left open. Parameters (2.19) are explicit but only valid "for
  sufficiently large n". Third-party estimate (Szeider, cited by Majdoddin): below ~2^{1.09·10^10}
  variables the main-text algorithm degenerates to exhaustive search.
- Appendix D's contraction theorem has a 9/10-power degree condition and constants A, b, C.

### Things I don't fully understand
- Appendix D.4 (the recursion on wires with 2-adic tapering of Hermite multiplicities). Read the
  summary only. Not implemented.
- §5–6 (Beigel–Tarui lift, Tanner-code tests, greedy row selection by exact counting). Not
  implemented: not needed for correctness on small inputs.
- The exact derandomisation of the schedules (Lemma 4.5). I use a greedy balancer instead. Any
  schedule gives a correct algorithm; only the cost changes.

---

## 3. Conceptual differences

| | Anthropic | OpenAI (main text) |
|---|---|---|
| Algebra | Integer **rank factorisations** of a 4-way equality ("crossing"). Eight-square / octonion identity | **Polynomial interpolation** in an auxiliary variable T. Low-degree substitution built from commuting involutions and a Krawtchouk-type interpolant |
| What is "small" | Bits crossing a cut between merged node groups | Number of integer measurements that couple variable groups, plus number of Taylor coefficients |
| Recursion | Deep recursion on cliques, K grows doubly exponentially per level | Flat: one design per point, one Hermite solve (the App. D route recurses) |
| Cost driver | Edge bits, **linear in m** (≈3m); n-bound only via sparsification + App. D | Variables n (after Tovey copying: occurrences ≈ 3m) |
| Negative numbers | Essential: signed tables, cancellations | Essential: rational substitutions, inclusion–exclusion |
| Space | Not headlined | Polynomial (headline claim) |
| Shared idea | Both break the sum into pieces that interact only through a small summary (θ bits / measurements ȳ), and both pay a polynomial-in-λ (or Q·C_q) number of extra terms to make that summary small | |

Both papers fall back on **tensor-network contraction** for their strongest bounds (Anthropic: the
whole method; OpenAI: Appendix D, Theorem D.1). That suggests the real common content is
"closed Boolean tensor networks of bounded degree contract in 2^{O(s/√log s)}", which would be
a statement worth checking on its own.
