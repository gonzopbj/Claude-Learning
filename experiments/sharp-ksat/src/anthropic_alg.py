"""Claude (Anthropic), "A 2^o(n) Time Algorithm for 3-SAT" (Oct 2026): rewrite-and-merge.

What is implemented:

  * the eight-square sign σ and the identity 2[x=v][y=u] = R + C           (§2.4, eq. 12, 27-31)
  * the pieces A_0..A_t of Lemma 2 for λ-bit strings, with their explicit row-wise and
    column-wise factorisations through c_row(i) = min(2λ, 4t+3i) and c_col(i) = min(2λ, 7t-3i)
    bits                                                                     (§4, Lemma 2)
  * the formula -> network of §6 (occurrence bits, equality tests chained per variable) and the
    grouping/padding of Lemma 1 into a (K, L)-clique with integer tables      (§6, Lemma 1)
  * ONE level of Proposition 3 on a g x g grid (K = g², g = 2 or 3): expand every rectangle's
    two diagonals into Σ_i A_i, and for each ι: R -> {0..t} merge rows or columns (whichever has
    the smaller total U, eq. 35-36), contract the merged parts, add, divide by 2^{t|R|}
  * level 0 ("brute force over edge strings" in the paper) as an exact tensor contraction, or
    literally as brute force for tiny cliques.

Deviations (see notes/decisions.md): the paper's constants (K0 = 64, g = 3K^8) are unusable,
so this is a 1-level instance with K = 4 or 9; groupings are found by local search instead of
trying all K^g groupings; tables are materialised as dense integer tensors instead of being
evaluated lazily per query; level 0 is an einsum contraction by default.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import opt_einsum as oe

from .cnf import CNF, normalize

# ----------------------------------------------------------------------------- §2.4 identity


def _phi(p, q):
    P = [(p >> i) & 1 for i in range(3)]
    Qb = [(q >> i) & 1 for i in range(3)]
    s = sum(P[i] * Qb[j] for i in range(3) for j in range(i, 3))
    s += Qb[0] * P[1] * P[2] + P[0] * Qb[1] * P[2] + P[0] * P[1] * Qb[2]
    return s & 1


SIGMA = np.array([[1 - 2 * _phi(p, q) for q in range(8)] for p in range(8)], dtype=np.int64)
_X = np.arange(8)
_XOR0 = ((_X[:, None, None, None] ^ _X[None, :, None, None] ^ _X[None, None, :, None]
          ^ _X[None, None, None, :]) == 0).astype(np.int64)
# R[x,y,u,v] = [x^y^u^v=0] σ(x,y) σ(v,u);  C[x,y,u,v] = [x^y^u^v=0] σ(x,u) σ(v,y)
R_BLOCK = _XOR0 * SIGMA[:, :, None, None] * SIGMA.T[None, None, :, :]
C_BLOCK = _XOR0 * SIGMA[:, None, :, None] * SIGMA.T[None, :, None, :]


def psi(k, p, q):
    """ψ_k(p, q) = [p ⊕ q = k] σ(p, q)."""
    return int((p ^ q) == k) * int(SIGMA[p, q])


def eight_square_holds() -> bool:
    eq = np.zeros((8, 8, 8, 8), dtype=np.int64)
    for x, y in itertools.product(range(8), repeat=2):
        eq[x, y, y, x] = 1  # [x=v][y=u] at (x,y,u=y,v=x)
    return bool(np.array_equal(R_BLOCK + C_BLOCK, 2 * eq))


# ----------------------------------------------------------------------------- Lemma 2 pieces


def n_blocks(lam: int) -> int:
    return -(-lam // 3)


def c_row(i, lam):
    t = n_blocks(lam)
    return min(2 * lam, 4 * t + 3 * i)


def c_col(i, lam):
    t = n_blocks(lam)
    return min(2 * lam, 7 * t - 3 * i)


def _blocks(lam):
    """Map each λ-bit string value to its t blocks of 3 bits (zero-padded)."""
    t = n_blocks(lam)
    vals = np.arange(1 << lam)
    return [(vals >> (3 * s)) & 7 for s in range(t)]


def pieces_dense(lam: int):
    """A_0..A_t as dense int arrays of shape (2^λ,)*4 indexed [x, y, u, v]."""
    t = n_blocks(lam)
    bl = _blocks(lam)
    P = [np.ones((1,) * 4, dtype=np.int64)]
    for s in range(t):
        b = bl[s]
        Rs = R_BLOCK[np.ix_(b, b, b, b)]
        Cs = C_BLOCK[np.ix_(b, b, b, b)]
        newP = []
        for i in range(len(P) + 1):
            term = 0
            if i < len(P):
                term = term + P[i] * Rs
            if i >= 1:
                term = term + P[i - 1] * Cs
            newP.append(term)
        P = newP
    return [np.broadcast_to(a, (1 << lam,) * 4).copy() for a in P]


def _theta_list(t, i, cheap_on_C: bool):
    """Enumerate θ = (Q, θ') for Lemma 2(b): Q ⊆ [t], |Q| = i, per-block payloads.

    For the row-wise factorisation the R-blocks (s ∉ Q) cost 3 bits (k) and the C-blocks
    (s ∈ Q) cost 6 bits (x_s, y_s). Column-wise, the roles swap. `cheap_on_C` selects the
    column-wise case. Returns a list of (Q tuple, payload tuple).
    """
    out = []
    for Qset in itertools.combinations(range(t), i):
        per = []
        for s in range(t):
            cheap = (s in Qset) if cheap_on_C else (s not in Qset)
            per.append(range(8) if cheap else range(64))
        for pay in itertools.product(*per):
            out.append((Qset, pay))
    return out


def factor_piece(i, lam, rowwise=True):
    """Explicit factorisation of A_i through c_row(i) (rowwise) or c_col(i) bits.

    Returns (F, G, nbits) with A_i[x,y,u,v] = Σ_θ F[x,y,θ] G[u,v,θ] (rowwise) or
    A_i[x,y,u,v] = Σ_θ F[x,u,θ] G[y,v,θ] (columnwise). nbits = log2 of the paper's θ-space
    size (the number of distinct θ actually used can be smaller; unused ones are zero).
    """
    t = n_blocks(lam)
    N = 1 << lam
    cbits = c_row(i, lam) if rowwise else c_col(i, lam)
    if cbits >= 2 * lam:
        # trivial factorisation through the 2λ bits of one side
        A = pieces_dense(lam)[i]
        if rowwise:
            F = np.eye(N * N, dtype=np.int64).reshape(N, N, N * N)
            G = A.reshape(N * N, N, N).transpose(1, 2, 0).copy()
        else:
            F = np.eye(N * N, dtype=np.int64).reshape(N, N, N * N)
            G = A.transpose(0, 2, 1, 3).reshape(N * N, N, N).transpose(1, 2, 0).copy()
        return F, G, 2 * lam
    bl = _blocks(lam)
    thetas = _theta_list(t, i, cheap_on_C=not rowwise)
    F = np.ones((N, N, len(thetas)), dtype=np.int64)
    G = np.ones((N, N, len(thetas)), dtype=np.int64)
    for ti, (Qset, pay) in enumerate(thetas):
        f = np.ones((N, N), dtype=np.int64)
        g = np.ones((N, N), dtype=np.int64)
        for s in range(t):
            bs = bl[s]
            P1, P2 = bs[:, None], bs[None, :]   # first and second argument blocks
            isC = s in Qset
            if rowwise:
                # F over (x, y), G over (u, v)
                if not isC:  # R block, θ = k: ψ_k(x,y) ψ_k(v,u)
                    k = pay[s]
                    f *= ((P1 ^ P2) == k) * SIGMA[P1, P2]
                    # G[u, v] = ψ_k(v, u)
                    g *= ((P2 ^ P1) == k) * SIGMA[P2, P1]
                else:        # C block, θ = (θ1, θ2) = (x_s, y_s)
                    th1, th2 = pay[s] & 7, pay[s] >> 3
                    f *= (P1 == th1) * (P2 == th2)
                    # G[u, v] = [θ1^θ2^u^v = 0] σ(θ1, u) σ(v, θ2)
                    g *= ((th1 ^ th2 ^ P1 ^ P2) == 0) * SIGMA[th1, P1] * SIGMA[P2, th2]
            else:
                # F over (x, u), G over (y, v)
                if isC:      # C block, θ = k: ψ_k(x,u) ψ_k(v,y)
                    k = pay[s]
                    f *= ((P1 ^ P2) == k) * SIGMA[P1, P2]
                    # G[y, v] = ψ_k(v, y)
                    g *= ((P2 ^ P1) == k) * SIGMA[P2, P1]
                else:        # R block, θ = (x_s, u_s)
                    th1, th2 = pay[s] & 7, pay[s] >> 3
                    f *= (P1 == th1) * (P2 == th2)
                    # G[y, v] = [θ1^y^θ2^v=0] σ(θ1, y) σ(v, θ2)
                    g *= ((th1 ^ P1 ^ th2 ^ P2) == 0) * SIGMA[th1, P1] * SIGMA[P2, th2]
        F[:, :, ti] = f
        G[:, :, ti] = g
    return F, G, cbits


# ----------------------------------------------------------------------------- cliques


@dataclass
class Clique:
    """(K, L)-clique: node w has a dense table of shape (2^L,)*(K-1), axes = other nodes in
    increasing order; the value is Σ over one L-bit string per edge of Π_w W_w."""
    K: int
    L: int
    tables: list

    def axes(self, w):
        return [u for u in range(self.K) if u != w]


def _sym_gen():
    c = itertools.count()
    return lambda: oe.get_symbol(next(c))


def clique_level0(cl: Clique, mode="contract"):
    K = cl.K
    if K == 1:
        return int(cl.tables[0].sum()) if cl.tables[0].ndim else int(cl.tables[0])
    if mode == "brute":
        edges = list(itertools.combinations(range(K), 2))
        tot = 0
        for vals in itertools.product(range(1 << cl.L), repeat=len(edges)):
            ev = dict(zip(edges, vals))
            prod = 1
            for w in range(K):
                idx = tuple(ev[(min(w, u), max(w, u))] for u in cl.axes(w))
                prod *= int(cl.tables[w][idx])
                if prod == 0:
                    break
            tot += prod
        return tot
    new = _sym_gen()
    esym = {e: new() for e in itertools.combinations(range(K), 2)}
    terms = ["".join(esym[(min(w, u), max(w, u))] for u in cl.axes(w)) for w in range(K)]
    return int(oe.contract(",".join(terms) + "->", *cl.tables, optimize="optimal" if K <= 5 else "greedy"))


def clique_level1(cl: Clique, stats=None):
    """One level of Proposition 3 on the g x g grid. Exact: returns the clique's value."""
    K, lam = cl.K, cl.L
    g = math.isqrt(K)
    if g * g != K or g < 2:
        raise ValueError("level 1 needs K = g^2, g >= 2")
    t = n_blocks(lam)
    node = lambda a, j: a * g + j  # noqa: E731
    rects = [(a, a2, j, j2) for a, a2 in itertools.combinations(range(g), 2)
             for j, j2 in itertools.combinations(range(g), 2)]
    # cache factorisations
    fac = {}

    def get_fac(i, rowwise):
        key = (i, rowwise)
        if key not in fac:
            fac[key] = factor_piece(i, lam, rowwise)
        return fac[key]

    total = 0
    U_before = 2 * (K * (K - 1) // 2) * lam  # every bit counted at both ends, as in §3
    stats = stats if stats is not None else {}
    stats.update(K=K, g=g, lam=lam, t=t, n_rects=len(rects), n_terms=(t + 1) ** len(rects),
                 U_before=U_before)
    U_chosen, max_part_bits, max_cut_bits = [], 0, 0
    for iota in itertools.product(range(t + 1), repeat=len(rects)):
        # totals of eq. (35): straight edges between two rows: g*lam; rectangles: c_row / c_col
        pair_bits_row = {}
        pair_bits_col = {}
        for (a, a2, j, j2), i in zip(rects, iota):
            pair_bits_row[(a, a2)] = pair_bits_row.get((a, a2), g * lam) + c_row(i, lam)
            pair_bits_col[(j, j2)] = pair_bits_col.get((j, j2), g * lam) + c_col(i, lam)
        # eq. (35): U = 2 Σ_{pairs} (gλ + Σ_q c(ι(q))), every bit counted at both ends
        U_row = 2 * sum(pair_bits_row.values())
        U_col = 2 * sum(pair_bits_col.values())
        rowwise = U_row <= U_col
        U_chosen.append(min(U_row, U_col))
        Z, pb, cb = _contract_term(cl, g, rects, iota, rowwise, get_fac, node)
        max_part_bits = max(max_part_bits, pb)
        max_cut_bits = max(max_cut_bits, cb)
        total += Z
    div = 1 << (t * len(rects))
    if total % div:
        raise ArithmeticError("sum of terms not divisible by 2^{t|R|}: identity broken")
    stats.update(U_after_mean=float(np.mean(U_chosen)), U_after_max=int(max(U_chosen)),
                 max_part_log2=max_part_bits, max_cut_bits=max_cut_bits)
    return total // div


def _contract_term(cl, g, rects, iota, rowwise, get_fac, node):
    """Z_ι: merge rows (or columns), contract each merged part, then contract the parts."""
    K, lam = cl.K, cl.L
    new = _sym_gen()
    # symbols: straight edges share one symbol; diagonal copies get one symbol per end
    sym = {}
    rect_of_diag = {}
    for q, (a, a2, j, j2) in enumerate(rects):
        rect_of_diag[(node(a, j), node(a2, j2))] = q
        rect_of_diag[(node(a, j2), node(a2, j))] = q
    for w, u in itertools.combinations(range(K), 2):
        if (w, u) in rect_of_diag or (u, w) in rect_of_diag:
            sym[(w, u)] = new()
            sym[(u, w)] = new()
        else:
            s = new()
            sym[(w, u)] = sym[(u, w)] = s
    theta = [new() for _ in rects]
    # part p = row index (rowwise) or column index
    part_of = (lambda w: w // g) if rowwise else (lambda w: w % g)  # noqa: E731
    parts = [[] for _ in range(g)]  # list of (tensor, index string)
    for w in range(K):
        parts[part_of(w)].append((cl.tables[w], "".join(sym[(w, u)] for u in cl.axes(w))))
    tdims = []
    for q, ((a, a2, j, j2), i) in enumerate(zip(rects, iota)):
        F, G, _ = get_fac(i, rowwise)
        tdims.append(F.shape[2])
        nx, ny, nu, nv = node(a, j), node(a, j2), node(a2, j), node(a2, j2)
        x = sym[(nx, nv)]
        y = sym[(ny, nu)]
        u = sym[(nu, ny)]
        v = sym[(nv, nx)]
        if rowwise:   # F(x,y,θ) in row a, G(u,v,θ) in row a2
            parts[a].append((F, x + y + theta[q]))
            parts[a2].append((G, u + v + theta[q]))
        else:         # F(x,u,θ) in column j, G(y,v,θ) in column j2
            parts[j].append((F, x + u + theta[q]))
            parts[j2].append((G, y + v + theta[q]))
    merged, max_bits = [], 0
    for pl in parts:
        counts = {}
        for _, ix in pl:
            for c in ix:
                counts[c] = counts.get(c, 0) + 1
        # an index is internal to the part if it appears twice inside it
        out = "".join(c for c in dict.fromkeys("".join(ix for _, ix in pl)) if counts[c] == 1)
        T = oe.contract(",".join(ix for _, ix in pl) + "->" + out, *[t for t, _ in pl],
                        optimize="greedy")
        merged.append((T, out))
        max_bits = max(max_bits, math.log2(max(T.size, 1)))
    # cut bits between parts (the level-0 instance on g merged nodes)
    allidx = "".join(ix for _, ix in merged)
    cut_bits = 0.0
    for c in set(allidx):
        # dimension of that index
        for T, ix in merged:
            if c in ix:
                cut_bits += math.log2(T.shape[ix.index(c)])
                break
    Z = oe.contract(",".join(ix for _, ix in merged) + "->", *[T for T, _ in merged], optimize="optimal")
    return int(Z), max_bits, cut_bits


# ----------------------------------------------------------------------------- formula -> clique


def _group_clauses(clauses, K, seed=0, iters=2000):
    """Balanced grouping of clauses into K groups minimising the largest pair of group edge bits."""
    m = len(clauses)
    rng = np.random.default_rng(seed)
    grp = np.arange(m) % K
    rng.shuffle(grp)

    def pair_bits(grp):
        bits = np.zeros((K, K), dtype=np.int64)
        byvar = {}
        for ci, c in enumerate(clauses):
            for l in c:
                byvar.setdefault(abs(l), set()).add(int(grp[ci]))
        for gs in byvar.values():
            gs = sorted(gs)
            for g1, g2 in zip(gs, gs[1:]):
                bits[g1, g2] += 1
                bits[g2, g1] += 1
        return bits

    def score(grp):
        b = pair_bits(grp)
        return (int(b.max()), int(b.sum()))

    best = score(grp)
    # swaps only: group sizes stay balanced (differ by at most one), so no group is empty
    # and the grid keeps its diagonals (the paper only bounds sizes from above).
    for _ in range(iters):
        i, j = int(rng.integers(m)), int(rng.integers(m))
        if grp[i] == grp[j]:
            continue
        grp[i], grp[j] = grp[j], grp[i]
        s = score(grp)
        if s <= best:
            best = s
        else:
            grp[i], grp[j] = grp[j], grp[i]
    return grp


def formula_to_clique(clauses, K, seed=0):
    """§6 network on clause groups, then Lemma 1 padding -> (K, L)-clique with dense tables.

    Occurrences of a variable are chained in group order, so all occurrences inside one
    group are tied by in-group tests (one shared bit) and consecutive groups by one edge bit.
    """
    grp = _group_clauses(clauses, K, seed) if clauses else np.zeros(0, dtype=np.int64)
    byvar = {}
    for ci, c in enumerate(clauses):
        for l in c:
            byvar.setdefault(abs(l), set()).add(int(grp[ci]))
    tests = {}  # (g1, g2) with g1<g2 -> list of variables
    for v, gs in sorted(byvar.items()):
        gs = sorted(gs)
        for g1, g2 in zip(gs, gs[1:]):
            tests.setdefault((g1, g2), []).append(v)
    L = max([1] + [len(vs) for vs in tests.values()])
    tables = []
    for gamma in range(K):
        new = _sym_gen()
        vsym = {}
        terms, tensors = [], []
        for ci, c in enumerate(clauses):
            if grp[ci] != gamma:
                continue
            for l in c:
                vsym.setdefault(abs(l), new())
            t = np.ones((2,) * len(c), dtype=np.int64)
            t[tuple(0 if l > 0 else 1 for l in c)] = 0
            tensors.append(t)
            terms.append("".join(vsym[abs(l)] for l in c))
        out = ""
        for u in range(K):
            if u == gamma:
                continue
            vs = tests.get((min(gamma, u), max(gamma, u)), [])
            for pos in range(L):
                s = new()
                out += s
                if pos < len(vs):
                    tensors.append(np.eye(2, dtype=np.int64))
                    terms.append(vsym[vs[pos]] + s)
                else:
                    tensors.append(np.array([1, 0], dtype=np.int64))  # padded bit = 0
                    terms.append(s)
        if not tensors:
            tables.append(np.array(1, dtype=np.int64))
            continue
        T = oe.contract(",".join(terms) + "->" + out, *tensors, optimize="greedy")
        # reshape binary legs into one 2^L index per neighbour (bit pos -> bit pos of value)
        T = np.asarray(T, dtype=np.int64).reshape((2,) * len(out))
        nb = K - 1
        T = T.reshape([2] * (nb * L))
        # axis order: neighbour-major, bit pos 0..L-1 ; value = Σ bit_pos << pos
        perm = []
        for b in range(nb):
            perm += [b * L + pos for pos in reversed(range(L))]
        T = T.transpose(perm).reshape((1 << L,) * nb)
        tables.append(T)
    edge_bits = sum(len(vs) for vs in tests.values())
    return Clique(K, L, tables), {"L": L, "edge_bits": edge_bits, "groups": grp.tolist()}


class AnthropicCounter:
    """#SAT via §6 network -> (K, L)-clique -> [one level of Prop. 3] -> level-0 contraction."""

    def __init__(self, g=2, levels=1, level0="contract", seed=0, max_L=6):
        self.g, self.levels, self.level0, self.seed, self.max_L = g, levels, level0, seed, max_L
        self.stats: dict = {}

    def count(self, cnf: CNF) -> int:
        clauses, has_empty = normalize(cnf)
        self.stats = {"g": self.g, "levels": self.levels}
        if has_empty:
            return 0
        occ = {abs(l) for c in clauses for l in c}
        n_free = cnf.n - len(occ)
        if not clauses:
            return 1 << cnf.n
        K = self.g * self.g
        cl, info = formula_to_clique(clauses, K, self.seed)
        self.stats.update(L=info["L"], edge_bits=info["edge_bits"])
        if cl.L > self.max_L:
            raise MemoryError(f"L={cl.L} exceeds max_L={self.max_L} (dense tables too large)")
        if self.levels == 0:
            Z = clique_level0(cl, self.level0)
        else:
            Z = clique_level1(cl, self.stats)
        return Z << n_free


def anthropic_count(cnf: CNF, **kw) -> int:
    return AnthropicCounter(**kw).count(cnf)
