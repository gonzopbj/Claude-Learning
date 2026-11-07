"""OpenAI, "The Exponential Time Hypothesis is False" (Oct 2026), main-text algorithm.

What is implemented (paper §2.1–2.8 / §3–4), structure-faithful but with FREE parameters:

  1. Count = H(0), H(T) = Σ_x Π_e Φ_e(<P_{e,v}(T), x>)   (replacement linear forms, deg_T ≤ R-1)
  2. Replacement rows built on auxiliary coordinates B = ([r]x[r])^Q:
        M0 = V0 f(W0),  f interpolates (-1)^j on an interval I of R integers near Q(r-1)/(2r),
        measurement rows C = eigenvectors ψ of the swaps with j(ψ) ∉ I,
        <P_u(T), ζ> = ((I - M0) ζ)_u + Σ_w M0[u,w] Π_{s ∈ E_uw} (1 - T/s) ζ_w
  3. Schedules a_v, d_e ∈ [r]^Q put variables / clauses into groups at each point t = 1..Q;
     occurrences sharing a label go to different copies of B (h copies).
  4. At each point t: H(t + z) mod z^J by inclusion-exclusion over U ⊆ E, |E| ≤ q,
     each restricted sum Z_{t,E,U} evaluated by enumerating measurement tuples ȳ and then
     each group separately  (eq. 2.6 / 2.9).
  5. Hermite interpolation from the Taylor data at t = 1..Q recovers H(0).

Deviations from the paper (see notes/decisions.md):
  - arithmetic modulo 31-bit primes (+ CRT) instead of exact rationals;
  - r, Q, R are free knobs (the paper's (2.19) values only exist for astronomically large n);
  - greedy-balanced pseudo-random schedules instead of Lemma 4.5's derandomisation;
  - no sparsification, no §5-6 parity route, no Appendix D contraction theorem;
  - by default no Tovey copying (copy=False); copy=True applies the §2.8 reduction
    (OR-gate splitting of long clauses + implication-cycle copying).
  - enumeration of measurement tuples is over *reachable* values of each group's measurement
    vector (the paper enumerates the full box {-D..D}^c and gets B_g = 0 on unreachable ones;
    skipping those zero terms does not change the sum).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .cnf import CNF, normalize

PRIMES = (2147483647, 2147483629, 2147483587, 2147483579)


# ----------------------------------------------------------------------------- modular helpers

def _inv(a, p):
    return pow(int(a) % p, p - 2, p)


def _pmul(a, b, p):
    """Truncated polynomial product along the last axis, mod p (int64 safe for p < 2^31)."""
    J = a.shape[-1]
    shape = np.broadcast_shapes(a.shape, b.shape)
    c = np.zeros(shape, dtype=np.int64)
    for i in range(J):
        c[..., i:] = (c[..., i:] + a[..., i:i + 1] * b[..., : J - i]) % p
    return c


def _crt(residues, primes):
    x, mod = 0, 1
    for r, p in zip(residues, primes):
        # x ≡ x (mod mod), x ≡ r (mod p)
        k = ((r - x) * _inv(mod, p)) % p
        x += mod * k
        mod *= p
    return x, mod


def _solve_mod(A, b, p):
    """Solve A h = b mod p for an overdetermined consistent system; return h (list of ints)."""
    A = [list(map(int, row)) + [int(bb)] for row, bb in zip(A, b)]
    nrows, ncols = len(A), len(A[0]) - 1
    piv_cols, r = [], 0
    for c in range(ncols):
        pr = next((i for i in range(r, nrows) if A[i][c] % p), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        iv = _inv(A[r][c], p)
        A[r] = [(v * iv) % p for v in A[r]]
        for i in range(nrows):
            if i != r and A[i][c] % p:
                f = A[i][c]
                A[i] = [(vi - f * vr) % p for vi, vr in zip(A[i], A[r])]
        piv_cols.append(c)
        r += 1
    if r < ncols:
        raise ArithmeticError("Hermite system singular mod p")
    for i in range(r, nrows):
        if A[i][-1] % p:
            raise ArithmeticError("Hermite system inconsistent (bug: degree bound violated?)")
    h = [0] * ncols
    for i, c in enumerate(piv_cols):
        h[c] = A[i][-1]
    return h


# ----------------------------------------------------------------------------- §2.8 reduction

def reduce_bounded_occurrence(clauses, n_vars):
    """OR-gate splitting of clauses longer than 3, then Tovey copying (paper §2.8).

    clauses: list of tuples of distinct literals over variables 1..n_vars.
    Returns (clauses', n_vars') with every clause of width ≤ 3 and every variable occurring
    in ≤ 3 clauses; satisfying assignments are in bijection.
    """
    nxt = n_vars
    out = []
    for c in clauses:
        if len(c) <= 3:
            out.append(tuple(c))
            continue
        acc = c[0]
        for lit in c[1:]:
            nxt += 1
            z = nxt  # z = acc ∨ lit
            out += [(-acc, z), (-lit, z), (acc, lit, -z)]
            acc = z
        out.append((acc,))
    occ: dict = {}
    for ci, c in enumerate(out):
        for pos, l in enumerate(c):
            occ.setdefault(abs(l), []).append((ci, pos))
    new = [list(c) for c in out]
    cycles = []
    for v, lst in occ.items():
        s = len(lst)
        if s < 2:
            continue
        copies = []
        for ci, pos in lst:
            nxt += 1
            copies.append(nxt)
            l = new[ci][pos]
            new[ci][pos] = nxt if l > 0 else -nxt
        for i in range(s):
            cycles.append((-copies[i], copies[(i + 1) % s]))
    res = [tuple(c) for c in new] + cycles
    # relabel occurring variables to 1..N
    used = sorted({abs(l) for c in res for l in c})
    ren = {v: i + 1 for i, v in enumerate(used)}
    res = [tuple(ren[abs(l)] * (1 if l > 0 else -1) for l in c) for c in res]
    return res, len(used)


# ----------------------------------------------------------------------------- construction

@dataclass
class Design:
    r: int
    Q: int
    R: int
    I: list
    f_vals: list            # f(j) for j = 0..Q (mod p)
    kappa: list             # κ(a), a = 0..Q
    stats: dict = field(default_factory=dict)


def _lagrange_eval(xs, ys, x, p):
    tot = 0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if j != i:
                num = num * (x - xj) % p
                den = den * (xi - xj) % p
        tot = (tot + yi * num * _inv(den, p)) % p
    return tot


def _krawtchouk(j, a, Q):
    return sum((-1) ** i * math.comb(a, i) * math.comb(Q - a, j - i) for i in range(0, min(a, j) + 1))


def make_design(r, Q, R, p):
    if not (r >= 1 and Q >= 1 and 1 <= R <= Q + 1):
        raise ValueError("need r>=1, Q>=1, 1<=R<=Q+1")
    pf = (r - 1) / (2 * r)
    A0 = math.floor(Q * pf - (R - 1) / 2)
    A0 = max(0, min(A0, Q - R + 1))
    I = list(range(A0, A0 + R))
    f_vals = [_lagrange_eval(I, [(-1) ** j % p for j in I], j, p) for j in range(Q + 1)]
    inv2Q = _inv(pow(2, Q, p), p)
    kappa = []
    for a in range(Q + 1):
        s = sum(f_vals[j] * _krawtchouk(j, a, Q) for j in range(Q + 1)) % p
        kappa.append(s * inv2Q % p)
    for a in range(R, Q + 1):
        assert kappa[a] == 0, "κ(a) must vanish for a > deg f"
    return Design(r, Q, R, I, f_vals, kappa)


def _psi_basis(r):
    """Per-position eigenbasis of the swap on [r]x[r]: list of (minus?, {(a,d): coeff})."""
    out = []
    for a in range(r):
        out.append((0, {(a, a): 1}))
    for a in range(r):
        for b in range(a + 1, r):
            out.append((0, {(a, b): 1, (b, a): 1}))
            out.append((1, {(a, b): 1, (b, a): -1}))
    return out


class OpenAICounter:
    """Exact #k-SAT via the OpenAI replacement-form / Hermite-interpolation scheme."""

    def __init__(self, r=2, Q=2, R=1, copy=False, seed=0, max_q=None):
        self.r, self.Q, self.R = r, Q, R
        self.copy = copy
        self.seed = seed
        self.max_q = max_q
        self.stats: dict = {}

    # ---- public ------------------------------------------------------------------------
    def count(self, cnf: CNF) -> int:
        clauses, has_empty = normalize(cnf)
        self.stats = {"r": self.r, "Q": self.Q, "R": self.R, "copy": self.copy}
        if has_empty:
            return 0
        if not clauses:
            return 1 << cnf.n
        occ_vars = sorted({abs(l) for c in clauses for l in c})
        n_free = cnf.n - len(occ_vars)
        ren = {v: i + 1 for i, v in enumerate(occ_vars)}
        clauses = [tuple(ren[abs(l)] * (1 if l > 0 else -1) for l in c) for c in clauses]
        N = len(occ_vars)
        if self.copy:
            clauses, N = reduce_bounded_occurrence(clauses, N)
        self.stats.update(n_occ=len(occ_vars), n_internal=N, m_internal=len(clauses))
        bound_bits = len(occ_vars) + 1
        primes, mod = [], 1
        for p in PRIMES:
            primes.append(p)
            mod *= p
            if mod > (1 << bound_bits):
                break
        res = [self._count_mod(clauses, N, p) for p in primes]
        val, mod = _crt(res, primes)
        return val << n_free

    # ---- core --------------------------------------------------------------------------
    def _count_mod(self, clauses, N, p):
        r, Q, R = self.r, self.Q, self.R
        des = make_design(r, Q, R, p)
        rng = np.random.default_rng(self.seed)
        nvar = N
        m = len(clauses)
        # --- schedules (balanced pseudo-random) ---
        a = np.zeros((nvar, Q), dtype=np.int64)
        d = np.zeros((m, Q), dtype=np.int64)
        for s in range(Q):
            perm = rng.permutation(nvar)
            a[perm, s] = np.arange(nvar) % r
            perm = rng.permutation(m)
            d[perm, s] = np.arange(m) % r
        # --- occurrences, labels, copies ---
        occs = []  # (clause idx, var idx 0-based, positive?)
        for e, c in enumerate(clauses):
            for l in c:
                occs.append((e, abs(l) - 1, l > 0))
        labels = [tuple((int(a[v, s]), int(d[e, s])) for s in range(Q)) for (e, v, _) in occs]
        copy_of, mult = [], {}
        for lab in labels:
            j = mult.get(lab, 0)
            copy_of.append(j)
            mult[lab] = j + 1
        h = max(mult.values())
        # filled vectors: zeta[j][w] = variable index
        zeta = [dict() for _ in range(h)]
        for o, (lab, j) in enumerate(zip(labels, copy_of)):
            zeta[j][lab] = occs[o][1]
        # --- M0 rows for used labels ---
        Qs = list(range(Q))
        subsets = [T for k in range(0, R) for T in itertools.combinations(Qs, k)]

        def m0_row(u):
            row = {}
            for T in subsets:
                kap = des.kappa[len(T)]
                if kap == 0:
                    continue
                Tset = set(T)
                w = tuple(u[s] if s in Tset else (u[s][1], u[s][0]) for s in Qs)
                row[w] = (row.get(w, 0) + kap) % p
            return {w: c for w, c in row.items() if c}

        def phi_points(u, w):
            # E_{u,w} = {s : a_w(s) != d_u(s)} as 1-based points
            return [s + 1 for s in Qs if w[s][0] != u[s][1]]

        # --- measurement rows ψ (j(ψ) ∉ I) restricted to used labels ---
        basis = _psi_basis(r)
        Iset = set(des.I)
        used_labels = set(labels)
        meas = []  # (coef_psi, psi as dict label->±1 over used labels)
        for combo in itertools.product(range(len(basis)), repeat=Q):
            jpsi = sum(basis[b][0] for b in combo)
            if jpsi in Iset:
                continue
            norm2 = 1
            for b in combo:
                norm2 *= len(basis[b][1])
            vec = {}
            for lab in used_labels:
                val = 1
                for s, b in enumerate(combo):
                    val *= basis[b][1].get(lab[s], 0)
                    if val == 0:
                        break
                if val:
                    vec[lab] = val
            if not vec:
                continue
            fj = des.f_vals[jpsi]
            coef = (1 - ((-1) ** jpsi) * fj) * _inv(norm2, p) % p
            if coef == 0:
                continue
            meas.append((coef, vec))
        self.stats.update(D=r ** (2 * Q), h=h, n_meas_psi=len(meas), I=des.I)

        # --- per-occurrence form pieces ---
        # direct part: list of (var, coef, E-points) ; meas part: {(psi_idx, copy): coef}
        direct, mpart = [], []
        maxdeg = 0
        for o, (lab, j) in enumerate(zip(labels, copy_of)):
            row = m0_row(lab)
            dl = []
            for w, c in row.items():
                if w in zeta[j]:
                    pts = phi_points(lab, w)
                    maxdeg = max(maxdeg, len(pts))
                    dl.append((zeta[j][w], c, pts))
            direct.append(dl)
            mp = {}
            for pi, (coef, vec) in enumerate(meas):
                pu = vec.get(lab, 0)
                if pu:
                    mp[pi] = coef * pu % p
            mpart.append(mp)
        # --- degree bound, Taylor order ---
        M = len(occs)
        dH = M * maxdeg
        J = dH // Q + 1
        if self.max_q is not None and J - 1 > self.max_q:
            raise RuntimeError(f"q={J-1} exceeds max_q")
        npts = -(-(dH + 1) // J)
        q = J - 1
        self.stats.update(M=M, deg_bound=dH, J=J, q=q, points_used=npts,
                          C_q=sum(math.comb(nvar, s) * 2 ** s for s in range(min(q, nvar) + 1)))

        # sanity: forms restore X_v at T = 0
        self._check_T0(direct, mpart, meas, zeta, copy_of, occs, p)

        rows_b = []  # Taylor data per point: list of length J
        n_inner = 0
        n_combos = 0
        for t in range(1, npts + 1):
            Ht = np.zeros(J, dtype=np.int64)
            for es in range(0, min(q, nvar) + 1):
                for E in itertools.combinations(range(nvar), es):
                    for us in range(0, es + 1):
                        for U in itertools.combinations(E, us):
                            sign = -1 if (es - us) % 2 else 1
                            Z, nc = self._Z(t, E, set(U), J, p, a, d, clauses, occs, copy_of,
                                            direct, mpart, meas, zeta, nvar)
                            n_inner += 1
                            n_combos += nc
                            Ht = (Ht + sign * Z) % p
            rows_b.append(Ht)
        self.stats.update(n_restricted_sums=n_inner, n_measurement_tuples=n_combos)
        # --- Hermite interpolation ---
        A, b = [], []
        for t, Ht in zip(range(1, npts + 1), rows_b):
            for jj in range(J):
                A.append([(math.comb(k, jj) * pow(t, k - jj, p)) % p if k >= jj else 0
                          for k in range(dH + 1)])
                b.append(int(Ht[jj]))
        hvec = _solve_mod(A, b, p)
        return hvec[0] % p

    def _check_T0(self, direct, mpart, meas, zeta, copy_of, occs, p):
        for o in range(len(occs)):
            coefs = {}
            for v, c, _ in direct[o]:
                coefs[v] = (coefs.get(v, 0) + c) % p
            j = copy_of[o]
            for pi, c in mpart[o].items():
                for lab, sgn in meas[pi][1].items():
                    if lab in zeta[j]:
                        v = zeta[j][lab]
                        coefs[v] = (coefs.get(v, 0) + c * sgn) % p
            coefs = {v: c for v, c in coefs.items() if c}
            if coefs != {occs[o][1]: 1}:
                raise AssertionError(f"replacement form at T=0 is {coefs}, expected X_{occs[o][1]}")

    def _Z(self, t, E, U, J, p, a, d, clauses, occs, copy_of, direct, mpart, meas, zeta, nvar):
        """Z_{t,E,U}(z) mod (z^J, p) by measurement-tuple enumeration (eq. 2.6)."""
        r = self.r
        s_idx = t - 1
        groups = [np.flatnonzero(a[:, s_idx] == g) for g in range(r)]
        pos_in_group = {}
        for g, vs in enumerate(groups):
            for i, v in enumerate(vs):
                pos_in_group[int(v)] = (g, i)
        Eset = set(E)
        # measurement rows per group: key (psi, copy) -> coefficient vector over group vars
        rows = [dict() for _ in range(r)]
        for j in range(len(zeta)):
            for pi, (coef, vec) in enumerate(meas):
                for lab, sgn in vec.items():
                    v = zeta[j].get(lab)
                    if v is None:
                        continue
                    g, i = pos_in_group[v]
                    key = (pi, j)
                    rv = rows[g].setdefault(key, np.zeros(len(groups[g]), dtype=np.int64))
                    rv[i] += sgn
        row_keys = [list(rows[g].keys()) for g in range(r)]
        Cg = [np.array([rows[g][k] for k in row_keys[g]], dtype=np.int64).reshape(len(row_keys[g]), len(groups[g]))
              for g in range(r)]
        # clause -> group at t; per-occurrence local coefficient vector and shift polynomial
        occ_by_group = [[] for _ in range(r)]
        for o, (e, v, pos) in enumerate(occs):
            occ_by_group[int(d[e, s_idx])].append(o)
        tz = self._phi_shift_cache(t, J, p)
        local = {}
        shift = {}
        for o in range(len(occs)):
            e = occs[o][0]
            g = int(d[e, s_idx])
            loc = np.zeros(len(groups[g]), dtype=np.int64)
            sh = np.zeros(J, dtype=np.int64)
            for v, c, pts in direct[o]:
                val_t, poly = tz(tuple(pts))
                if val_t:
                    gv, i = pos_in_group[v]
                    assert gv == g, "design violated: free coordinate outside assigned group"
                    loc[i] = (loc[i] + c * val_t) % p
                if v in U:
                    sh = (sh + c * poly) % p
            local[o] = loc
            shift[o] = sh
        # enumerate each group's assignments (x_i = 1 on E)
        enum = []
        for g in range(r):
            vs = groups[g]
            free = [i for i, v in enumerate(vs) if int(v) not in Eset]
            k = len(free)
            X = np.zeros((1 << k, len(vs)), dtype=np.int64)
            for i, v in enumerate(vs):
                if int(v) in Eset:
                    X[:, i] = 1
            if k:
                bits = (np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1
                X[:, free] = bits
            Y = X @ Cg[g].T if Cg[g].size else np.zeros((len(X), 0), dtype=np.int64)
            keys, inv = np.unique(Y, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            buckets = [np.flatnonzero(inv == kk) for kk in range(len(keys))]
            enum.append((X, keys, buckets))
        # contributions of each group's measurement values to every occurrence's offset
        contrib = []
        for h_ in range(r):
            X, keys, _ = enum[h_]
            lam = np.zeros((len(occs), len(row_keys[h_])), dtype=np.int64)
            for ri, (pi, j) in enumerate(row_keys[h_]):
                for o in range(len(occs)):
                    if copy_of[o] == j:
                        c = mpart[o].get(pi)
                        if c:
                            lam[o, ri] = c
            kv = keys % p
            contrib.append((kv @ lam.T) % p if lam.size else np.zeros((len(keys), len(occs)), dtype=np.int64))
        # per-group precomputed local values: (A_g, |occ_g|)
        pre = []
        for g in range(r):
            X, keys, buckets = enum[g]
            og = occ_by_group[g]
            Lmat = np.array([local[o] for o in og], dtype=np.int64).reshape(len(og), X.shape[1])
            vals = (X @ Lmat.T) % p
            sh = np.array([shift[o] for o in og], dtype=np.int64).reshape(len(og), J)
            pre.append((og, vals, sh))
        clause_occ = {}
        for o, (e, v, pos) in enumerate(occs):
            clause_occ.setdefault(e, []).append((o, pos))
        total = np.zeros(J, dtype=np.int64)
        ncomb = 0
        one = np.zeros(J, dtype=np.int64)
        one[0] = 1
        for combo in itertools.product(*[range(len(enum[g][1])) for g in range(r)]):
            ncomb += 1
            off = np.zeros(len(occs), dtype=np.int64)
            for h_ in range(r):
                off = (off + contrib[h_][combo[h_]]) % p
            prod = one.copy()
            for g in range(r):
                og, vals, sh = pre[g]
                bidx = enum[g][2][combo[g]]
                nb = len(bidx)
                if not og:
                    Bg = one * (nb % p)
                    prod = _pmul(prod, Bg % p, p)
                    continue
                # values as polynomials: (nb, |og|, J)
                V = np.zeros((nb, len(og), J), dtype=np.int64)
                V[:, :, :] = sh[None, :, :]
                V[:, :, 0] = (V[:, :, 0] + vals[bidx] + off[og][None, :]) % p
                col = {o: i for i, o in enumerate(og)}
                acc = np.broadcast_to(one, (nb, J)).copy()
                es = sorted({occs[o][0] for o in og})
                for e in es:
                    notsat = np.broadcast_to(one, (nb, J)).copy()
                    for o, pos in clause_occ[e]:
                        L = V[:, col[o], :]
                        one_minus_L = (-L) % p if pos else L.copy()
                        if pos:
                            one_minus_L[:, 0] = (one_minus_L[:, 0] + 1) % p
                        notsat = _pmul(notsat, one_minus_L, p)
                    Phi = (-notsat) % p
                    Phi[:, 0] = (Phi[:, 0] + 1) % p
                    acc = _pmul(acc, Phi, p)
                Bg = acc.sum(axis=0) % p
                prod = _pmul(prod, Bg, p)
            total = (total + prod) % p
        return total, ncomb

    @staticmethod
    def _phi_shift_cache(t, J, p):
        cache = {}

        def f(pts):
            if pts in cache:
                return cache[pts]
            # φ(T) = Π_{s in pts} (1 - T/s); value at t and (φ(t+z) - φ(t)) mod z^J
            poly = np.zeros(J, dtype=np.int64)
            poly[0] = 1
            for s in pts:
                inv_s = _inv(s, p)
                fac = np.zeros(J, dtype=np.int64)
                fac[0] = (1 - t * inv_s) % p
                if J > 1:
                    fac[1] = (-inv_s) % p
                poly = _pmul(poly, fac, p)
            val = int(poly[0])
            dpoly = poly.copy()
            dpoly[0] = 0
            cache[pts] = (val, dpoly)
            return cache[pts]

        return f


def openai_count(cnf: CNF, **kw) -> int:
    return OpenAICounter(**kw).count(cnf)
