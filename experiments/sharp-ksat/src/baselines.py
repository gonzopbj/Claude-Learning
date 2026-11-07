"""Baseline exact model counters.

- `brute_force`      : vectorised enumeration of all 2^n assignments (numpy, chunked).
- `dpll_count`       : DPLL #SAT with unit propagation, connected components and a
                       component cache (a small "sharpSAT-lite").
- `tn_count`         : the formula as a tensor network (clause tensors, variables as
                       shared/hyper indices), contracted with an opt_einsum path.

Each returns an int. The `*_stats` variants also return a dict of intermediate quantities.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from .cnf import CNF, normalize

# ----------------------------------------------------------------------------- brute force


def brute_force(cnf: CNF, chunk_bits: int = 20) -> int:
    clauses, has_empty = normalize(cnf)
    if has_empty:
        return 0
    n = cnf.n
    if not clauses:
        return 1 << n
    # only enumerate occurring variables; free ones multiply by 2
    occ = sorted({abs(l) for c in clauses for l in c})
    idx = {v: i for i, v in enumerate(occ)}
    k = len(occ)
    total = 0
    cb = min(chunk_bits, k)
    base = np.arange(1 << cb, dtype=np.int64)
    low_bits = [((base >> i) & 1).astype(bool) for i in range(cb)]
    for hi in range(1 << (k - cb)):
        ok = np.ones(1 << cb, dtype=bool)
        for c in clauses:
            sat = np.zeros(1 << cb, dtype=bool)
            for l in c:
                i = idx[abs(l)]
                if i < cb:
                    sat |= low_bits[i] if l > 0 else ~low_bits[i]
                elif ((hi >> (i - cb)) & 1) == (l > 0):
                    sat[:] = True
                    break
            ok &= sat
            if not ok.any():
                break
        total += int(ok.sum())
    return total << (n - k)


# ----------------------------------------------------------------------------- DPLL #SAT


class _DPLL:
    def __init__(self):
        self.cache: dict = {}
        self.decisions = 0
        self.cache_hits = 0
        self.max_depth = 0

    @staticmethod
    def _assign(clauses, lit):
        out = []
        for c in clauses:
            if lit in c:
                continue
            if -lit in c:
                c = tuple(l for l in c if l != -lit)
                if not c:
                    return None
            out.append(c)
        return out

    @staticmethod
    def _components(clauses):
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for c in clauses:
            vs = [abs(l) for l in c]
            for v in vs:
                parent.setdefault(v, v)
            r0 = find(vs[0])
            for v in vs[1:]:
                r = find(v)
                if r != r0:
                    parent[r] = r0
        groups: dict = {}
        for c in clauses:
            groups.setdefault(find(abs(c[0])), []).append(c)
        return list(groups.values())

    def _branch(self, clauses, depth):
        cnt = Counter(abs(l) for c in clauses for l in c)
        v = cnt.most_common(1)[0][0]
        allv = set(cnt)
        total = 0
        for lit in (v, -v):
            self.decisions += 1
            sub = self._assign(clauses, lit)
            if sub is None:
                continue
            sub, _ = self._propagate_tracking(sub)
            if sub is None:
                continue
            remaining = {abs(l) for c in sub for l in c}
            forced = self._last_forced
            free = len(allv) - 1 - len(forced) - len(remaining)
            total += self._count_components(sub, depth + 1) << free
        return total

    def _propagate_tracking(self, clauses):
        forced = set()
        while True:
            unit = next((c[0] for c in clauses if len(c) == 1), None)
            if unit is None:
                self._last_forced = forced
                return clauses, forced
            forced.add(abs(unit))
            clauses = self._assign(clauses, unit)
            if clauses is None:
                self._last_forced = forced
                return None, forced

    def _count_components(self, clauses, depth):
        self.max_depth = max(self.max_depth, depth)
        if not clauses:
            return 1
        result = 1
        for comp in self._components(clauses):
            key = frozenset(comp)
            r = self.cache.get(key)
            if r is None:
                r = self._branch(comp, depth)
                self.cache[key] = r
            else:
                self.cache_hits += 1
            if r == 0:
                return 0
            result *= r
        return result


def dpll_count_stats(cnf: CNF):
    clauses, has_empty = normalize(cnf)
    st = {"decisions": 0, "cache_hits": 0, "max_depth": 0, "cache_size": 0}
    if has_empty:
        return 0, st
    solver = _DPLL()
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10000))
    try:
        sub, forced = solver._propagate_tracking(clauses)
        if sub is None:
            res = 0
        else:
            remaining = {abs(l) for c in sub for l in c}
            free = cnf.n - len(forced) - len(remaining)
            res = solver._count_components(sub, 0) << free
    finally:
        sys.setrecursionlimit(old)
    st.update(decisions=solver.decisions, cache_hits=solver.cache_hits,
              max_depth=solver.max_depth, cache_size=len(solver.cache))
    return res, st


def dpll_count(cnf: CNF) -> int:
    return dpll_count_stats(cnf)[0]


# ----------------------------------------------------------------------------- tensor network


def clause_tensor(lits):
    """Indicator tensor of a clause over its (distinct) literals, shape (2,)*k."""
    k = len(lits)
    t = np.ones((2,) * k, dtype=np.float64)
    falsifying = tuple(0 if l > 0 else 1 for l in lits)
    t[falsifying] = 0.0
    return t


def tn_network(cnf: CNF):
    """Return (tensors, index_lists, n_free) for the formula as a hyper-index tensor network."""
    clauses, has_empty = normalize(cnf)
    if has_empty:
        return None, None, 0
    occ = sorted({abs(l) for c in clauses for l in c})
    tensors = [clause_tensor(c) for c in clauses]
    inds = [[abs(l) for l in c] for c in clauses]
    return tensors, inds, cnf.n - len(occ)


def tn_count_stats(cnf: CNF, optimize="auto"):
    import opt_einsum as oe

    tensors, inds, n_free = tn_network(cnf)
    st = {"log2_flops": 0.0, "width": 0.0}
    if tensors is None:
        return 0, st
    if not tensors:
        return 1 << n_free, st
    occ = sorted({v for ix in inds for v in ix})
    if len(occ) > 52:
        dtype = object  # exactness beyond 2^53
    else:
        dtype = None
    sym = {v: oe.get_symbol(i) for i, v in enumerate(occ)}
    eq = ",".join("".join(sym[v] for v in ix) for ix in inds) + "->"
    path, info = oe.contract_path(eq, *tensors, optimize=optimize)
    st["log2_flops"] = float(np.log2(max(float(info.opt_cost), 1.0)))
    st["width"] = float(np.log2(max(float(info.largest_intermediate), 1.0)))
    if dtype is object:
        tensors = [t.astype(object) for t in tensors]
    val = oe.contract(eq, *tensors, optimize=path)
    val = int(round(float(val))) if dtype is None else int(val)
    return val << n_free, st


def tn_count(cnf: CNF) -> int:
    return tn_count_stats(cnf)[0]
