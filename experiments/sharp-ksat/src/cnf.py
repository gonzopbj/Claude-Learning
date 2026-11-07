"""CNF formulas: representation, DIMACS I/O, generators.

A formula is a `CNF(n, clauses)` where clauses is a tuple of tuples of non-zero ints
(DIMACS literals: +v / -v for variable v in 1..n). Clauses are stored *raw*: duplicate
literals, tautologies, empty clauses and repeated clauses are all allowed, and every
counter in this package must handle them. The count is always over all 2^n assignments
of the n declared variables, including variables that occur in no clause.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CNF:
    n: int
    clauses: tuple

    def __post_init__(self):
        cl = tuple(tuple(int(l) for l in c) for c in self.clauses)
        object.__setattr__(self, "clauses", cl)
        for c in cl:
            for l in c:
                if l == 0 or abs(l) > self.n:
                    raise ValueError(f"literal {l} out of range for n={self.n}")

    @property
    def m(self) -> int:
        return len(self.clauses)

    def k(self) -> int:
        return max((len(set(c)) for c in self.clauses), default=0)


def normalize(cnf: CNF):
    """Drop duplicate literals within a clause, drop tautologies, drop repeated clauses.

    Returns (clauses, has_empty): `clauses` is a list of sorted tuples of distinct literals;
    `has_empty` is True when some clause is empty (formula unsatisfiable).
    """
    out, seen, has_empty = [], set(), False
    for c in cnf.clauses:
        s = set(c)
        if not s:
            has_empty = True
            continue
        if any(-l in s for l in s):
            continue  # tautology
        t = tuple(sorted(s, key=lambda l: (abs(l), l)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out, has_empty


# ----------------------------------------------------------------------------- DIMACS

def parse_dimacs(text: str) -> CNF:
    n = None
    lits: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c") or line.startswith("%"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1] != "cnf":
                raise ValueError(f"bad problem line: {line!r}")
            n = int(parts[2])
            continue
        lits.extend(int(tok) for tok in line.split())
    if n is None:
        raise ValueError("missing 'p cnf' line")
    clauses, cur = [], []
    for l in lits:
        if l == 0:
            clauses.append(tuple(cur))
            cur = []
        else:
            cur.append(l)
    if cur:
        clauses.append(tuple(cur))
    return CNF(n, tuple(clauses))


def to_dimacs(cnf: CNF) -> str:
    lines = [f"p cnf {cnf.n} {cnf.m}"]
    lines += [" ".join(map(str, c)) + " 0" for c in cnf.clauses]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- generators

def random_kcnf(n: int, m: int, k: int = 3, rng=None) -> CNF:
    """Uniform random k-CNF: each clause picks k distinct variables and random signs."""
    rng = np.random.default_rng(rng)
    if k > n:
        raise ValueError("k > n")
    clauses = []
    for _ in range(m):
        vs = rng.choice(n, size=k, replace=False) + 1
        signs = rng.integers(0, 2, size=k) * 2 - 1
        clauses.append(tuple(int(v * s) for v, s in zip(vs, signs)))
    return CNF(n, tuple(clauses))


def random_kcnf_ratio(n: int, ratio: float, k: int = 3, rng=None) -> CNF:
    return random_kcnf(n, int(round(ratio * n)), k, rng)


def planted_kcnf(n: int, m: int, k: int = 3, rng=None):
    """Planted k-CNF: sample a hidden assignment, keep only clauses it satisfies.

    Returns (cnf, hidden) where hidden is a 0/1 array of length n. The formula is
    guaranteed satisfiable. (Plain rejection sampling: clause distribution is uniform
    over the 2^k - 1 sign patterns consistent with the hidden assignment.)
    """
    rng = np.random.default_rng(rng)
    hidden = rng.integers(0, 2, size=n)
    clauses = []
    while len(clauses) < m:
        vs = rng.choice(n, size=k, replace=False)
        signs = rng.integers(0, 2, size=k)  # 1 = positive literal
        if not np.any(hidden[vs] == signs):
            continue
        clauses.append(tuple(int((v + 1) * (1 if s else -1)) for v, s in zip(vs, signs)))
    return CNF(n, tuple(clauses)), hidden


def pigeonhole(holes: int) -> CNF:
    """PHP: holes+1 pigeons into `holes` holes. Unsatisfiable. n = (holes+1)*holes."""
    p = holes + 1

    def var(i, j):  # pigeon i in hole j
        return i * holes + j + 1

    clauses = [tuple(var(i, j) for j in range(holes)) for i in range(p)]
    for j in range(holes):
        for i1, i2 in itertools.combinations(range(p), 2):
            clauses.append((-var(i1, j), -var(i2, j)))
    return CNF(p * holes, tuple(clauses))


def grid_independent_sets(w: int, h: int) -> CNF:
    """2-CNF whose models are the independent sets of the w x h grid graph.

    Structured, low treewidth (min(w,h)): a family where contraction-based methods should
    shine and where #models is known from transfer matrices.
    """
    def var(x, y):
        return y * w + x + 1

    clauses = []
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                clauses.append((-var(x, y), -var(x + 1, y)))
            if y + 1 < h:
                clauses.append((-var(x, y), -var(x, y + 1)))
    return CNF(w * h, tuple(clauses))


def edge_case_formulas():
    """Hand-written edge cases with their known counts."""
    return [
        (CNF(0, ()), 1),
        (CNF(3, ()), 8),
        (CNF(2, ((),)), 0),                      # empty clause
        (CNF(2, ((1, -1),)), 4),                 # tautology
        (CNF(2, ((1, 1, 1),)), 2),               # duplicate literals
        (CNF(2, ((1,), (-1,))), 0),              # contradictory units
        (CNF(3, ((1, 2), (1, 2), (2, 1))), 6),   # repeated clauses
        (CNF(4, ((1, -1, 2), (3,))), 8),         # tautology + unit, var 4 free
        (CNF(3, ((1, 2, 3),) * 2 + ((-1, -2, -3),)), 6),
        (CNF(5, ((), (1, 2))), 0),
    ]
