import numpy as np

from src.baselines import brute_force
from src.cnf import (CNF, grid_independent_sets, normalize, parse_dimacs, pigeonhole,
                     planted_kcnf, random_kcnf, to_dimacs)


def test_dimacs_roundtrip():
    f = random_kcnf(10, 30, 3, 0)
    g = parse_dimacs("c comment\n" + to_dimacs(f))
    assert g == f


def test_dimacs_multiline_clause():
    g = parse_dimacs("p cnf 3 2\n1 -2\n3 0 -1 0\n")
    assert g.clauses == ((1, -2, 3), (-1,))


def test_random_kcnf_distinct_vars():
    f = random_kcnf(8, 100, 3, 1)
    assert all(len({abs(l) for l in c}) == 3 for c in f.clauses)


def test_planted_is_satisfied_by_hidden():
    for s in range(20):
        f, hid = planted_kcnf(12, 60, 3, s)
        for c in f.clauses:
            assert any((hid[abs(l) - 1] == 1) == (l > 0) for l in c)


def test_pigeonhole_unsat():
    for h in (1, 2, 3):
        assert brute_force(pigeonhole(h)) == 0


def test_grid_independent_sets_path_is_fibonacci():
    fib = [1, 2]
    for _ in range(12):
        fib.append(fib[-1] + fib[-2])
    for w in range(1, 10):
        assert brute_force(grid_independent_sets(w, 1)) == fib[w]


def test_normalize():
    cl, empty = normalize(CNF(3, ((1, 1, 2), (2, 1), (1, -1), ())))
    assert cl == [(1, 2)] and empty
