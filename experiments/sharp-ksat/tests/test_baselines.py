import pytest

from src.baselines import brute_force, dpll_count, tn_count
from src.cnf import CNF, edge_case_formulas, grid_independent_sets, pigeonhole

from .helpers import random_instances

INSTANCES = random_instances(300, seed=11, n_max=20)


@pytest.mark.parametrize("f,expected", edge_case_formulas())
def test_edge_cases(f, expected):
    assert brute_force(f) == expected
    assert dpll_count(f) == expected
    assert tn_count(f) == expected


def test_brute_force_tiny_exhaustive():
    # every clause set over 2 variables of width <= 2, against a direct count
    import itertools
    lits = [1, -1, 2, -2]
    clauses = [c for k in (1, 2) for c in itertools.combinations(lits, k)]
    for mask in range(1 << len(clauses)):
        cl = tuple(c for i, c in enumerate(clauses) if mask >> i & 1)
        f = CNF(2, cl)
        direct = sum(all(any((x >> (abs(l) - 1) & 1) == (l > 0) for l in c) for c in cl)
                     for x in range(4))
        assert brute_force(f) == direct


@pytest.mark.parametrize("i", range(len(INSTANCES)))
def test_dpll_and_tn_match_brute_force(i):
    f = INSTANCES[i]
    b = brute_force(f)
    assert dpll_count(f) == b
    assert tn_count(f) == b


def test_structured():
    assert dpll_count(pigeonhole(4)) == 0
    assert tn_count(pigeonhole(3)) == 0
    g = grid_independent_sets(4, 4)
    assert dpll_count(g) == tn_count(g) == brute_force(g) == 1234  # OEIS A006506
