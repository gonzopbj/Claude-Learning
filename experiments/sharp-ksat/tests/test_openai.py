import itertools

import numpy as np
import pytest

from src.baselines import brute_force, dpll_count
from src.cnf import CNF, edge_case_formulas, pigeonhole, random_kcnf
from src.openai_alg import (OpenAICounter, _krawtchouk, _psi_basis, make_design,
                            reduce_bounded_occurrence)

P = 2147483647
PARAMS = [dict(r=2, Q=2, R=1), dict(r=3, Q=1, R=1), dict(r=2, Q=3, R=2), dict(r=3, Q=2, R=2)]


@pytest.mark.parametrize("r,Q,R", [(2, 2, 1), (2, 3, 2), (3, 2, 2), (2, 4, 3), (3, 3, 2)])
def test_M0_fixes_kernel_of_C(r, Q, R):
    """(2.14): M0 ψ = ψ for every eigenvector ψ with j(ψ) ∈ I (these span ker C)."""
    des = make_design(r, Q, R, P)
    basis = _psi_basis(r)
    # M0 = Σ_T κ(|T|) Π_{[Q]\T}; Π_s acts on a basis vector by its eigenvalue (-1)^{minus}
    for combo in itertools.product(range(len(basis)), repeat=Q):
        signs = [(-1) ** basis[b][0] for b in combo]
        val = 0
        for k in range(Q + 1):
            for T in itertools.combinations(range(Q), k):
                ev = 1
                for s in range(Q):
                    if s not in T:
                        ev *= signs[s]
                val = (val + des.kappa[k] * ev) % P
        j = sum(basis[b][0] for b in combo)
        expected = ((-1) ** j * des.f_vals[j]) % P
        assert val == expected
        if j in des.I:
            assert val == 1


def test_kappa_vanishes_above_degree():
    for r, Q, R in [(2, 5, 3), (3, 4, 2), (4, 6, 4)]:
        des = make_design(r, Q, R, P)
        assert all(des.kappa[a] == 0 for a in range(R, Q + 1))


def test_krawtchouk_orthogonality_small():
    Q = 4
    for a in range(Q + 1):
        assert sum(_krawtchouk(j, a, Q) for j in range(Q + 1)) == (2 ** Q if a == 0 else 0)


def test_reduction_preserves_count_and_bounds():
    from collections import Counter
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(2, 8))
        k = int(rng.integers(2, min(n, 5) + 1))
        f = random_kcnf(n, int(rng.integers(1, 10)), k, rng)
        cl = [tuple(c) for c in f.clauses]
        red, N = reduce_bounded_occurrence(cl, n)
        assert max(len(c) for c in red) <= 3
        assert max(Counter(abs(l) for c in red for l in c).values()) <= 3
        # variables of f that occur nowhere are absent from `red`
        occ = len({abs(l) for c in cl for l in c})
        assert dpll_count(CNF(N, tuple(red))) << (n - occ) == brute_force(f)


@pytest.mark.parametrize("f,expected", edge_case_formulas())
def test_edge_cases(f, expected):
    for p in PARAMS:
        assert OpenAICounter(**p).count(f) == expected


def _instances(count, seed, max_occ, m_max):
    from .helpers import random_instances
    return random_instances(count, seed, n_max=20, m_max=m_max, max_occ=max_occ)


R1 = _instances(200, seed=21, max_occ=10, m_max=30)


@pytest.mark.parametrize("i", range(len(R1)))
def test_matches_brute_force_R1(i):
    f = R1[i]
    p = PARAMS[i % 2]
    assert OpenAICounter(**p, seed=i).count(f) == brute_force(f)


R2 = _instances(40, seed=22, max_occ=6, m_max=5)


@pytest.mark.parametrize("i", range(len(R2)))
def test_matches_brute_force_hermite(i):
    """R = 2: nontrivial Taylor data + Hermite interpolation + inclusion-exclusion over (E, U)."""
    f = R2[i]
    p = PARAMS[2 + i % 2]
    assert OpenAICounter(**p, seed=i).count(f) == brute_force(f)


R3 = _instances(20, seed=23, max_occ=4, m_max=3)


@pytest.mark.parametrize("i", range(len(R3)))
def test_matches_brute_force_faithful_copy(i):
    """copy=True: the §2.8 reduction (OR gates + Tovey cycles) feeding the same routine."""
    f = R3[i]
    assert OpenAICounter(r=2, Q=2, R=1, copy=True, seed=i).count(f) == brute_force(f)


def test_unsat_structured():
    assert OpenAICounter(r=2, Q=2, R=1).count(pigeonhole(2)) == 0
