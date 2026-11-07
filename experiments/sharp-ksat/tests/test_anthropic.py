import itertools

import numpy as np
import pytest

from src.anthropic_alg import (C_BLOCK, R_BLOCK, SIGMA, AnthropicCounter, Clique, c_col,
                               c_row, clique_level0, clique_level1, eight_square_holds,
                               factor_piece, n_blocks, pieces_dense)
from src.baselines import brute_force
from src.cnf import edge_case_formulas, pigeonhole

from .helpers import random_instances


def test_eight_square_identity_all_4096():
    assert eight_square_holds()


def test_eq28_cancellation():
    # σ(p,q)σ(p',q') + σ(p,q')σ(p',q) = 0 whenever p⊕q = p'⊕q' and q ≠ q'
    for p, q, p2, q2 in itertools.product(range(8), repeat=4):
        if p ^ q == p2 ^ q2 and q != q2:
            assert SIGMA[p, q] * SIGMA[p2, q2] + SIGMA[p, q2] * SIGMA[p2, q] == 0


def test_R_and_C_factor_through_3_bits():
    R = R_BLOCK.reshape(64, 64)                           # (x,y) x (u,v)
    C = C_BLOCK.transpose(0, 2, 1, 3).reshape(64, 64)     # (x,u) x (y,v)
    assert np.linalg.matrix_rank(R) <= 8
    assert np.linalg.matrix_rank(C) <= 8


@pytest.mark.parametrize("lam", [1, 2, 3, 4, 5])
def test_lemma2(lam):
    A = pieces_dense(lam)
    t = n_blocks(lam)
    N = 1 << lam
    eq = np.zeros((N,) * 4, dtype=np.int64)
    for x, y in itertools.product(range(N), repeat=2):
        eq[x, y, y, x] = 1
    assert np.array_equal(sum(A), (1 << t) * eq)                      # (a)
    for i in range(t + 1):
        assert np.abs(A[i]).max() <= 2 ** t
        for rw in (True, False):                                      # (b)
            F, G, cb = factor_piece(i, lam, rw)
            rec = np.einsum("xyk,uvk->xyuv", F, G) if rw else np.einsum("xuk,yvk->xyuv", F, G)
            assert np.array_equal(rec, A[i])
            assert cb == (c_row(i, lam) if rw else c_col(i, lam))
        assert c_row(i, lam) + c_col(i, lam) <= 4 * (11 / 12) * lam + 8  # (c)


@pytest.mark.parametrize("seed", range(12))
def test_one_level_equals_brute_force_on_random_signed_cliques(seed):
    rng = np.random.default_rng(seed)
    L = 1 + seed % 3
    tabs = [rng.integers(-4, 5, size=(1 << L,) * 3) for _ in range(4)]
    cl = Clique(4, L, tabs)
    b = clique_level0(cl, "brute")
    assert clique_level0(cl) == b
    assert clique_level1(cl) == b


@pytest.mark.parametrize("f,expected", edge_case_formulas())
def test_edge_cases(f, expected):
    for lv in (0, 1):
        assert AnthropicCounter(g=2, levels=lv).count(f) == expected


INST = random_instances(200, seed=31, n_max=20, m_max=9)


@pytest.mark.parametrize("i", range(len(INST)))
def test_formula_pipeline_matches_brute_force(i):
    f = INST[i]
    b = brute_force(f)
    assert AnthropicCounter(g=2, levels=0, seed=i).count(f) == b
    assert AnthropicCounter(g=2, levels=1, seed=i).count(f) == b


def test_unsat_structured():
    assert AnthropicCounter(g=2, levels=1).count(pigeonhole(2)) == 0
