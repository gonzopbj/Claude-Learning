import numpy as np

from src.cnf import CNF, normalize, random_kcnf


def random_instances(count, seed, n_max=20, m_max=None, k_choices=(2, 3, 4), max_occ=None):
    """Random formulas with n ≤ n_max covering k = 2, 3, 4, sat and unsat, plus raw-clause noise
    (duplicate literals, tautologies, repeated clauses, occasional empty clause)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        n = int(rng.integers(1, n_max + 1))
        k = int(rng.choice([kk for kk in k_choices if kk <= n] or [n]))
        mm = m_max if m_max is not None else 5 * n
        m = int(rng.integers(0, mm + 1))
        f = random_kcnf(n, m, k, rng)
        cl = list(f.clauses)
        r = rng.random()
        if cl and r < 0.1:
            c = cl[0]
            cl[0] = c + (c[0],)                       # duplicate literal
        elif cl and r < 0.2:
            v = int(rng.integers(1, n + 1))
            cl.append((v, -v))                        # tautology
        elif cl and r < 0.3:
            cl.append(cl[-1])                         # repeated clause
        elif r < 0.33:
            cl.append(())                             # empty clause
        f = CNF(n, tuple(cl))
        if max_occ is not None:
            occ = {abs(l) for c in normalize(f)[0] for l in c}
            if len(occ) > max_occ:
                continue
        out.append(f)
    return out
