"""Run one (method, family, n, seed) instance; print one JSON line. Called by run_experiments."""
import json
import resource
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from src import baselines  # noqa: E402
from src.anthropic_alg import AnthropicCounter  # noqa: E402
from src.cnf import (grid_independent_sets, pigeonhole, planted_kcnf,  # noqa: E402
                     random_kcnf_ratio)
from src.openai_alg import OpenAICounter  # noqa: E402


def make_instance(family, n, seed):
    if family.startswith("rand"):
        ratio = float(family[4:])
        return random_kcnf_ratio(n, ratio, 3, seed * 1000 + n)
    if family == "planted":
        return planted_kcnf(n, int(round(4.26 * n)), 3, seed * 1000 + n)[0]
    if family == "php":          # n = h(h+1)
        h = n
        return pigeonhole(h)
    if family == "gridis":       # 3 x w grid, n = 3w
        return grid_independent_sets(n // 3, 3)
    raise ValueError(family)


def run(method, f):
    if method == "brute":
        return baselines.brute_force(f), {}
    if method == "dpll":
        return baselines.dpll_count_stats(f)
    if method == "tn":
        return baselines.tn_count_stats(f)
    if method == "openai":
        c = OpenAICounter(r=2, Q=2, R=1)
        return c.count(f), c.stats
    if method == "openai_R2":
        c = OpenAICounter(r=2, Q=3, R=2)
        return c.count(f), c.stats
    if method == "anthropic":
        c = AnthropicCounter(g=2, levels=1)
        return c.count(f), c.stats
    if method == "anthropic_l0":
        c = AnthropicCounter(g=2, levels=0)
        return c.count(f), c.stats
    raise ValueError(method)


if __name__ == "__main__":
    method, family, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    f = make_instance(family, n, seed)
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    status = "ok"
    try:
        count, stats = run(method, f)
    except MemoryError as e:
        count, stats, status = None, {"error": str(e)}, "oom"
    dt = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v))
             for k, v in stats.items()}
    print(json.dumps({"method": method, "family": family, "n": f.n, "param": n, "m": f.m,
                      "seed": seed, "status": status, "count": count, "time": dt,
                      "peak_rss_mb": rss / 1024, "base_rss_mb": base_rss / 1024, "stats": stats}))
