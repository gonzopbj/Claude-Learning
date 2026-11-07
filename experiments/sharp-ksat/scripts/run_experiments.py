"""Scale n per (method, family) until the median instance exceeds the budget.

Each instance runs in a fresh subprocess with a wall-clock timeout. Results are appended
to results/runs.jsonl (one JSON object per instance). Counts are cross-checked across
methods at the end.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLAN = {
    # family: list of n (or family parameter) values, in increasing order
    "rand2.0": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 100, 120, 140, 160, 200, 250, 300],
    "rand4.26": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200],
    "rand6.0": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 100, 120, 140, 160, 200, 250],
    "planted": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200],
    "php": [1, 2, 3, 4, 5, 6, 7, 8],           # holes h, n = h(h+1)
    "gridis": [6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90, 120, 180, 240, 300],
}
METHODS = ["brute", "dpll", "tn", "openai", "anthropic", "anthropic_l0", "openai_R2"]


def run_one(method, family, n, seed, timeout):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "worker.py"), method, family, str(n), str(seed)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return {"method": method, "family": family, "param": n, "seed": seed,
                    "status": "error", "time": None, "err": out.stderr[-500:]}
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"method": method, "family": family, "param": n, "seed": seed,
                "status": "timeout", "time": timeout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--families", nargs="*", default=list(PLAN))
    ap.add_argument("--stop_frac", type=float, default=0.25,
                    help="stop scaling once the median time exceeds stop_frac * timeout")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as fh:
        for fam in a.families:
            for meth in a.methods:
                reps = 3 if fam in ("php", "gridis") else a.instances  # deterministic families
                for n in PLAN[fam]:
                    res = list(ex.map(lambda s: run_one(meth, fam, n, s, a.timeout), range(reps)))
                    for r in res:
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    times = [r["time"] if r["status"] == "ok" else float("inf") for r in res]
                    med = statistics.median(times)
                    nbad = sum(r["status"] != "ok" for r in res)
                    print(f"{fam:9s} {meth:13s} n={n:4d} median={med:9.3f}s fails={nbad}", flush=True)
                    if med > a.stop_frac * a.timeout or nbad > len(res) // 2:
                        break


if __name__ == "__main__":
    main()
