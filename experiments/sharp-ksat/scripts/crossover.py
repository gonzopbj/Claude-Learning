"""Theoretical crossover analysis from the bounds as stated in the two papers.

(A) Hidden-constant bounds: T(n) = 2^{C * g(n)}. For each C, find the smallest n with
    C*g(n) < n*log2(b) for b in {2 (brute force), 1.6423 (best #3-SAT, Kutzkov), 1.3 (PPSZ-type
    decision)}. Polynomial factors are ignored, which favours the new algorithms.
(B) Explicit schedules:
    - Anthropic Theorem 2 recursion (eq. 38): K0=64, e0=1/2, K_{l+1}=9K_l^16,
      e_{l+1}=(11/12+2/K_l)e_l+2/K_l. Exponent ≈ 6 e_l m (U ≈ 6m bits counted at both ends)
      plus an additive log2(K^g) = 3K^8 log2 K for trying all groupings.
    - OpenAI explicit parameters (2.19) and the three costs (2.11), for bounded-occurrence
      formulas after Tovey copying (N ≈ 3m copied variables for a 3-CNF with m clauses).
Writes results/crossover.json and figures/crossover_*.png.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG2E = 1 / math.log(2)
BASES = {"2^n (brute force)": 1.0, "1.6423^n (best #3-SAT)": math.log2(1.6423),
         "1.3^n (PPSZ-type SAT)": math.log2(1.3)}


def solve_log2n(per_var_exp, target, lo=1.0, hi=1e300):
    """Smallest x = log2 n (bisection in log space) with per_var_exp(x) < target, assuming
    per_var_exp decreasing for large x. Returns None if not reached below hi."""
    if per_var_exp(hi) >= target:
        return None
    if per_var_exp(lo) < target:
        return lo
    for _ in range(400):
        mid = math.sqrt(lo * hi) if hi / lo > 4 else (lo + hi) / 2
        if per_var_exp(mid) < target:
            hi = mid
        else:
            lo = mid
    return hi


def bounds(k=3, r=4.26):
    # each returns the exponent per variable, as a function of x = log2 n (natural logs where the
    # paper uses natural logs: OpenAI uses ln, Anthropic log2)
    def anth_m(C):   # 2^{C k m / sqrt(log2 m)}, m = r n
        return lambda x: C * k * r / math.sqrt(x + math.log2(r))

    def anth_n(C):   # 2^{C k^2 n log2log2 n / sqrt(log2 n)}
        return lambda x: C * k * k * math.log2(max(x, 2)) / math.sqrt(x)

    def oai_sharp(C):  # 2^{C n ln ln n / sqrt(ln n)}
        return lambda x: C * math.log(max(x / LOG2E, math.e)) / math.sqrt(x / LOG2E)

    def oai_dense(C):  # m <= Dn: 2^{C n / sqrt(ln n)}
        return lambda x: C / math.sqrt(x / LOG2E)

    def oai_lean(C):   # Lean: 2^{C n (ln ln n)^2 / sqrt(ln n)}
        return lambda x: C * math.log(max(x / LOG2E, math.e)) ** 2 / math.sqrt(x / LOG2E)

    return {"Anthropic, m-bound (k=3, m=4.26n)": anth_m, "Anthropic, n-bound (k=3)": anth_n,
            "OpenAI, sharp n-bound": oai_sharp, "OpenAI, m<=Dn bound": oai_dense,
            "OpenAI, Lean-verified bound": oai_lean}


def anthropic_schedule(levels=40):
    K, e = 64.0, 0.5
    out = []
    log2K = 6.0
    for l in range(levels):
        out.append({"level": l, "log2_K": log2K, "e": e})
        Kinv = 2.0 ** (-log2K) if log2K < 1000 else 0.0
        e = (11 / 12 + 2 * Kinv) * e + 2 * Kinv
        log2K = math.log2(9) + 16 * log2K
    return out


def openai_explicit(lam):
    """Three-cost exponent per (copied) variable at λ = log2 n with the (2.19) parameters.
    Returns (valid, f) where f = log2(cost)/n."""
    nu = math.ceil(math.log2(lam))
    r = math.floor(math.sqrt(lam) / nu ** 2)
    Q = math.floor(lam / (128 * nu))
    R = 4 * math.ceil(math.sqrt(lam))
    valid = r >= 2 and R >= 32 and 6 * R <= Q
    if r < 1 or Q < 1:
        return False, float("inf"), dict(r=r, Q=Q, R=R)
    tau = 1 / (r * lam)
    log2D = 2 * Q * math.log2(max(r, 1))
    qn = min(3 * R / Q, 1.0)          # q / n
    taylor = qn * math.log2(2 * math.e / qn) if qn < 1 else math.log2(3)
    meas = 2 * tau * (log2D + 1)       # log2(2D+1) ≈ log2 D + 1
    groups = 2 / r
    return valid, taylor + meas + groups, dict(r=r, Q=Q, R=R, taylor=taylor, meas=meas, groups=groups)


def main():
    res = {"hidden_constant": {}, "anthropic_schedule": [], "openai_explicit": []}
    Cs = np.logspace(-2, 1.5, 60)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (bname, target) in zip(axes, BASES.items()):
        for name, mk in bounds().items():
            ys = []
            for C in Cs:
                x = solve_log2n(mk(C), target)
                ys.append(math.log10(x) if x else np.nan)
            res["hidden_constant"].setdefault(name, {})[bname] = {
                f"{C:.3g}": (10 ** y if not np.isnan(y) else None) for C, y in zip(Cs, ys)}
            ax.plot(Cs, ys, label=name)
        ax.set_xscale("log")
        ax.set_title(f"beat {bname}")
        ax.set_xlabel("hidden constant C in the exponent")
        ax.axhline(math.log10(math.log2(1e80)), ls=":", c="k", lw=1)
        ax.text(Cs[0], math.log10(math.log2(1e80)) + 0.05, "n = 10^80", fontsize=7)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("log10( log2 n* )   (crossover n*)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Crossover vs hidden constant (polynomial factors ignored, which favours the new algorithms)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "crossover_sensitivity.png"), dpi=140)

    # explicit schedules
    sched = anthropic_schedule()
    res["anthropic_schedule"] = sched
    oai = []
    for lg in np.linspace(4, 60, 400):
        lam = 2.0 ** lg
        v, f, info = openai_explicit(lam)
        oai.append({"log2_lambda": lg, "valid": v, "f": f, **info})
    res["openai_explicit"] = oai[::8]
    # crossovers for explicit OpenAI params at density r (N ≈ 3 r n copied variables)
    expl = {}
    for dens in (1.0, 4.26):
        for bname, target in BASES.items():
            hit = next((o for o in oai if o["valid"] and o["f"] * 3 * dens < target), None)
            expl[f"density {dens}, {bname}"] = None if hit is None else hit["log2_lambda"]
    res["openai_explicit_crossover_log2_log2_n"] = expl
    # Anthropic Thm-2: levels needed so that 6 e_l r < target
    an = {}
    for dens in (1.0, 4.26):
        for bname, target in BASES.items():
            hit = next((s for s in sched if 6 * s["e"] * dens < target), None)
            an[f"density {dens}, {bname}"] = None if hit is None else {
                "level": hit["level"], "log2_K": hit["log2_K"],
                "log10_log2_overhead": math.log10(3 * 8 * hit["log2_K"] / 8) + 8 * hit["log2_K"] * math.log10(2)}
    res["anthropic_thm2_levels"] = an
    json.dump(res, open(os.path.join(ROOT, "results", "crossover.json"), "w"), indent=1, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ls = [s["level"] for s in sched[:35]]
    ax.semilogy(ls, [6 * s["e"] for s in sched[:35]], "o-", ms=3, label="6·e_ℓ (exponent per clause)")
    for dens, c in ((1.0, "C1"), (4.26, "C2")):
        ax.axhline(1 / dens, ls="--", c=c, lw=1, label=f"beats 2^n at m={dens}n below this")
    ax.set_xlabel("recursion level ℓ")
    ax.set_ylabel("exponent per clause")
    ax2 = ax.twinx()
    ax2.plot(ls, [math.log10(s["log2_K"]) for s in sched[:35]], "k:", lw=1)
    ax2.set_ylabel("log10(log2 K_ℓ)  (dotted)")
    ax.set_title("Anthropic Thm 2 schedule (eq. 38)")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    xs = [o["log2_lambda"] for o in oai if np.isfinite(o["f"])]
    ax.plot(xs, [o["f"] for o in oai if np.isfinite(o["f"])], label="three-cost exponent per copied variable")
    ax.plot(xs, [o["groups"] for o in oai if np.isfinite(o["f"])], ":", label="  group term 2/r")
    ax.plot(xs, [o["taylor"] for o in oai if np.isfinite(o["f"])], ":", label="  Taylor term")
    vx = [o["log2_lambda"] for o in oai if o["valid"]]
    if vx:
        ax.axvline(min(vx), c="k", ls="--", lw=1, label="(2.19) parameters valid from here")
    ax.axhline(1 / (3 * 4.26), c="C3", ls="--", lw=1, label="needed to beat 2^n at m=4.26n")
    ax.set_yscale("log")
    ax.set_xlabel("log2 λ = log2 log2 n")
    ax.set_title("OpenAI explicit parameters (2.19), costs (2.11)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "crossover_explicit.png"), dpi=140)
    print(json.dumps({k: res[k] for k in ("openai_explicit_crossover_log2_log2_n", "anthropic_thm2_levels")}, indent=1))
    for name in res["hidden_constant"]:
        for b in BASES:
            d = res["hidden_constant"][name][b]
            pick = {c: d[c] for c in list(d)[::12]}
            print(name, "|", b, "|", {c: (f"log2 n*≈{v:.3g}" if v else None) for c, v in pick.items()})


if __name__ == "__main__":
    main()
