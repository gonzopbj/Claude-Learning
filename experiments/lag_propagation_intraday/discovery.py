"""
Full-sample lag discovery: for every ordered (driver, target) asset pair and
lag k in 1..max_lag bars, test whether driver's return this bar predicts
target's return k bars later, beyond what's explained by chance / multiple
testing. Bar unit (minute/hour) is whatever `rets` is sampled at -- pass the
matching `max_lag` from run.py.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05  # FDR level


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int):
    """corr(x_t, y_{t+k}) using only the overlapping, aligned samples."""
    xa = x[: len(x) - k]
    yb = y[k:]
    mask = ~(np.isnan(xa) | np.isnan(yb))
    xa, yb = xa[mask], yb[mask]
    if len(xa) < 30:
        return np.nan, np.nan, len(xa)
    r, p = stats.pearsonr(xa, yb)
    return r, p, len(xa)


def full_sample_scan(rets: pd.DataFrame, assets: list[str], max_lag: int) -> pd.DataFrame:
    """Scan all ordered pairs x all lags, full sample. Returns a long DataFrame."""
    cols = {a: rets[a].to_numpy() for a in assets}
    rows = []
    for driver, target in itertools.permutations(assets, 2):
        x, y = cols[driver], cols[target]
        for k in range(1, max_lag + 1):
            r, p, n = lagged_corr(x, y, k)
            rows.append((driver, target, k, r, p, n))
    df = pd.DataFrame(rows, columns=["driver", "target", "lag", "corr", "pvalue", "n"])
    return df


def add_fdr(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    df = df.dropna(subset=["pvalue"]).copy()
    reject, qval, _, _ = multipletests(df["pvalue"].to_numpy(), alpha=alpha, method="fdr_bh")
    df["qvalue"] = qval
    df["significant"] = reject
    return df


def summarize(df_sig: pd.DataFrame, n_pairs: int, n_tests: int) -> dict:
    """Aggregate the scan into the headline numbers for the report."""
    sig = df_sig[df_sig["significant"]]
    best_per_pair = (
        sig.assign(abscorr=sig["corr"].abs())
        .sort_values("abscorr", ascending=False)
        .drop_duplicates(subset=["driver", "target"])
    )
    n_pairs_significant = best_per_pair.shape[0]

    return {
        "n_ordered_pairs": n_pairs,
        "n_pair_lag_tests": n_tests,
        "fdr_alpha": ALPHA,
        "n_significant_pair_lag_tests": int(sig.shape[0]),
        "n_pairs_with_any_significant_lag": int(n_pairs_significant),
        "pct_pairs_with_any_significant_lag": round(100 * n_pairs_significant / n_pairs, 2),
        "lag_distribution_bars": best_per_pair["lag"].value_counts().sort_index().to_dict(),
        "median_best_lag_bars": float(best_per_pair["lag"].median()) if n_pairs_significant else None,
        "mean_abs_corr_significant": float(best_per_pair["corr"].abs().mean()) if n_pairs_significant else None,
        "all_significant_pairs": best_per_pair.sort_values("abscorr", ascending=False)
        [["driver", "target", "lag", "corr", "qvalue"]]
        .to_dict("records"),
    }
