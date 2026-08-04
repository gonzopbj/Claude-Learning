"""
Walk-forward "trade the lag" backtest.

At each rebalance (every REFRESH trading days), for every tradable target
asset we scan all other assets (drivers) at lags 1..MAX_LAG using only the
trailing WINDOW days of returns (no lookahead), and pick the driver/lag with
the strongest trailing correlation. Between rebalances we trade that target
off the chosen driver's realized (already-public) return, lagged by the
chosen number of days.

Two honesty checks are built in:
  1. A time-series-momentum baseline (each asset traded off its own trailing
     return) -- the bar the cross-asset lag signal actually has to clear.
  2. A permutation-null control: same mechanics, but the driver/lag is drawn
     uniformly at random from the same candidate pool instead of picked by
     argmax. If the real strategy's Sharpe doesn't clearly separate from this
     null distribution, the "edge" is indistinguishable from what scanning
     ~230 candidates per target per rebalance will produce by luck alone
     (classic multiple-testing / data-snooping bias).
"""
import numpy as np
import pandas as pd

MAX_LAG = 10
WINDOW = 252
REFRESH = 21
COST_BPS = 0.0003
TARGET_VOL_ANNUAL = 0.08
VOL_LOOKBACK = 63
MAX_LEVERAGE = 3.0
ANNUALIZATION = 252
MOM_LOOKBACK = 21


def _lagged_corr_matrices(window: np.ndarray, max_lag: int) -> np.ndarray:
    """window: (W, n) returns. Returns (max_lag, n, n): mats[k-1, a, b] = corr(a_t, b_{t+k})."""
    W, n = window.shape
    mats = np.full((max_lag, n, n), np.nan)
    for k in range(1, max_lag + 1):
        early = window[: W - k, :]
        late = window[k:, :]
        e_mu, e_sd = early.mean(0), early.std(0, ddof=1)
        l_mu, l_sd = late.mean(0), late.std(0, ddof=1)
        e_sd = np.where(e_sd == 0, np.nan, e_sd)
        l_sd = np.where(l_sd == 0, np.nan, l_sd)
        ez = (early - e_mu) / e_sd
        lz = (late - l_mu) / l_sd
        N = W - k
        mats[k - 1] = (ez.T @ lz) / (N - 1)
    return mats


def _select_best(mats: np.ndarray, target_idx: int, n: int, exclude=None):
    """Return (driver_idx, lag, corr) maximizing |corr| for this target, driver != target."""
    col = mats[:, :, target_idx].copy()  # (max_lag, n)
    col[:, target_idx] = np.nan
    if exclude is not None:
        col[:, exclude] = np.nan
    if np.all(np.isnan(col)):
        return None
    flat_idx = np.nanargmax(np.abs(col))
    k_i, a_i = np.unravel_index(flat_idx, col.shape)
    return int(a_i), int(k_i + 1), float(col[k_i, a_i])


def compute_rebalance_windows(rets_df: pd.DataFrame, assets: list[str], tradable_assets: list[str],
                               window: int = WINDOW, refresh: int = REFRESH, max_lag: int = MAX_LAG):
    """Walk forward through the sample computing the lagged-correlation matrices at
    each rebalance date (using only the trailing `window` days -- no lookahead).
    This is the expensive, selection-independent part; call `select_assignments`
    on the result to get either the best-pick strategy or a random-pick null draw
    without recomputing correlations each time."""
    R = rets_df[assets].to_numpy()
    T, n = R.shape
    asset_idx = {a: i for i, a in enumerate(assets)}
    tradable_idx = [asset_idx[a] for a in tradable_assets]

    records = []
    for reb in range(window, T, refresh):
        end = min(reb + refresh, T)
        win = R[reb - window: reb, :]
        mats = _lagged_corr_matrices(win, max_lag)
        records.append({"start": reb, "end": end, "window": win, "mats": mats})
    return records, R, tradable_idx


def select_assignments(records, tradable_idx: list[int], n_assets: int, max_lag: int = MAX_LAG,
                        mode: str = "best", rng: np.random.Generator | None = None):
    """mode='best': argmax|corr| driver/lag per target (the actual strategy).
    mode='random': uniformly random driver/lag from the same candidate pool per
    target (the permutation-null control). Returns a new list of records with
    an 'assignment' key added; does not mutate `records`."""
    out = []
    for rec in records:
        mats = rec["mats"]
        assignment = {}
        for ti in tradable_idx:
            if mode == "random":
                col = mats[:, :, ti].copy()  # (max_lag, n_assets)
                col[:, ti] = np.nan
                valid_flat = np.flatnonzero(~np.isnan(col))
                if valid_flat.size == 0:
                    continue
                pick = valid_flat[rng.integers(valid_flat.size)]
                k_i, a_i = np.unravel_index(pick, col.shape)
                corr = float(col[k_i, a_i])
                assignment[ti] = (int(a_i), int(k_i + 1), corr)
            else:
                best = _select_best(mats, ti, n_assets)
                if best is not None:
                    assignment[ti] = best
        out.append({"start": rec["start"], "end": rec["end"], "window": rec["window"], "assignment": assignment})
    return out


def signals_from_rebalances(records, R: np.ndarray, tradable_idx: list[int], max_lag: int = MAX_LAG) -> np.ndarray:
    """Build the (T, n_tradable) raw signal matrix from a rebalance table."""
    T, n = R.shape
    ntar = len(tradable_idx)
    raw = np.zeros((T, ntar))
    for rec in records:
        s, e = rec["start"], rec["end"]
        win = rec["window"]
        win_std = win.std(axis=0, ddof=1)
        for j, ti in enumerate(tradable_idx):
            if ti not in rec["assignment"]:
                continue
            driver, lag, corr = rec["assignment"][ti]
            sd = win_std[driver] if win_std[driver] > 0 else np.nan
            if np.isnan(sd):
                continue
            # driver return at t-lag, for every t in [s, e)
            src_rows = np.arange(s - lag, e - lag)
            valid = src_rows >= 0
            vals = np.zeros(e - s)
            vals[valid] = R[src_rows[valid], driver] / sd
            vals = np.clip(vals, -3, 3)
            raw[s:e, j] = np.sign(corr) * vals
    return raw


def raw_signal_to_portfolio(raw: np.ndarray, target_rets: np.ndarray, cost_bps: float = COST_BPS,
                             target_vol_annual: float = TARGET_VOL_ANNUAL, vol_lookback: int = VOL_LOOKBACK,
                             max_leverage: float = MAX_LEVERAGE) -> pd.DataFrame:
    """raw: (T, ntar) raw per-target signal. target_rets: (T, ntar) same-day returns
    (row t = the return realized going INTO day t, i.e. aligned so that raw[t-1]
    is the position held going into target_rets[t])."""
    T, ntar = raw.shape
    gross = np.abs(raw).sum(axis=1)
    weight = np.divide(raw, gross[:, None], out=np.zeros_like(raw), where=gross[:, None] > 0)

    port_raw_ret = np.zeros(T)
    port_raw_ret[1:] = (weight[:-1] * target_rets[1:]).sum(axis=1)

    scale = np.ones(T)
    for t in range(vol_lookback + 1, T):
        realized = port_raw_ret[t - vol_lookback:t].std(ddof=1) * np.sqrt(ANNUALIZATION)
        if realized > 1e-8:
            scale[t] = min(target_vol_annual / realized, max_leverage)

    scaled_weight = weight * scale[:, None]
    turnover = np.abs(np.diff(scaled_weight, axis=0, prepend=np.zeros((1, ntar)))).sum(axis=1)
    costs = turnover * cost_bps

    strat_ret = np.zeros(T)
    strat_ret[1:] = (scaled_weight[:-1] * target_rets[1:]).sum(axis=1) - costs[1:]

    return pd.DataFrame({
        "strategy_return": strat_ret,
        "gross_exposure": gross,
        "n_active": (raw != 0).sum(axis=1),
        "turnover": turnover,
        "scale": scale,
    })


def momentum_baseline(rets_df: pd.DataFrame, tradable_assets: list[str], lookback: int = MOM_LOOKBACK,
                       **kwargs) -> pd.DataFrame:
    R = rets_df[tradable_assets].to_numpy()
    T, n = R.shape
    cum = pd.DataFrame(R, columns=tradable_assets).rolling(lookback).sum().to_numpy()
    raw = np.sign(np.nan_to_num(cum, nan=0.0))
    return raw_signal_to_portfolio(raw, R, **kwargs)


def buy_hold_baseline(rets_df: pd.DataFrame, tradable_assets: list[str]) -> pd.DataFrame:
    R = rets_df[tradable_assets].to_numpy()
    T, n = R.shape
    w = np.full(n, 1.0 / n)
    ret = R @ w
    turnover = np.zeros(T)
    turnover[0] = 1.0
    costs = turnover * COST_BPS
    return pd.DataFrame({"strategy_return": ret - costs, "gross_exposure": np.ones(T),
                          "n_active": np.full(T, n), "turnover": turnover, "scale": np.ones(T)})


def performance_metrics(strategy_return: pd.Series) -> dict:
    r = strategy_return.dropna()
    ann_ret = r.mean() * ANNUALIZATION
    ann_vol = r.std(ddof=1) * np.sqrt(ANNUALIZATION)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    hit_rate = (r > 0).mean()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "n_days": int(len(r)),
    }
