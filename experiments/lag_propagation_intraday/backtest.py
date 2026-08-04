"""
Walk-forward "trade the lag" backtest, intraday version.

Same mechanics as the daily experiment (experiments/lag_propagation), with
two changes driven by what we learned there:

1. Bar frequency and all the windows/lags/annualization factors are
   parameters, not day-denominated constants, so this runs at 1-hour or
   1-minute resolution.
2. Confidence gating: `apply_confidence_gate` drops a target's position to
   flat for any rebalance period where the selected driver/lag's trailing
   |corr| falls below a threshold, instead of always trading every target
   uniformly regardless of how weak its best pick was. The daily experiment
   traded every signal it found, diluting real edges with noise; this tests
   whether being selective recovers some of that.
"""
import numpy as np
import pandas as pd


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


def _select_best(mats: np.ndarray, target_idx: int, n: int):
    col = mats[:, :, target_idx].copy()  # (max_lag, n)
    col[:, target_idx] = np.nan
    if np.all(np.isnan(col)):
        return None
    flat_idx = np.nanargmax(np.abs(col))
    k_i, a_i = np.unravel_index(flat_idx, col.shape)
    return int(a_i), int(k_i + 1), float(col[k_i, a_i])


def compute_rebalance_windows(rets_df: pd.DataFrame, assets: list[str], tradable_assets: list[str],
                               window: int, refresh: int, max_lag: int):
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


def select_assignments(records, tradable_idx: list[int], n_assets: int,
                        mode: str = "best", rng: np.random.Generator | None = None):
    out = []
    for rec in records:
        mats = rec["mats"]
        assignment = {}
        for ti in tradable_idx:
            if mode == "random":
                col = mats[:, :, ti].copy()
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


def apply_confidence_gate(records, min_abs_corr: float):
    """Drop any (target) assignment whose |corr| is below the threshold --
    that target sits out (flat) for the period instead of trading a
    low-confidence pick. Returns a new list; does not mutate `records`."""
    out = []
    for rec in records:
        gated = {ti: v for ti, v in rec["assignment"].items() if abs(v[2]) >= min_abs_corr}
        out.append({**rec, "assignment": gated})
    return out


def pick_confidence_distribution(records) -> np.ndarray:
    """All |corr| values actually picked across the whole walk-forward run --
    used to set gating thresholds from the empirical distribution rather than
    an arbitrary absolute number."""
    vals = [abs(v[2]) for rec in records for v in rec["assignment"].values()]
    return np.array(vals)


def signals_from_rebalances(records, R: np.ndarray, tradable_idx: list[int]) -> np.ndarray:
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
            src_rows = np.arange(s - lag, e - lag)
            valid = src_rows >= 0
            vals = np.zeros(e - s)
            vals[valid] = R[src_rows[valid], driver] / sd
            vals = np.clip(vals, -3, 3)
            raw[s:e, j] = np.sign(corr) * vals
    return raw


def _event_driven_block(driver_ret: np.ndarray, sd: float, corr: float, lag: int, entry_z: float) -> np.ndarray:
    """One rebalance block, one target. Fires a fixed-size directional bet only
    when the driver's z-scored return crosses `entry_z`, holds it for exactly
    `lag` bars (the discovered propagation window), then flattens -- instead of
    continuously resizing the position every bar off the driver's latest return.
    A new trigger while a position is already open overwrites it (most recent
    information wins).

    Pure-numpy forward-fill via cummax on indices -- no pandas Series
    construction per block, which matters when this runs for ~1000 rebalances
    x 3 targets per backtest call.
    """
    n = len(driver_ret)
    z = driver_ret / sd
    trigger = np.abs(z) >= entry_z
    trigger_val = np.where(trigger, np.sign(corr) * np.sign(driver_ret), 0.0)

    idx = np.arange(n)
    last_trig_idx = np.maximum.accumulate(np.where(trigger, idx, 0))
    last_trig_val = trigger_val[last_trig_idx]

    age = idx - last_trig_idx
    valid = (age >= 1) & (age <= lag)
    return np.where(valid, last_trig_val, 0.0)


def signals_from_rebalances_event_driven(records, R: np.ndarray, tradable_idx: list[int],
                                          entry_z: float) -> np.ndarray:
    """Event-driven counterpart to `signals_from_rebalances`: same driver/lag/corr
    assignments (from `select_assignments` / `apply_confidence_gate`), but positions
    are discrete trigger-and-hold bets instead of a continuously reweighted signal."""
    T, n = R.shape
    ntar = len(tradable_idx)
    raw = np.zeros((T, ntar))
    for rec in records:
        s, e = rec["start"], rec["end"]
        win_std = rec["window"].std(axis=0, ddof=1)
        for j, ti in enumerate(tradable_idx):
            if ti not in rec["assignment"]:
                continue
            driver, lag, corr = rec["assignment"][ti]
            sd = win_std[driver]
            if not (sd > 0):
                continue
            raw[s:e, j] = _event_driven_block(R[s:e, driver], sd, corr, lag, entry_z)
    return raw


def raw_signal_to_portfolio(raw: np.ndarray, target_rets: np.ndarray, annualization: float,
                             cost_bps: float, target_vol_annual: float = 0.15,
                             vol_lookback: int = 500, max_leverage: float = 3.0) -> pd.DataFrame:
    T, ntar = raw.shape
    gross = np.abs(raw).sum(axis=1)
    weight = np.divide(raw, gross[:, None], out=np.zeros_like(raw), where=gross[:, None] > 0)

    port_raw_ret = np.zeros(T)
    port_raw_ret[1:] = (weight[:-1] * target_rets[1:]).sum(axis=1)

    # rolling std of port_raw_ret, evaluated one bar back (no lookahead), vectorized
    # -- a plain Python per-bar loop here is O(T) numpy calls and doesn't scale to
    # T ~ 10^6 minute bars.
    rolling_std = pd.Series(port_raw_ret).rolling(vol_lookback).std(ddof=1).shift(1).to_numpy()
    realized = rolling_std * np.sqrt(annualization)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(realized > 1e-8, np.minimum(target_vol_annual / realized, max_leverage), 1.0)
    scale = np.nan_to_num(scale, nan=1.0)

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


def momentum_baseline(rets_df: pd.DataFrame, tradable_assets: list[str], lookback: int, **kwargs) -> pd.DataFrame:
    R = rets_df[tradable_assets].to_numpy()
    cum = pd.DataFrame(R, columns=tradable_assets).rolling(lookback).sum().to_numpy()
    raw = np.sign(np.nan_to_num(cum, nan=0.0))
    return raw_signal_to_portfolio(raw, R, **kwargs)


def buy_hold_baseline(rets_df: pd.DataFrame, tradable_assets: list[str], cost_bps: float) -> pd.DataFrame:
    R = rets_df[tradable_assets].to_numpy()
    T, n = R.shape
    w = np.full(n, 1.0 / n)
    ret = R @ w
    turnover = np.zeros(T)
    turnover[0] = 1.0
    costs = turnover * cost_bps
    return pd.DataFrame({"strategy_return": ret - costs, "gross_exposure": np.ones(T),
                          "n_active": np.full(T, n), "turnover": turnover, "scale": np.ones(T)})


def performance_metrics(strategy_return: pd.Series, annualization: float) -> dict:
    r = strategy_return.dropna()
    ann_ret = r.mean() * annualization
    ann_vol = r.std(ddof=1) * np.sqrt(annualization)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    hit_rate = (r[r != 0] > 0).mean() if (r != 0).any() else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "n_bars": int(len(r)),
    }
