"""Evaluate one trained run: alignment, prediction error ID/OOD, graph recovery, planning.

  python evaluate.py runs/<name> [--plan 1] [--threads 1]

Everything decoded to physical units goes through a ridge probe z -> true state fitted on the
run's OWN training latents, so all variants are compared in the same (physical) space.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

import env as E
from model import WorldModel
from train import ROOT, encode_all, get_dataset, observations

K = E.N_OBJ
NON_DISTRACTOR = [0, 1, 2, 3]


def load(run: str):
    cfg = json.load(open(os.path.join(run, "config.json")))
    m = WorldModel(cfg["regime"], cfg["variant"], d=cfg["d"], layers=cfg["layers"], seed=cfg["seed"])
    m.load_state_dict(torch.load(os.path.join(run, "model.pt")))
    m.eval()
    return m, cfg


def encode_states(model, states: np.ndarray, regime: str) -> torch.Tensor:
    """states (B, T+1, K, 4) -> latents for observations 1..T, (B, T, K, d)."""
    return encode_all(model, torch.tensor(observations(states, regime)))


# ----------------------------------------------------------------------------- probes / alignment
def fit_probe(model, regime):
    ds = get_dataset("train")
    S = ds["states"][:300]
    z = encode_states(model, S, regime).numpy()
    Z = z.reshape(-1, K * z.shape[-1])
    Y = S[:, 1:].reshape(len(Z), -1)
    probe = Ridge(alpha=1e-3).fit(Z, Y)
    # factor x object alignment: R^2 of object n's position from factor k alone
    R = np.zeros((K, K))
    zs = z.reshape(-1, K, z.shape[-1])
    pos = S[:, 1:, :, :2].reshape(-1, K, 2)
    n_fit = len(zs) * 2 // 3
    for k in range(K):
        for n in range(K):
            r = Ridge(alpha=1e-3).fit(zs[:n_fit, k], pos[:n_fit, n])
            R[k, n] = max(0.0, r2_score(pos[n_fit:, n], r.predict(zs[n_fit:, k])))
    rows, cols = linear_sum_assignment(-R)
    perm = np.empty(K, int)
    perm[cols] = rows            # perm[object n] = factor index matched to object n
    full_r2 = r2_score(Y, probe.predict(Z), multioutput="raw_values").reshape(K, 4)
    return probe, R, perm, full_r2


def decode(probe, z: torch.Tensor) -> np.ndarray:
    flat = z.reshape(-1, z.shape[-2] * z.shape[-1]).numpy()
    return probe.predict(flat).reshape(*z.shape[:-2], K, 4)


# ----------------------------------------------------------------------------- prediction error
@torch.no_grad()
def prediction_errors(model, cfg, probe, shift: str, H=(1, 5)):
    ds = get_dataset("test", shift)
    S, A = ds["states"], ds["actions"]
    z = encode_states(model, S, cfg["regime"])         # obs k <-> state k+1
    act = torch.tensor(A[:, 1:])
    st = S[:, 1:]
    T = z.shape[1]
    out = {}
    starts = np.arange(0, T - max(H), 3)
    for h in H:
        errs, copy, lat, latcopy, dec, deccopy = [], [], [], [], [], []
        for t0 in starts:
            zt = z[:, t0]
            zz = zt
            for k in range(h):
                zz = model.pred(zz, act[:, t0 + k])
                if shift == "do1" and cfg["regime"] == "state":
                    zz = zz.clone()
                    zz[:, 1] = z[:, t0 + k + 1, 1]          # intervened node re-encoded (teacher forced)
            pred = decode(probe, zz)
            true = st[:, t0 + h]
            tgt, src = decode(probe, z[:, t0 + h]), decode(probe, zt)
            dec.append(np.linalg.norm(pred[..., :2] - tgt[..., :2], axis=-1))
            deccopy.append(np.linalg.norm(src[..., :2] - tgt[..., :2], axis=-1))
            errs.append(np.linalg.norm(pred[..., :2] - true[..., :2], axis=-1))       # (B, K)
            copy.append(np.linalg.norm(st[:, t0, :, :2] - true[..., :2], axis=-1))
            lat.append(((zz - z[:, t0 + h]) ** 2).sum(-1).numpy())
            latcopy.append(((zt - z[:, t0 + h]) ** 2).sum(-1).numpy())
        e, c = np.concatenate(errs), np.concatenate(copy)
        l, lc = np.concatenate(lat), np.concatenate(latcopy)
        dd, dc = np.concatenate(dec), np.concatenate(deccopy)
        objs = [0, 2, 3] if shift == "do1" else NON_DISTRACTOR
        out[f"h{h}"] = dict(
            rmse_obj=np.sqrt((e ** 2).mean(0)).tolist(),
            nrmse_obj=(np.sqrt((e ** 2).mean(0)) / np.sqrt((c ** 2).mean(0) + 1e-12)).tolist(),
            nrmse_nd=float(np.sqrt((e[:, objs] ** 2).mean()) / np.sqrt((c[:, objs] ** 2).mean())),
            latent_nmse=float(l[:, objs].sum() / lc[:, objs].sum()),
            dec_rmse_obj=np.sqrt((dd ** 2).mean(0)).tolist(),
            dec_nrmse_obj=(np.sqrt((dd ** 2).mean(0)) / np.sqrt((dc ** 2).mean(0) + 1e-12)).tolist(),
            dec_nrmse=float(np.sqrt((dd[:, objs] ** 2).mean()) / np.sqrt((dc[:, objs] ** 2).mean())),
        )
    return out


# ----------------------------------------------------------------------------- graph recovery
def jacobian_graph(model, cfg, perm):
    """J[i, j] = E || d zhat_i / d z_j ||_F (objects ordered by matched factor), plus action column."""
    ds = get_dataset("test")
    z = encode_states(model, ds["states"][:50], cfg["regime"])[:, :-1].reshape(-1, K, cfg["d"])[:400]
    a = torch.tensor(ds["actions"][:50, 1:]).reshape(-1, 2)[:400]
    z = z[:, perm]                                     # reorder factors to object order
    inv = np.argsort(perm)
    J = np.zeros((K, K + 1))
    for b in range(0, len(z), 100):
        zb = z[b:b + 100].clone().requires_grad_(True)
        ab = a[b:b + 100].clone().requires_grad_(True)
        out = model.pred(zb[:, inv], ab)[:, perm]      # model in its own factor order
        for i in range(K):
            for c in range(cfg["d"]):
                gz, ga = torch.autograd.grad(out[:, i, c].sum(), (zb, ab), retain_graph=True)
                J[i, :K] += (gz ** 2).sum((0, 2)).numpy()
                J[i, K] += (ga ** 2).sum().item()
    J = np.sqrt(J / len(z))
    return J


@torch.no_grad()
def attention_graph(model, cfg, perm):
    ds = get_dataset("test")
    z = encode_states(model, ds["states"][:50], cfg["regime"])[:, :-1].reshape(-1, K, cfg["d"])[:400]
    a = torch.tensor(ds["actions"][:50, 1:]).reshape(-1, 2)[:400]
    model.pred(z, a)
    P = model.pred.last_attn[0].mean((0, 1)).numpy()[:K]      # (K, K+1) in factor order
    Pm = np.zeros_like(P)
    for i in range(K):
        for j in range(K):
            Pm[i, j] = P[perm[i], perm[j]]
        Pm[i, K] = P[perm[i], K]
    return Pm


def graph_scores(M: np.ndarray, kind: str):
    """M (K, K+1) edge scores in object order. Off-diagonal object->object entries scored against
    TRUE_PARENTS; the action column reported separately."""
    off = ~np.eye(K, dtype=bool)
    truth = E.TRUE_PARENTS[off]
    if kind == "jac":                        # scale-free: normalize each row by its self-dependence
        M = M / (np.diag(M[:, :K])[:, None] + 1e-12)
    s = M[:, :K][off]
    n_true = int(truth.sum())
    res = dict(auroc=float(roc_auc_score(truth, s)), auprc=float(average_precision_score(truth, s)))
    topk = np.zeros_like(s, dtype=int)
    topk[np.argsort(-s)[:n_true]] = 1
    res["shd_oraclek"] = int((topk != truth).sum())
    if kind == "gate":
        res["shd_thr0.5"] = int(((s > 0.5).astype(int) != truth).sum())
        res["n_edges_thr0.5"] = int((s > 0.5).sum())
        act = M[:, K]
        res["action_col"] = act.round(3).tolist()
        res["action_correct_thr0.5"] = bool(((act > 0.5).astype(int) == E.ACTION_PARENT).all())
    res["matrix"] = M.round(4).tolist()
    return res


# ----------------------------------------------------------------------------- planning
def cem_plan(cost_fn, n_env, H, rng, n_samples=128, iters=4, n_elite=20, init_mu=None):
    mu = np.zeros((n_env, H, 2)) if init_mu is None else init_mu
    sd = np.full((n_env, H, 2), 0.7)
    for _ in range(iters):
        acts = np.clip(mu[:, None] + sd[:, None] * rng.normal(size=(n_env, n_samples, H, 2)), -1, 1)
        c = cost_fn(acts)                                        # (n_env, n_samples)
        el = np.argsort(c, 1)[:, :n_elite]
        ea = np.take_along_axis(acts, el[:, :, None, None], 1)
        mu, sd = ea.mean(1), ea.std(1) + 0.05
    return mu


def plan_episodes(model, cfg, probe, shift="id", n_env=40, steps=75, H=15, replan=5, seed=123, method="model",
                  goal_steps=30):
    rng = np.random.default_rng(seed)
    params = E.make_params(shift, n_env, rng)
    s_prev = E.reset(n_env, rng, params)
    s = E.step(s_prev, np.zeros((n_env, 2)), rng, params)
    # goal = where object 2 ends up after a random smooth rollout from this start (reachable by construction,
    # as in LeWM / DINO-WM evaluation); only object 2's goal position is used in the cost.
    g_rng = np.random.default_rng(seed + 7)
    sg = s.copy()
    goal_actions = E.smooth_actions(n_env, goal_steps, g_rng)
    for t in range(goal_steps):
        sg = E.step(sg, goal_actions[:, t], g_rng, params)
    goal = sg[:, 2, :2].copy()
    if method == "model":
        W2 = probe.coef_.reshape(K, 4, -1)[2, :2]             # (2, K*d) -> object-2 position
        b2 = probe.intercept_.reshape(K, 4)[2, :2]
    det = dict(params, noise_v=0.0, drive_3=0.0, ou_sigma=0.0)
    w = np.linspace(0.2, 1.0, H)
    prng = np.random.default_rng(seed + 1)
    mu = None
    for t in range(0, steps, replan):
        if method == "random":
            plan = np.clip(prng.normal(0, 0.7, size=(n_env, H, 2)), -1, 1)
        else:
            if method == "model":
                with torch.no_grad():
                    obs = observations(np.stack([s_prev, s], 1), cfg["regime"])[:, 0]
                    z0 = model.enc(torch.tensor(obs, dtype=torch.float32))   # (n_env, K, d)

                def cost_fn(acts):
                    n = acts.shape[1]
                    zz = z0.repeat_interleave(n, 0)
                    a = torch.tensor(acts.reshape(-1, H, 2), dtype=torch.float32)
                    with torch.no_grad():
                        zs = model.rollout(zz, a).reshape(n_env, n, H, -1).numpy()
                    p2 = zs @ W2.T + b2
                    return ((((p2 - goal[:, None, None]) ** 2).sum(-1)) * w).sum(-1)
            else:   # oracle: true deterministic simulator
                def cost_fn(acts):
                    n = acts.shape[1]
                    ss = np.repeat(s, n, 0)
                    c = np.zeros(n_env * n)
                    p_det = dict(det)
                    if params.get("clamp_1") is not None:
                        p_det["clamp_1"] = np.repeat(params["clamp_1"], n, 0)
                    for h in range(H):
                        ss = E.step(ss, acts.reshape(-1, H, 2)[:, h], prng, p_det)
                        c += w[h] * ((ss[:, 2, :2] - np.repeat(goal, n, 0)) ** 2).sum(-1)
                    return c.reshape(n_env, n)
            init = None if mu is None else np.concatenate([mu[:, replan:], np.zeros((n_env, replan, 2))], 1)
            plan = cem_plan(cost_fn, n_env, H, prng, init_mu=init)
            mu = plan
        for k in range(replan):
            s_prev, s = s, E.step(s, plan[:, k], rng, params)
    d = np.linalg.norm(s[:, 2, :2] - goal, axis=-1)
    return dict(success=float((d < 0.1).mean()), success15=float((d < 0.15).mean()), final_dist=float(d.mean()), dists=d.round(4).tolist())


# ----------------------------------------------------------------------------- main
def evaluate(run: str, do_plan: bool = True):
    model, cfg = load(run)
    probe, R, perm, full_r2 = fit_probe(model, cfg["regime"])
    matched = R[perm, np.arange(K)]
    res = dict(name=cfg["name"], regime=cfg["regime"], variant=cfg["variant"], seed=cfg["seed"],
               frozen=bool(cfg.get("frozen_from")), lam_A=cfg["lam_A"], layers=cfg["layers"],
               alignment=dict(R=R.round(3).tolist(), perm=perm.tolist(), matched_r2=matched.round(3).tolist(),
                              min_matched_r2=float(matched.min()),
                              probe_r2_pos=full_r2[:, :2].mean(1).round(3).tolist(),
                              probe_r2_vel=full_r2[:, 2:].mean(1).round(3).tolist()))
    res["pred"] = {sh: prediction_errors(model, cfg, probe, sh) for sh in ["id", "physics", "distractor", "do1"]}
    res["graph_jac"] = graph_scores(jacobian_graph(model, cfg, perm), "jac")
    res["graph_attn"] = graph_scores(attention_graph(model, cfg, perm), "attn")
    if cfg["variant"] in ("soft", "hard", "mult"):
        G = model.pred.gate_probs().detach().numpy()
        Gm = np.zeros_like(G)
        for i in range(K):
            for j in range(K):
                Gm[i, j] = G[perm[i], perm[j]]
            Gm[i, K] = G[perm[i], K]
        res["graph_gate"] = graph_scores(Gm, "gate")
    if do_plan:
        res["plan"] = {sh: plan_episodes(model, cfg, probe, sh) for sh in ["id", "distractor"]}
    json.dump(res, open(os.path.join(run, "eval.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--plan", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    r = evaluate(args.run, bool(args.plan))
    print(json.dumps({k: v for k, v in r.items() if k in ("name", "alignment")}, indent=None)[:600])
    print("pred id h1", r["pred"]["id"]["h1"]["nrmse_nd"], "jac auroc", r["graph_jac"]["auroc"],
          "plan", {k: v["success"] for k, v in r.get("plan", {}).items()})
