"""Train one world model.

  python train.py --regime state --variant soft --seed 0
  python train.py --regime pixel --variant dense --seed 0                      # end-to-end
  python train.py --regime pixel --variant hard --seed 0 --frozen_from pixel_dense_L1_lA0.02_s0   # stage 2

Loss (LeWM-style, no stop-gradient, no EMA):
  mean_t ||rollout(z_0, a_0..a_{t-1})_t - z_t||^2  +  lam_sig * SIGReg(z)  +  lam_A * mean sigma(A_offdiag)
  (+ lam_ent * attention entropy for dense_ent)
SIGReg: state regime -> per factor (each factor's d dims; the encoder is per-object so a joint
isotropy target would fight the true cross-object correlation); pixel regime -> the full K*d
vector per time step, as in LeWM.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from env import render, rollout
from model import WorldModel

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


# ----------------------------------------------------------------------------- data
def get_dataset(split: str, shift: str = "id", B: int | None = None, T: int | None = None) -> dict:
    """Cached rollouts. states (B, T+1, 5, 4), actions (B, T, 2)."""
    os.makedirs(DATA, exist_ok=True)
    defaults = {"train": (1000, 50, 0), "val": (100, 50, 1), "test": (200, 40, 2)}
    b, t, base = defaults[split]
    B, T = B or b, T or t
    seed = base * 100 + {"id": 0, "physics": 1, "distractor": 2, "do1": 3, "coupled": 4, "decoupled": 5}[shift]
    path = os.path.join(DATA, f"{split}_{shift}_{B}x{T}.npz")
    if not os.path.exists(path):
        S, A = rollout(B, T, np.random.default_rng(seed), shift)
        np.savez_compressed(path, states=S.astype(np.float32), actions=A.astype(np.float32))
    d = np.load(path)
    return {"states": d["states"], "actions": d["actions"]}


def observations(states: np.ndarray, regime: str) -> np.ndarray:
    """Observation at index t uses frames t-1, t (pixel). Index 0 is dropped for both regimes so
    that observation k corresponds to state k+1."""
    if regime == "state":
        return states[:, 1:]
    frames = render(states)                                    # (B, T+1, 3, R, R)
    return np.concatenate([frames[:, :-1], frames[:, 1:]], 2)  # (B, T, 6, R, R)


class Windows:
    """Random windows of length W+1 observations with the W actions between them."""

    def __init__(self, ds: dict, regime: str, W: int):
        self.obs = torch.tensor(observations(ds["states"], regime))
        self.act = torch.tensor(ds["actions"][:, 1:])           # action k maps obs k -> obs k+1
        self.st = torch.tensor(ds["states"][:, 1:])
        self.W = W
        self.B, self.T = self.obs.shape[:2]

    def sample(self, n: int, g: torch.Generator):
        b = torch.randint(0, self.B, (n,), generator=g)
        t = torch.randint(0, self.T - self.W, (n,), generator=g)
        idx = t[:, None] + torch.arange(self.W + 1)
        return (self.obs[b[:, None], idx], self.act[b[:, None], idx[:, :-1]], self.st[b[:, None], idx])


# ----------------------------------------------------------------------------- loss
def loss_terms(model: WorldModel, obs, act, cfg, frozen=False):
    if frozen:
        z = obs                                                 # precomputed latents
    else:
        z = model.enc(obs)                                      # (B, W+1, K, d)
    zhat = model.rollout(z[:, 0], act)                          # (B, W, K, d)
    pred = ((zhat - z[:, 1:]) ** 2).mean()
    if frozen:
        sig = torch.zeros(())
    elif model.regime == "state":
        t = torch.randint(0, z.shape[1], ()).item()             # one random time step (compute)
        sig = model.sigreg(z[:, t].transpose(0, 1))             # (K, B, d): per factor
    else:
        t = torch.randint(0, z.shape[1], ()).item()
        sig = model.sigreg(z[:, t].flatten(-2))                 # (B, K*d)
    sp = model.pred.sparsity()
    ent = model.pred.attn_entropy() if cfg["variant"] == "dense_ent" else torch.zeros(())
    idm = torch.zeros(())
    if cfg.get("lam_idm", 0) > 0 and not frozen:
        # inverse dynamics (EB-JEPA): predict a_t from (z_t, z_{t+1}); forces action-relevant content
        pair = torch.cat([z[:, :-1].flatten(-2), z[:, 1:].flatten(-2)], -1)
        idm = ((model.idm(pair) - act) ** 2).mean()
    total = pred + cfg["lam_sig"] * sig + cfg["lam_A"] * sp + cfg["lam_ent"] * ent + cfg.get("lam_idm", 0) * idm
    return total, dict(pred=pred.item(), sig=sig.item(), sparsity=sp.item(), ent=ent.item(), idm=idm.item())


@torch.no_grad()
def encode_all(model: WorldModel, obs: torch.Tensor, bs: int = 512) -> torch.Tensor:
    flat = obs.reshape(-1, *obs.shape[2:])
    out = torch.cat([model.enc(flat[i:i + bs]) for i in range(0, len(flat), bs)])
    return out.reshape(*obs.shape[:2], *out.shape[1:])


def train(cfg: dict) -> str:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    out = os.path.join(ROOT, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)

    model = WorldModel(cfg["regime"], cfg["variant"], d=cfg["d"], layers=cfg["layers"], seed=cfg["seed"])
    if cfg.get("lam_idm", 0) > 0:
        D = 2 * model.pred.K * cfg["d"]
        model.idm = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
    frozen = cfg.get("frozen_from")
    env_tr = cfg.get("train_env", "id")
    ds = get_dataset("train", env_tr)
    if cfg.get("n_train"):
        ds = {k: v[:cfg["n_train"]] for k, v in ds.items()}      # low-data arm: first n episodes
    tr = Windows(ds, cfg["regime"], cfg["W"])
    va = Windows(get_dataset("val", env_tr), cfg["regime"], cfg["W"])
    if frozen:
        src = torch.load(os.path.join(ROOT, "runs", frozen, "model.pt"))
        enc_state = {k[4:]: v for k, v in src.items() if k.startswith("enc.")}
        model.enc.load_state_dict(enc_state)
        for p in model.enc.parameters():
            p.requires_grad_(False)
        tr.obs, va.obs = encode_all(model, tr.obs), encode_all(model, va.obs)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    g = torch.Generator().manual_seed(cfg["seed"])
    gv = torch.Generator().manual_seed(999)
    log, t0 = [], time.time()
    for step in range(cfg["steps"]):
        model.train()
        obs, act, _ = tr.sample(cfg["bs"], g)
        loss, terms = loss_terms(model, obs, act, cfg, frozen=bool(frozen))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0 or step == cfg["steps"] - 1:
            model.eval()
            with torch.no_grad():
                vo, vact, _ = va.sample(1024, gv)
                gv.manual_seed(999)
                _, vterms = loss_terms(model, vo, vact, cfg, frozen=bool(frozen))
            rec = dict(step=step, time=round(time.time() - t0, 1), **{f"tr_{k}": v for k, v in terms.items()},
                       **{f"va_{k}": v for k, v in vterms.items()})
            log.append(rec)
            if cfg["verbose"]:
                print(json.dumps(rec), flush=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"))
    return out


DEFAULTS = dict(regime="state", variant="dense", seed=0, d=16, layers=1, W=3, bs=256, lr=2e-3, wd=1e-3,
                steps=4000, lam_sig=0.09, lam_A=0.02, lam_ent=0.01, threads=1, log_every=500,
                frozen_from=None, name=None, verbose=1, lam_idm=0.0, n_train=0, train_env="id")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    cfg = vars(ap.parse_args())
    if cfg["name"] is None:
        tag = "_fz" if cfg["frozen_from"] else ""
        tag += "_idm" if cfg["lam_idm"] > 0 else ""
        tag += f"_n{cfg['n_train']}" if cfg["n_train"] else ""
        tag += "_cpl" if cfg["train_env"] == "coupled" else ""
        cfg["name"] = f"{cfg['regime']}_{cfg['variant']}{tag}_L{cfg['layers']}_lA{cfg['lam_A']}_s{cfg['seed']}"
    print("->", train(cfg))
