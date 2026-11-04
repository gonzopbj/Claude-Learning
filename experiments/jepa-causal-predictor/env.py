"""Synthetic 2D multi-object world with a known directed interaction graph.

Objects (index: role):
  0  controlled object   : pushed by the 2D action force
  1  follower of 0       : pulled toward p0 + offset (spring, force on 1 only)
  2  follower of 1 and 3 : pulled toward p1 + offset and toward p3 (two parents)
  3  autonomous oscillator: anchored spring + noise, no parents
  4  distractor          : Ornstein-Uhlenbeck random walk, no parents, no children

True graph (parent -> child), over objects, plus the action:
  action -> 0, 0 -> 1, 1 -> 2, 3 -> 2.     Self-loops (inertia) are always present.

State per object: (x, y, vx, vy). Box is [0, 1]^2 with reflecting walls.
Everything is vectorized over a batch of independent episodes.
"""
from __future__ import annotations

import numpy as np

N_OBJ = 5
STATE_DIM = 4
ACTION_DIM = 2
LO, HI = 0.05, 0.95

# TRUE_PARENTS[i, j] = 1  iff  j is a parent of i (j != i).
TRUE_PARENTS = np.zeros((N_OBJ, N_OBJ), dtype=int)
TRUE_PARENTS[1, 0] = 1
TRUE_PARENTS[2, 1] = 1
TRUE_PARENTS[2, 3] = 1
# action parents: ACTION_PARENT[i] = 1 iff action drives i
ACTION_PARENT = np.array([1, 0, 0, 0, 0])

COLORS = np.array([
    [1.0, 0.1, 0.1],   # 0 red
    [0.1, 0.9, 0.1],   # 1 green
    [0.2, 0.3, 1.0],   # 2 blue
    [1.0, 0.9, 0.1],   # 3 yellow
    [0.8, 0.2, 0.9],   # 4 magenta (distractor)
])

DEFAULT_PARAMS = dict(
    dt=0.1,
    force_action=3.0,
    k_follow=6.0,      # spring constant, follower <- parent
    k_23=6.0,          # spring 3 -> 2
    k_anchor=2.0,      # object 3 anchor spring
    damping=1.0,
    damping_0=1.5,
    mass=1.0,
    noise_v=0.02,
    off_10=(0.15, 0.0),
    off_21=(0.0, 0.15),
    anchor_3=(0.5, 0.5),
    drive_3=0.15,       # extra noise injection that keeps 3 oscillating
    ou_theta=0.1, ou_sigma=0.08,   # distractor velocity OU
    clamp_1=None,      # if an (B,2) array: do(p1 = clamp_1, v1 = 0)
)

SHIFTS = {
    "id": {},
    # changed physics: stiffer springs, lighter damping, heavier objects
    "physics": dict(k_follow=9.0, k_23=9.0, damping=0.6, mass=1.3),
    # distractor moves 3x faster: its own mechanism changes, nothing else does
    "distractor": dict(ou_sigma=0.24),
    # hard intervention: object 1 frozen at a random location; 2 still follows 1
    "do1": "clamp",
}


def make_params(shift: str = "id", B: int = 1, rng: np.random.Generator | None = None) -> dict:
    p = dict(DEFAULT_PARAMS)
    s = SHIFTS[shift]
    if s == "clamp":
        rng = rng or np.random.default_rng(0)
        p["clamp_1"] = rng.uniform(0.15, 0.85, size=(B, 2))
    else:
        p.update(s)
    return p


def reset(B: int, rng: np.random.Generator, params: dict | None = None) -> np.ndarray:
    s = np.zeros((B, N_OBJ, STATE_DIM))
    s[:, :, :2] = rng.uniform(0.15, 0.85, size=(B, N_OBJ, 2))
    s[:, :, 2:] = rng.normal(0, 0.15, size=(B, N_OBJ, 2))
    if params is not None and params.get("clamp_1") is not None:
        s[:, 1, :2] = params["clamp_1"]
        s[:, 1, 2:] = 0.0
    return s


def step(s: np.ndarray, a: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """One semi-implicit Euler step. s: (B, N_OBJ, 4), a: (B, 2) in [-1, 1]."""
    P = params
    dt = P["dt"]
    p, v = s[:, :, :2], s[:, :, 2:]
    F = np.zeros_like(p)
    off10, off21, anch = np.array(P["off_10"]), np.array(P["off_21"]), np.array(P["anchor_3"])

    F[:, 0] = P["force_action"] * a - P["damping_0"] * v[:, 0]
    F[:, 1] = P["k_follow"] * (p[:, 0] + off10 - p[:, 1]) - P["damping"] * v[:, 1]
    F[:, 2] = (P["k_follow"] * (p[:, 1] + off21 - p[:, 2])
               + P["k_23"] * (p[:, 3] - p[:, 2]) - P["damping"] * v[:, 2])
    F[:, 3] = P["k_anchor"] * (anch - p[:, 3]) - 0.2 * v[:, 3] \
        + P["drive_3"] * rng.normal(size=(len(s), 2))

    v_new = v + dt * F / P["mass"] + P["noise_v"] * rng.normal(size=v.shape)
    # distractor: OU velocity, ignores forces entirely
    v_new[:, 4] = (1 - P["ou_theta"]) * v[:, 4] + P["ou_sigma"] * rng.normal(size=(len(s), 2))
    p_new = p + dt * v_new

    # reflecting walls
    below, above = p_new < LO, p_new > HI
    p_new = np.where(below, 2 * LO - p_new, p_new)
    p_new = np.where(above, 2 * HI - p_new, p_new)
    v_new = np.where(below | above, -v_new, v_new)
    p_new = np.clip(p_new, LO, HI)

    if P.get("clamp_1") is not None:
        p_new[:, 1] = P["clamp_1"]
        v_new[:, 1] = 0.0
    return np.concatenate([p_new, v_new], axis=-1)


def smooth_actions(B: int, T: int, rng: np.random.Generator, rho: float = 0.8, scale: float = 0.6) -> np.ndarray:
    a = np.zeros((B, T, ACTION_DIM))
    cur = rng.uniform(-1, 1, size=(B, ACTION_DIM))
    for t in range(T):
        cur = rho * cur + scale * rng.normal(size=(B, ACTION_DIM))
        cur = np.clip(cur, -1, 1)
        a[:, t] = cur
    return a


def rollout(B: int, T: int, rng: np.random.Generator, shift: str = "id", actions: np.ndarray | None = None):
    """Returns states (B, T+1, N_OBJ, 4) and actions (B, T, 2)."""
    params = make_params(shift, B, rng)
    s = reset(B, rng, params)
    if actions is None:
        actions = smooth_actions(B, T, rng)
    out = [s]
    for t in range(T):
        s = step(s, actions[:, t], rng, params)
        out.append(s)
    return np.stack(out, 1), actions


# ----------------------------------------------------------------------------- rendering
RES = 32
_grid = (np.arange(RES) + 0.5) / RES
_GX, _GY = np.meshgrid(_grid, _grid, indexing="xy")   # (RES, RES)
BLOB_SIGMA = 1.3 / RES


def render(states: np.ndarray) -> np.ndarray:
    """states (..., N_OBJ, 4) -> images (..., 3, RES, RES) float32 in [0,1], max-composited blobs."""
    lead = states.shape[:-2]
    st = states.reshape(-1, N_OBJ, STATE_DIM)
    px, py = st[:, :, 0], st[:, :, 1]
    d2 = (_GX[None, None] - px[:, :, None, None]) ** 2 + (_GY[None, None] - py[:, :, None, None]) ** 2
    blob = np.exp(-d2 / (2 * BLOB_SIGMA ** 2))                         # (M, N, R, R)
    img = (blob[:, :, None] * COLORS[None, :, :, None, None]).max(1)   # (M, 3, R, R)
    return img.reshape(*lead, 3, RES, RES).astype(np.float32)
