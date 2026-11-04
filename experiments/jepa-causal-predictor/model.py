"""JEPA world model with a factor-token predictor whose attention can be gated by a learned adjacency.

Encoders
  StateEncoder : per-object MLP (no weight sharing) -> K factor tokens. Factor k sees only object k,
                 so factor/object alignment is given by construction ("state regime").
  PixelEncoder : CNN on two stacked frames -> flat vector -> split into K chunks ("pixel regime").
                 Nothing ties a chunk to an object.

Predictor (one attention layer by default, so the gate matrix IS the one-step dependency graph)
  tokens = K factor tokens + 1 action token. Factor i attends to {self, other factors, action}.
  Gate G[i, j] in [0, 1] for j != i; self-loop G[i, i] = 1 is pinned and never penalized (this is
  what makes the gate scale identifiable under the softmax normalization).

  variants
    dense     G = 1
    dense_ent G = 1, plus an attention-entropy penalty (sparsity without structure)
    soft      p_ij ∝ exp(l_ij) * sigmoid(alpha_ij)                  (= logits + log sigma(alpha))
    hard      p_ij ∝ exp(l_ij) * m_ij, m ~ straight-through Gumbel-sigmoid(alpha); eval: 1[sigma>0.5]
    mult      softmax(l_ij * sigmoid(alpha_ij))  -- the literal "multiply A into the scores" version
    oracle    G = true graph (fixed)
    random    G = random graph with the same number of edges as the true one (fixed, seeded)

SIGReg (LeJEPA): Epps-Pulley test of the projected embeddings against N(0,1) on random 1-D slices.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import ACTION_DIM, ACTION_PARENT, N_OBJ, STATE_DIM, TRUE_PARENTS

GATED = ("soft", "hard", "mult")


# ----------------------------------------------------------------------------- SIGReg
class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (Balestriero & LeCun, LeJEPA).

    For M random unit directions u, project x = Z u (N samples) and compute the Epps-Pulley
    statistic  N * int |phi_N(t) - exp(-t^2/2)|^2 w(t) dt  with w(t) = exp(-t^2/2), by trapezoid
    on t in [0, t_max] (the integrand is even). Averaged over slices. Directions resampled per call.
    """

    def __init__(self, n_slices: int = 1024, n_knots: int = 17, t_max: float = 3.0):
        super().__init__()
        t = torch.linspace(0, t_max, n_knots)
        w = torch.full((n_knots,), t_max / (n_knots - 1))
        w[0] = w[-1] = w[0] / 2
        self.register_buffer("t", t)
        self.register_buffer("w", w * 2 * torch.exp(-t ** 2 / 2))   # x2: integrate over [-t_max, t_max]
        self.register_buffer("phi", torch.exp(-t ** 2 / 2))
        self.n_slices = n_slices

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., N, D): statistic over the sample axis N, averaged over slices and leading axes."""
        if z.dim() == 2:
            z = z.unsqueeze(0)
        n, d = z.shape[-2], z.shape[-1]
        u = torch.randn(d, self.n_slices, device=z.device)
        u = u / u.norm(dim=0, keepdim=True)
        xt = (z @ u).unsqueeze(-1) * self.t                  # (..., N, M, knots)
        err = (torch.cos(xt).mean(-3) - self.phi) ** 2 + torch.sin(xt).mean(-3) ** 2
        return ((err * self.w).sum(-1) * n).mean()


# ----------------------------------------------------------------------------- encoders
class StateEncoder(nn.Module):
    def __init__(self, d: int, hidden: int = 64, K: int = N_OBJ):
        super().__init__()
        self.K, self.d = K, d
        self.w1 = nn.Parameter(torch.randn(K, STATE_DIM, hidden) / math.sqrt(STATE_DIM))
        self.b1 = nn.Parameter(torch.zeros(K, hidden))
        self.w2 = nn.Parameter(torch.randn(K, hidden, hidden) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(K, hidden))
        self.w3 = nn.Parameter(torch.randn(K, hidden, d) / math.sqrt(hidden))
        self.b3 = nn.Parameter(torch.zeros(K, d))

    def forward(self, s: torch.Tensor) -> torch.Tensor:      # s (..., K, 4) -> (..., K, d)
        h = F.gelu(torch.einsum("...ki,kih->...kh", s, self.w1) + self.b1)
        h = F.gelu(torch.einsum("...ki,kih->...kh", h, self.w2) + self.b2)
        return torch.einsum("...ki,kih->...kh", h, self.w3) + self.b3


class PixelEncoder(nn.Module):
    def __init__(self, d: int, K: int = N_OBJ):
        super().__init__()
        self.K, self.d = K, d
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.GELU(),     # 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # 8
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),    # 4
            nn.Flatten(), nn.Linear(64 * 16, 256), nn.GELU(), nn.Linear(256, K * d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (..., 6, 32, 32) -> (..., K, d)
        lead = x.shape[:-3]
        z = self.net(x.reshape(-1, *x.shape[-3:]))
        return z.reshape(*lead, self.K, self.d)


# ----------------------------------------------------------------------------- predictor
def true_gate(K: int = N_OBJ) -> np.ndarray:
    """(K, K+1) gate over [factors..., action]; diagonal included as 1."""
    g = np.zeros((K, K + 1))
    g[:, :K] = TRUE_PARENTS[:K, :K]
    g[:, K] = ACTION_PARENT[:K]
    g[np.arange(K), np.arange(K)] = 1
    return g


def random_gate(seed: int, K: int = N_OBJ) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    g = np.zeros((K, K + 1))
    off = [(i, j) for i in range(K) for j in range(K) if i != j]
    n_edges = int(TRUE_PARENTS[:K, :K].sum())
    for idx in rng.choice(len(off), n_edges, replace=False):
        g[off[idx]] = 1
    g[rng.integers(K), K] = 1          # one action edge
    g[np.arange(K), np.arange(K)] = 1
    return g


class GatedPredictor(nn.Module):
    def __init__(self, d: int, variant: str = "dense", K: int = N_OBJ, width: int = 64, heads: int = 4,
                 layers: int = 1, seed: int = 0, gumbel_tau: float = 0.5, alpha_init: float = 1.0):
        super().__init__()
        self.K, self.d, self.variant, self.heads, self.width = K, d, variant, heads, width
        self.gumbel_tau = gumbel_tau
        self.inp = nn.Linear(d, width)
        self.act = nn.Linear(ACTION_DIM, width)
        self.tok = nn.Parameter(torch.randn(K + 1, width) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(width), qkv=nn.Linear(width, 3 * width), o=nn.Linear(width, width),
                ln2=nn.LayerNorm(width), mlp=nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                                           nn.Linear(4 * width, width)))))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
        self.offdiag = torch.ones(K, K + 1)
        self.offdiag[torch.arange(K), torch.arange(K)] = 0
        self.register_buffer("offmask", self.offdiag.bool())
        if variant in GATED:
            self.alpha = nn.Parameter(torch.full((K, K + 1), alpha_init))
        fixed = {"oracle": true_gate(K), "random": random_gate(seed, K)}.get(variant)
        if fixed is not None:
            self.register_buffer("fixed_gate", torch.tensor(fixed, dtype=torch.float32))
        self.last_attn = None

    # -- gate ------------------------------------------------------------------
    def gate_probs(self) -> torch.Tensor:
        """(K, K+1) edge strengths in [0,1], diagonal = 1. For dense: ones."""
        K = self.K
        if self.variant in GATED:
            g = torch.sigmoid(self.alpha)
        elif hasattr(self, "fixed_gate"):
            g = self.fixed_gate
        else:
            g = torch.ones(K, K + 1, device=self.tok.device)
        return torch.where(self.offmask, g, torch.ones_like(g))

    def _gate_sample(self) -> torch.Tensor:
        g = self.gate_probs()
        if self.variant != "hard":
            return g
        if self.training:
            u = torch.rand_like(self.alpha).clamp(1e-6, 1 - 1e-6)
            logits = (self.alpha + torch.log(u) - torch.log1p(-u)) / self.gumbel_tau
            soft = torch.sigmoid(logits)
            m = (soft > 0.5).float() + soft - soft.detach()
        else:
            m = (g > 0.5).float()
        return torch.where(self.offmask, m, torch.ones_like(m))

    def sparsity(self) -> torch.Tensor:
        if self.variant not in GATED:
            return torch.zeros((), device=self.tok.device)
        return torch.sigmoid(self.alpha)[self.offmask].mean()

    # -- forward ---------------------------------------------------------------
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z (B, K, d), a (B, 2) -> predicted next z (B, K, d)."""
        B, K, _ = z.shape
        x = torch.cat([self.inp(z), self.act(a).unsqueeze(1)], 1) + self.tok
        G = self._gate_sample()                                    # (K, K+1)
        # action token attends only to itself
        G_full = torch.cat([G, F.one_hot(torch.tensor(K), K + 1).float().to(G).unsqueeze(0)], 0)
        H, dh = self.heads, self.width // self.heads
        attn_record = []
        for blk in self.blocks:
            h = blk["ln1"](x)
            q, k, v = blk["qkv"](h).reshape(B, K + 1, 3, H, dh).permute(2, 0, 3, 1, 4)
            logits = q @ k.transpose(-1, -2) / math.sqrt(dh)        # (B, H, T, T)
            if self.variant == "mult":
                p = torch.softmax(logits * G_full, -1)
            else:
                logits = logits - logits.amax(-1, keepdim=True)
                e = torch.exp(logits) * G_full
                p = e / e.sum(-1, keepdim=True)
            attn_record.append(p.detach())
            x = x + blk["o"]((p @ v).transpose(1, 2).reshape(B, K + 1, self.width))
            x = x + blk["mlp"](blk["ln2"](x))
        self.last_attn = attn_record
        self._last_p = p
        return z + self.out(x[:, :K])

    def attn_entropy(self) -> torch.Tensor:
        p = self._last_p[:, :, : self.K]                          # factor query rows
        return -(p * torch.log(p + 1e-9)).sum(-1).mean()


class WorldModel(nn.Module):
    def __init__(self, regime: str, variant: str, d: int = 16, layers: int = 1, seed: int = 0):
        super().__init__()
        n_slices = 64 if regime == "state" else 256
        self.regime = regime
        self.enc = StateEncoder(d) if regime == "state" else PixelEncoder(d)
        self.pred = GatedPredictor(d, variant, layers=layers, seed=seed)
        self.sigreg = SIGReg(n_slices=n_slices)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (B,K,d), actions (B,T,2) -> (B,T,K,d) predicted z_1..z_T."""
        out, z = [], z0
        for t in range(actions.shape[1]):
            z = self.pred(z, actions[:, t])
            out.append(z)
        return torch.stack(out, 1)
