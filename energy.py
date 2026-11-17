#!/usr/bin/env python3
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EnergyConfig:
    input_dim: int
    width: int = 128
    blocks: int = 4
    max_length: int = 2048
    context_width: int = 128
    context_layers: int = 2
    context_heads: int = 4
    context_ff_mult: int = 4
    context_dropout: float = .1


class ContextBlock(nn.Module):
    """Pre-norm causal self-attention block with an optional KV cache."""
    def __init__(self, width, heads, ff_mult):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_mult * width), nn.GELU(), nn.Linear(ff_mult * width, width))

    def forward(self, x, past=None, dropout=0.):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).view(
            batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if past is not None:
            k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
        offset = k.shape[-2] - length
        allowed = (torch.arange(k.shape[-2], device=x.device)[None, :]
                   <= offset + torch.arange(length, device=x.device)[:, None])
        scores = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).masked_fill(~allowed, float('-inf'))
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.attention_out(attended), dropout, self.training)
        x = x + F.dropout(self.ff(self.ff_norm(x)), dropout, self.training)
        return x, (k.detach(), v.detach())


class TokenEnergy(nn.Module):
    def __init__(self, cfg: EnergyConfig):
        super().__init__()
        self.cfg = cfg
        self.context_input = nn.Linear(cfg.input_dim, cfg.context_width)
        self.position = nn.Parameter(torch.randn(cfg.max_length, cfg.context_width) * .02)
        self.context_blocks = nn.ModuleList(ContextBlock(cfg.context_width, cfg.context_heads, cfg.context_ff_mult)
                                            for _ in range(cfg.context_layers))
        self.context_norm = nn.LayerNorm(cfg.context_width)
        self.input_skips = nn.ModuleList(nn.Linear(cfg.input_dim, cfg.width) for _ in range(cfg.blocks))
        self.context_biases = nn.ModuleList(nn.Linear(cfg.context_width, cfg.width, bias=False)
                                            for _ in range(cfg.blocks))
        self.hidden_maps = nn.ModuleList(nn.Linear(cfg.width, cfg.width, bias=False) for _ in range(cfg.blocks - 1))
        self.readout = nn.Linear(cfg.width, 1, bias=False)
        self.context_slope = nn.Linear(cfg.context_width, cfg.input_dim, bias=False)
        for layer in self.input_skips:
            nn.init.normal_(layer.weight, std=.1 / math.sqrt(cfg.input_dim))
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.readout.weight, std=1 / math.sqrt(cfg.width))
        nn.init.normal_(self.context_slope.weight, std=.01 / math.sqrt(cfg.context_width))

    def context(self, h, cache=None):
        """Causal context c_t from UNEDITED activations h[:, :t+1]. Returns (c, cache)."""
        offset = 0 if cache is None else cache['length']
        end = offset + h.shape[1]
        if end > self.cfg.max_length:
            raise ValueError(f'Sequence length {end} exceeds energy max_length {self.cfg.max_length}')
        x = self.context_input(h) + self.position[offset:end]
        layers = []
        for i, block in enumerate(self.context_blocks):
            x, kv = block(x, None if cache is None else cache['layers'][i], self.cfg.context_dropout)
            layers.append(kv)
        return self.context_norm(x), dict(length=end, layers=layers)

    def forward(self, z, c):
        """Scalar energy per position: z [..., D], c [..., W] -> [...]."""
        u = None
        for i, (skip, bias) in enumerate(zip(self.input_skips, self.context_biases)):
            value = skip(z) + bias(c)
            if u is not None:
                value = value + self.hidden_maps[i - 1](u)
            u = F.gelu(value)
        return self.readout(u).squeeze(-1) + (self.context_slope(c) * z).sum(-1) / math.sqrt(self.cfg.input_dim)


def rollout(energy, h, c, steps, alpha, noise_std=0., create_graph=False):
    """z_0 = h; z_{k+1} = z_k + noise - alpha * grad_z E(z_k, c).

    alpha is a float or a tensor broadcastable to h (e.g. [1, n, 1]).
    create_graph=True keeps the trajectory differentiable w.r.t. energy params.
    """
    z = h.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        for _ in range(steps):
            if not create_graph:
                z = z.detach().requires_grad_(True)
            if noise_std:
                z = z + noise_std * torch.randn_like(z)
            grad, = torch.autograd.grad(energy(z, c).sum(), z, create_graph=create_graph)
            z = z - alpha * grad
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite edited activation')
    return z if create_graph else z.detach()
