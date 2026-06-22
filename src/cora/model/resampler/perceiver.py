"""Flamingo-style Perceiver Resampler with cross-attention.

Ported from ``flamingo-pytorch`` by lucidrains. Uses learnable latent tokens
that cross-attend to vision features through alternating PerceiverAttention
and FeedForward layers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cora.model.resampler.resamplers import BaseResampler

__all__ = ["PerceiverResampler"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    """Pre-norm feed-forward block."""
    inner_dim = dim * mult
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class _PerceiverAttention(nn.Module):
    """Cross-attention layer where latents attend to media features.

    Queries come from latents; keys/values come from the concatenation of
    media features and latents (self + cross attention in a single step).
    """

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Cross-attend latents to media features.

        Args:
            x: Media features ``[B, T, N1, D]``.
            latents: Latent queries ``[B, T, N2, D]``.

        Returns:
            Updated latents ``[B, T, N2, D]``.
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat([x, latents], dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # [B, T, N, H*Dh] -> view [B, T, N, H, Dh] -> permute [B, H, T, N, Dh]
        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            B_dim = t.shape[0]
            T_dim = t.shape[1]
            N_dim = t.shape[2]
            dh = t.shape[3] // h
            return t.view(B_dim, T_dim, N_dim, h, dh).permute(0, 3, 1, 2, 4)

        q, k, v = _to_heads(q), _to_heads(k), _to_heads(v)
        q = q * self.scale

        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        # [B, H, T, N, Dh] -> [B, T, N, H*Dh]
        B_dim, _, T_dim, N_dim, Dh = out.shape
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B_dim, T_dim, N_dim, h * Dh)
        return self.to_out(out)


class _PerceiverResamplerModule(nn.Module):
    """Core Perceiver Resampler with learnable latents and cross-attention stack.

    Args:
        dim: Model dimension.
        depth: Number of cross-attention + FFN layers.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        num_latents: Number of learnable latent tokens.
        ff_mult: FFN expansion factor (0 to disable FFN).
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                _feed_forward(dim=dim, mult=ff_mult) if ff_mult > 0 else nn.Identity(),
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run cross-attention on media features.

        Args:
            x: ``[B, T, F, V, D]`` media features (T=time, F=frames, V=spatial).

        Returns:
            ``[B, T, num_latents, D]``.
        """
        b, T = x.shape[0], x.shape[1]
        F_dim, v = x.shape[2], x.shape[3]

        # Flatten frame and spatial dims
        x = x.reshape(b, T, F_dim * v, x.shape[-1])

        # Expand latents
        latents = self.latents.unsqueeze(0).unsqueeze(0).expand(b, T, -1, -1)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PerceiverResampler(BaseResampler):
    """Flamingo-style Perceiver Resampler.

    Wraps :class:`_PerceiverResamplerModule` with the standard
    ``BaseResampler`` interface. Input features are reshaped from
    ``[B, S, D]`` to the ``[B, 1, 1, S, D]`` format expected by the
    perceiver, then squeezed back.

    Args:
        in_dim: Input (vision) feature dimension.
        out_dim: Output (latent) feature dimension.
        num_latents: Number of learnable query latents.
        depth: Number of cross-attention layers.
        heads: Number of attention heads.
        dim_head: Dimension per attention head.
        ff_mult: FFN expansion factor.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_latents: int = 64,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
    ) -> None:
        self.num_latents = num_latents
        self._depth = depth
        self._heads = heads
        self._dim_head = dim_head
        self._ff_mult = ff_mult
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        self.perceiver = _PerceiverResamplerModule(
            dim=self.in_dim,
            depth=self._depth,
            dim_head=self._dim_head,
            heads=self._heads,
            num_latents=self.num_latents,
            ff_mult=self._ff_mult,
        )
        # Project to output dimension if different
        self.proj = nn.Linear(self.in_dim, self.out_dim) if self.in_dim != self.out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resample ``[B, S, D_in]`` → ``[B, num_latents, D_out]``.

        The input is treated as a single time-step with a single frame of
        *S* spatial tokens.
        """
        # [B, S, D] -> [B, T=1, F=1, V=S, D]
        out = self.perceiver(x[:, None, None])  # [B, 1, num_latents, D]
        out = out.squeeze(1)  # [B, num_latents, D]
        return self.proj(out)

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "num_latents": self.num_latents,
            "depth": self._depth,
            "heads": self._heads,
            "dim_head": self._dim_head,
            "ff_mult": self._ff_mult,
        })
        return cfg
