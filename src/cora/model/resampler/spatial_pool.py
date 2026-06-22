"""Spatial pooling resampler that reduces token count via 2-D pooling.

Supports average pooling, max pooling, and learned convolution-based pooling
over the spatial grid of vision tokens.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cora.model.resampler.resamplers import BaseResampler

__all__ = ["SpatialPoolResampler"]


class SpatialPoolResampler(BaseResampler):
    """Reduce vision tokens via 2-D spatial pooling.

    Reshapes the flat token sequence ``[B, S, D]`` into a spatial grid, applies
    pooling to reduce the spatial resolution, and flattens back.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        pool_size: Kernel / stride size for pooling.
        pool_mode: ``"average"``, ``"max"``, or ``"conv"``.
        depth: Number of additional MLP layers after pooling (0 = linear only).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        pool_size: int = 4,
        pool_mode: str = "average",
        depth: int = 2,
    ) -> None:
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self._depth = depth
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        if self.pool_mode == "average":
            self.pool: nn.Module = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        elif self.pool_mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        elif self.pool_mode == "conv":
            self.pool = nn.Conv2d(
                in_channels=self.in_dim,
                out_channels=self.in_dim,
                kernel_size=self.pool_size,
                stride=self.pool_size,
            )
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode!r}")

        # Optional MLP after pooling
        layers: list[nn.Module] = []
        current_dim = self.in_dim
        for i in range(self._depth):
            next_dim = self.out_dim if i == self._depth - 1 else max(self.in_dim, self.out_dim)
            layers.append(nn.Linear(current_dim, next_dim))
            if i < self._depth - 1:
                layers.append(nn.LayerNorm(next_dim))
                layers.append(nn.GELU())
            current_dim = next_dim
        self.proj = nn.Sequential(*layers) if layers else nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool then project ``[B, S, D_in]`` → ``[B, S', D_out]``."""
        B, S, D = x.shape

        # Infer spatial dims
        H = W = int(math.sqrt(S))
        if H * W < S:
            H += 1
            W = (S + H - 1) // H

        # Pad if necessary
        pad_len = H * W - S
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)], dim=1)

        # Reshape to spatial: [B, D, H, W]
        spatial = x.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        pooled = self.pool(spatial)  # [B, D', H', W']
        # Flatten back: [B, H'*W', D'] -> project
        pooled = pooled.flatten(2).transpose(1, 2).contiguous()
        return self.proj(pooled)

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "pool_size": self.pool_size,
            "pool_mode": self.pool_mode,
            "depth": self._depth,
        })
        return cfg
