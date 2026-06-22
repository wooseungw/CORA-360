"""C-Abstractor spatial-aware resampler.

Inspired by the Cambrian-1 C-Abstractor (honeybee projector).  Uses depthwise
separable convolution with Squeeze-and-Excitation (SE) attention to capture
local spatial patterns — unlike MLP (token-independent) or BiMamba
(sequential).

Key design choices:
  - **No external dependencies** (no timm / einops — pure PyTorch).
  - **Token-preserving by default** (``num_queries=0``), compatible with
    CORA's VICReg overlap pipeline.  Set ``num_queries`` to a perfect
    square (e.g. 144) for controlled spatial downsampling.
  - Two-stage conv processing (pre/post optional pooling), matching the
    original C-Abstractor layout.

Reference:
  Tong et al., "Cambrian-1: A Fully Open, Vision-Centric Exploration of
  Multimodal LLMs", 2024.  https://arxiv.org/abs/2406.16860
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from cora.model.resampler.resamplers import BaseResampler

__all__ = ["CAbstractorResampler"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for 2-D feature maps (NCHW input)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] → permute → norm → permute back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention.

    GlobalAvgPool → FC → ReLU → FC → Sigmoid → channel-wise multiply.
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(dim // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        scale = x.mean(dim=(2, 3))          # [N, C]
        scale = self.fc(scale).unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        return x * scale


class _SpatialBlock(nn.Module):
    """Depthwise-separable conv block with SE attention.

    Architecture per block::

        DepthwiseConv2d(dim, k=7, pad=3, groups=dim)
        → LayerNorm2d
        → PointwiseConv(dim → dim * expand)
        → SiLU
        → SE(dim * expand)
        → PointwiseConv(dim * expand → dim)
        + residual

    This captures **local spatial patterns** (depthwise conv) with
    **channel recalibration** (SE) — the key advantage over token-
    independent MLP resamplers.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expand: int = 4,
        se_reduction: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        inner = dim * expand
        padding = kernel_size // 2

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, inner, 1)
        self.act = nn.SiLU(inplace=True)
        self.se = _SqueezeExcitation(inner, reduction=se_reduction)
        self.pw2 = nn.Conv2d(inner, dim, 1)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout2d(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pw2(x)
        x = self.drop_path(x)
        return residual + x


# ---------------------------------------------------------------------------
# C-Abstractor Resampler
# ---------------------------------------------------------------------------

class CAbstractorResampler(BaseResampler):
    """Spatial-aware resampler using depthwise conv + SE attention.

    Two-stage architecture:

    1. **Input projection** ``Linear(in_dim → hidden_dim)``
    2. Reshape to 2-D grid ``[BV, D, H, W]``
    3. **Stage 1** — ``depth`` × :class:`_SpatialBlock` (pre-pool)
    4. **Optional spatial pooling** ``AdaptiveAvgPool2d`` when
       ``num_queries > 0`` (reduces H×W → hw×hw where hw=√num_queries)
    5. **Stage 2** — ``depth`` × :class:`_SpatialBlock` (post-pool)
    6. Reshape back ``[BV, S', D]``
    7. **Final norm** ``LayerNorm``
    8. **Output MLP** ``Linear → SiLU → Linear``

    When ``num_queries=0`` (default), no pooling is applied and the full
    spatial resolution is preserved — compatible with VICReg overlap loss.

    Args:
        in_dim: Vision encoder hidden size.
        out_dim: Output (latent) dimension.
        hidden_dim: Processing dimension inside conv blocks.
        depth: Number of :class:`_SpatialBlock` per stage (total = 2×depth).
        num_queries: Target token count after pooling.  Must be a perfect
            square or 0 (no pooling).  Default 0.
        kernel_size: Depthwise convolution kernel size.
        expand: Expansion factor in pointwise conv.
        se_reduction: SE bottleneck reduction ratio.
        mlp_depth: Depth of the output readout MLP.
        drop_path: Stochastic depth rate for spatial blocks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        depth: int = 3,
        num_queries: int = 0,
        kernel_size: int = 7,
        expand: int = 4,
        se_reduction: int = 4,
        mlp_depth: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        self.hidden_dim = hidden_dim or out_dim
        self.depth = depth
        self.num_queries = num_queries
        self.kernel_size = kernel_size
        self.expand = expand
        self.se_reduction = se_reduction
        self.mlp_depth = mlp_depth
        self.drop_path = drop_path
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        hd = self.hidden_dim
        dp_rates = [
            x.item()
            for x in torch.linspace(0, self.drop_path, 2 * self.depth)
        ]

        # 1. Input projection
        self.input_proj = nn.Linear(self.in_dim, hd)

        # 2. Stage 1 — pre-pool spatial blocks
        self.stage1 = nn.Sequential(*[
            _SpatialBlock(
                hd,
                kernel_size=self.kernel_size,
                expand=self.expand,
                se_reduction=self.se_reduction,
                drop_path=dp_rates[i],
            )
            for i in range(self.depth)
        ])

        # 3. Optional spatial pooling
        if self.num_queries > 0:
            hw = int(math.isqrt(self.num_queries))
            if hw * hw != self.num_queries:
                raise ValueError(
                    f"num_queries must be a perfect square, got {self.num_queries}"
                )
            self.sampler: Optional[nn.Module] = nn.AdaptiveAvgPool2d((hw, hw))
        else:
            self.sampler = None

        # 4. Stage 2 — post-pool spatial blocks
        self.stage2 = nn.Sequential(*[
            _SpatialBlock(
                hd,
                kernel_size=self.kernel_size,
                expand=self.expand,
                se_reduction=self.se_reduction,
                drop_path=dp_rates[self.depth + i],
            )
            for i in range(self.depth)
        ])

        # 5. Final norm
        self.final_norm = nn.LayerNorm(hd)

        # 6. Output readout MLP
        layers: list[nn.Module] = [nn.Linear(hd, self.out_dim)]
        for _ in range(self.mlp_depth - 1):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(self.out_dim, self.out_dim))
        self.readout = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process ``[BV, S, D_in]`` → ``[BV, S', D_out]``.

        ``S' = S`` when ``num_queries=0`` (default),
        ``S' = num_queries`` otherwise.
        """
        bv, seq, _ = x.shape

        # Infer spatial dims
        gh = gw = int(math.isqrt(seq))
        if gh * gw != seq:
            # Non-square: find closest factorisation
            for h in range(int(math.isqrt(seq)), 0, -1):
                if seq % h == 0:
                    gh, gw = h, seq // h
                    break

        # 1. Project + reshape to 2D
        x = self.input_proj(x)                            # [BV, S, hd]
        x = x.permute(0, 2, 1).view(bv, -1, gh, gw)      # [BV, hd, H, W]

        # 2. Stage 1 (pre-pool)
        x = self.stage1(x)

        # 3. Optional spatial pooling
        if self.sampler is not None:
            x = self.sampler(x)

        # 4. Stage 2 (post-pool)
        x = self.stage2(x)

        # 5. Reshape back + norm + readout
        _, c_dim, hp, wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bv, hp * wp, c_dim)  # [BV, S', hd]
        x = self.final_norm(x)
        x = self.readout(x)                                   # [BV, S', out_dim]

        return x

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "num_queries": self.num_queries,
            "kernel_size": self.kernel_size,
            "expand": self.expand,
            "se_reduction": self.se_reduction,
            "mlp_depth": self.mlp_depth,
            "drop_path": self.drop_path,
        })
        return cfg
