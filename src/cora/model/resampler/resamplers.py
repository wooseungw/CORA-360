"""Core resampler modules: BaseResampler, Identity, AvgPool, Conv, QFormer, MLP.

All resamplers inherit from :class:`BaseResampler` which defines a standard
``(in_dim, out_dim)`` interface with ``_build_layers()`` / ``forward()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

__all__ = [
    "BaseResampler",
    "IdentityResampler",
    "AvgPoolResampler",
    "ConvNeXtBlock",
    "ConvResampler",
    "QFormerResampler",
    "MLPResampler",
]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------

class BaseResampler(nn.Module):
    """Abstract base class for all resamplers.

    Subclasses must implement :meth:`_build_layers` and :meth:`forward`.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._build_layers()

    def _build_layers(self) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``[..., in_dim]`` → ``[..., out_dim]``."""
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "in_dim": self.in_dim, "out_dim": self.out_dim}


# -----------------------------------------------------------------------
# Simple Resamplers
# -----------------------------------------------------------------------

class IdentityResampler(BaseResampler):
    """Linear projection without any token pooling.

    ``forward(x)`` preserves sequence length, only transforms the last dim.
    """

    def _build_layers(self) -> None:
        self.proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AvgPoolResampler(BaseResampler):
    """Average-pool all tokens into one, then project.

    ``forward(x)`` with ``x: [B, S, D]`` → ``[B, 1, out_dim]``.
    """

    def _build_layers(self) -> None:
        self.proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1, keepdim=True))


# -----------------------------------------------------------------------
# ConvNeXt-based Resampler
# -----------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block adapted for 2-D feature maps."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) → (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        return residual + x


class ConvResampler(BaseResampler):
    """ConvNeXt-based resampler that reshapes tokens to a 2-D grid.

    ``forward(x)`` with ``x: [B, S, D]`` → ``[B, 1, out_dim]`` after global
    average pooling over the spatial grid.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_tokens: int = 144,
        depths: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        num_repeats: int = 1,
    ) -> None:
        self.num_tokens = num_tokens
        self.depths = depths or [2, 2]
        self.drop_path_rate = drop_path_rate
        self.num_repeats = num_repeats
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        self.input_proj = (
            nn.Linear(self.in_dim, self.out_dim)
            if self.in_dim != self.out_dim
            else nn.Identity()
        )
        self.repeated_stages = nn.ModuleList()

        for _ in range(self.num_repeats):
            dp_rates = [
                x.item()
                for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
            ]
            cur = 0
            stages = nn.ModuleList()
            for depth in self.depths:
                stage = nn.ModuleList(
                    [
                        ConvNeXtBlock(dim=self.out_dim, drop_path=dp_rates[cur + j])
                        for j in range(depth)
                    ]
                )
                stages.append(stage)
                cur += depth
            self.repeated_stages.append(stages)

        self.norm = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.head = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = x.shape
        x = self.input_proj(x)

        H = W = int(seq_len ** 0.5) if seq_len == int(seq_len ** 0.5) ** 2 else int(seq_len ** 0.5) + 1
        pad_len = H * W - seq_len
        if pad_len > 0:
            x = torch.cat(
                [x, torch.zeros(B, pad_len, self.out_dim, device=x.device, dtype=x.dtype)],
                dim=1,
            )

        x = x.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)

        for repeat_stages in self.repeated_stages:
            for stage in repeat_stages:
                for block in stage:
                    x = block(x)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.out_dim)
        if pad_len > 0:
            x = x[:, :seq_len, :]

        x = self.norm(x)
        x = x.mean(dim=1, keepdim=True)
        return self.head(x)


# -----------------------------------------------------------------------
# MLP Resampler
# -----------------------------------------------------------------------

class MLPResampler(BaseResampler):
    """Multi-layer perceptron resampler with optional token pooling.

    Args:
        vision_dim: Input feature dimension (vision encoder output).
        latent_dim: Output feature dimension (latent space).
        hidden_dim: Width of hidden layers (defaults to ``max(in, out)``).
        depth: Number of linear layers (≥ 1).
        use_ln: Insert LayerNorm between layers.
        pool_tokens: If set and < *S*, adaptively pool the sequence to this
            length.
        pool_type: ``"avg"`` or ``"max"`` pooling.
    """

    def __init__(
        self,
        vision_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        depth: int = 3,
        use_ln: bool = True,
        pool_tokens: Optional[int] = None,
        pool_type: str = "avg",
    ) -> None:
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_ln = use_ln
        self.pool_tokens = pool_tokens
        self.pool_type = pool_type
        super().__init__(vision_dim, latent_dim)

    def _build_layers(self) -> None:
        hidden_dim = self.hidden_dim or max(self.in_dim, self.out_dim)
        layers: list[nn.Module] = []
        current_dim = self.in_dim

        for _ in range(self.depth - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_ln:
                layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
            layers.append(nn.GELU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Transform ``[BV, S, D_in]`` → ``[BV, S', D_out]``."""
        BV, S, _ = vision_features.shape
        x = self.mlp(vision_features.reshape(-1, self.in_dim)).view(BV, S, self.out_dim)

        if self.pool_tokens is not None and 0 < self.pool_tokens < S:
            pool_fn = F.adaptive_max_pool1d if self.pool_type == "max" else F.adaptive_avg_pool1d
            x = pool_fn(x.transpose(1, 2), self.pool_tokens).transpose(1, 2)

        return x

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "use_ln": self.use_ln,
            "pool_tokens": self.pool_tokens,
            "pool_type": self.pool_type,
        })
        return cfg


# -----------------------------------------------------------------------
# QFormer Resampler
# -----------------------------------------------------------------------

class QFormerResampler(BaseResampler):
    """Mini Q-Former: learnable query tokens + BERT cross-attention stack.

    Args:
        in_dim: Vision feature dimension.
        out_dim: Output (latent) dimension.
        num_query: Number of learnable query tokens.
        num_hidden_layers: Depth of the BERT encoder.
        num_attention_heads: Number of attention heads per layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_query: int = 32,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
    ) -> None:
        self.num_query = num_query
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        cfg = BertConfig(
            hidden_size=self.in_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.in_dim * 4,
        )
        cfg.is_decoder = True
        cfg.add_cross_attention = True
        cfg.encoder_width = self.in_dim

        self.bert = BertModel(cfg)
        self.query = nn.Parameter(torch.randn(1, self.num_query, self.in_dim))
        self.proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-attend queries to vision features.

        Args:
            x: ``[B, S, D_in]`` vision features.

        Returns:
            ``[B, num_query, D_out]``.
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        attn_mask = torch.ones(x.size()[:-1], device=x.device, dtype=x.dtype)
        out = self.bert(
            inputs_embeds=q,
            encoder_hidden_states=x,
            encoder_attention_mask=attn_mask,
        )
        return self.proj(out.last_hidden_state)

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "num_query": self.num_query,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
        })
        return cfg
