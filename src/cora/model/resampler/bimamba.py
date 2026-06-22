"""Bidirectional Mamba (BiMamba) resampler.

Uses forward + backward Mamba SSM passes to capture bidirectional dependencies
in the vision token sequence. Requires the optional ``mamba_ssm`` package;
raises a clear :class:`ImportError` at construction time if unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from cora.model.resampler.resamplers import BaseResampler

__all__ = ["BiMambaResampler", "BidirectionalMambaResampler"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional mamba_ssm import
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba  # type: ignore[import-untyped]

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None  # type: ignore[assignment,misc]


def _ensure_mamba() -> None:
    if not MAMBA_AVAILABLE:
        raise ImportError(
            "mamba_ssm is not installed. Install it with:\n"
            "  pip install mamba-ssm causal-conv1d\n"
            "Note: CUDA must be properly configured."
        )


# ---------------------------------------------------------------------------
# Bidirectional Mamba block
# ---------------------------------------------------------------------------

class _BiMambaBlock(nn.Module):
    """Single bidirectional Mamba block with pre/post-norm and residual."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int,
        d_conv: int,
        expand: float,
        dropout: float,
        norm_first: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_block = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_block = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        working = self.norm(x) if self.norm_first else x

        fwd = self.forward_block(working)
        bwd = self.backward_block(torch.flip(working, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])

        combined = self.dropout(0.5 * (fwd + bwd))
        out = residual + combined
        return out if self.norm_first else self.norm(out)


# ---------------------------------------------------------------------------
# BiMamba Resampler
# ---------------------------------------------------------------------------

class BiMambaResampler(BaseResampler):
    """Resampler using a stack of bidirectional Mamba blocks.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dim: Internal processing dimension (defaults to *out_dim*).
        num_layers: Number of bidirectional Mamba blocks.
        d_state: SSM state dimension.
        d_conv: Convolution kernel size in Mamba.
        expand: Expansion factor for the inner MLP in Mamba.
        dropout: Dropout probability between blocks.
        norm_first: Pre-norm (True) vs. post-norm (False).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: float = 2.0,
        dropout: float = 0.0,
        norm_first: bool = True,
    ) -> None:
        self.hidden_dim = hidden_dim or out_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self._dropout = dropout
        self.norm_first = norm_first
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
        _ensure_mamba()

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            _BiMambaBlock(
                self.hidden_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self._dropout,
                norm_first=self.norm_first,
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process ``[BV, S, D_in]`` → ``[BV, S, D_out]``."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_proj(x)

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dropout": self._dropout,
            "norm_first": self.norm_first,
        })
        return cfg


BidirectionalMambaResampler = BiMambaResampler
