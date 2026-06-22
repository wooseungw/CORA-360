"""Masked-drop resampler: training-time random token dropping for regularisation.

During training, randomly keeps a fraction of vision tokens (MAE-style masking).
At inference time, all tokens are passed through unchanged.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from cora.model.resampler.resamplers import BaseResampler

__all__ = ["MaskedDropResampler"]


class MaskedDropResampler(BaseResampler):
    """Token-dropping resampler for regularisation / efficiency.

    Modes:
    * ``"fixed"`` – keep a fixed ratio of tokens.
    * ``"range"`` – keep a uniformly-sampled ratio between
      ``ratio_lower`` and ``ratio_upper``.

    During evaluation the module is an identity (optionally with a linear
    projection if ``in_dim != out_dim``).

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        mask_ratio: Fraction of tokens to **keep** in ``"fixed"`` mode.
        mask_mode: ``"fixed"`` or ``"range"``.
        skip_percentage: Probability of skipping masking entirely per sample.
        ratio_lower: Lower bound for ``"range"`` mode.
        ratio_upper: Upper bound for ``"range"`` mode.
        depth: MLP depth for optional projection (0 = linear only).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        mask_ratio: float = 0.5,
        mask_mode: str = "fixed",
        skip_percentage: float = 0.0,
        ratio_lower: float = 0.3,
        ratio_upper: float = 0.7,
        depth: int = 2,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.skip_percentage = skip_percentage
        self.ratio_lower = ratio_lower
        self.ratio_upper = ratio_upper
        self._depth = depth
        super().__init__(in_dim, out_dim)

    def _build_layers(self) -> None:
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
        """Optionally mask tokens then project.

        Args:
            x: ``[B, S, D_in]`` vision features.

        Returns:
            ``[B, S', D_out]`` where ``S' <= S`` during training.
        """
        x = self.proj(x)

        if not self.training:
            return x

        # Per-batch-item masking
        if self.skip_percentage > random.random():
            return x

        masked: List[torch.Tensor] = []
        for b in range(x.size(0)):
            tokens = x[b]  # [S, D]
            num_tokens = tokens.size(0)
            if self.mask_mode == "fixed":
                num_keep = max(1, int(num_tokens * self.mask_ratio))
            elif self.mask_mode == "range":
                ratio = random.uniform(self.ratio_lower, self.ratio_upper)
                num_keep = max(1, int(num_tokens * ratio))
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode!r}")
            kept, _, _ = self._random_masking(tokens.unsqueeze(0), num_keep)
            masked.append(kept.squeeze(0))

        # Stack if all have the same length, otherwise return as list
        if all(m.size(0) == masked[0].size(0) for m in masked):
            return torch.stack(masked, dim=0)
        # Pad to max length
        max_len = max(m.size(0) for m in masked)
        D = masked[0].size(-1)
        padded = torch.zeros(len(masked), max_len, D, device=x.device, dtype=x.dtype)
        for i, m in enumerate(masked):
            padded[i, :m.size(0)] = m
        return padded

    @staticmethod
    def _random_masking(
        x: torch.Tensor,
        len_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample random masking via argsort of noise.

        Args:
            x: ``[N, L, D]``.
            len_keep: Number of tokens to keep.

        Returns:
            Tuple of ``(x_masked, mask, ids_restore)``.
        """
        N, L, D = x.shape
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    @property
    def config(self) -> Dict[str, Any]:
        cfg = super().config
        cfg.update({
            "mask_ratio": self.mask_ratio,
            "mask_mode": self.mask_mode,
            "skip_percentage": self.skip_percentage,
            "ratio_lower": self.ratio_lower,
            "ratio_upper": self.ratio_upper,
            "depth": self._depth,
        })
        return cfg
