"""Projector modules for mapping between vision and language feature spaces.

* :class:`VICRegProjector` – MLP head for VICReg self-supervised loss computation.
* :class:`PanoramaProjector` – Projects resampled vision tokens into the LLM
  embedding space with panorama-aware positional encoding and multi-view
  stitching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import infer_hw, resolve_module_dtype
from .positional import PanoramaPositionalEncoding

__all__ = ["VICRegProjector", "PanoramaProjector"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VICReg Projector
# ---------------------------------------------------------------------------

class VICRegProjector(nn.Module):
    """MLP projector for VICReg / contrastive loss computation.

    Projects resampler outputs to a (potentially higher-dimensional) space
    where variance/covariance regularisation or contrastive loss is applied.

    When *dropout* > 0, two forward passes with different dropout masks
    produce two stochastic views of the same features — used by
    :class:`~cora.training.losses.PanoContrastiveLoss` for within-tile
    instance discrimination.

    Args:
        in_dim: Input feature dimension (from resampler).
        out_dim: Output projection dimension.
        hidden_dim: Hidden layer width (defaults to ``max(in_dim, out_dim)``).
        depth: Number of linear layers (≥ 1).
        use_ln: Apply LayerNorm between hidden layers.
        dropout: Dropout probability applied after each hidden activation.
            Set > 0 for contrastive loss (creates two stochastic views).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        depth: int = 2,
        use_ln: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hd = hidden_dim or max(in_dim, out_dim)
        layers: list[nn.Module] = []
        d_prev = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_prev, hd))
            layers.append(nn.LayerNorm(hd) if use_ln else nn.Identity())
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_prev = hd
        layers.append(nn.Linear(d_prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``[BV, S, D_in]`` → ``[BV, S, D_out]``."""
        BV, S, D = x.shape
        return self.mlp(x.reshape(-1, D)).view(BV, S, -1)


# ---------------------------------------------------------------------------
# Panorama Projector
# ---------------------------------------------------------------------------

class PanoramaProjector(nn.Module):
    """Panorama-aware projection and multi-view stitching layer.

    Pipeline: optional PE → linear projection → view stitching.

    Stitching modes:
    * ``"concat"`` – simple concatenation of all views.
    * ``"stride_views"`` (default) – strip overlapping columns then concatenate.
    * ``"resample"`` – differentiable grid-sample blending of overlapping views.

    Args:
        config: :class:`~cora.config.schema.ModelConfig` or compatible object.
        latent_dimension: Input feature dimension from the resampler.
        language_hidden_size: Target LLM hidden dimension.
    """

    def __init__(
        self,
        config: Any,
        latent_dimension: int,
        language_hidden_size: int,
        image_processing: Any = None,
    ) -> None:
        super().__init__()

        def _read(src: Any, key: str, default: Any) -> Any:
            if src is None:
                return default
            if hasattr(src, key):
                return getattr(src, key, default)
            if isinstance(src, dict):
                return src.get(key, default)
            return default

        def _cfg(key: str, default: Any) -> Any:
            return _read(config, key, default)

        # `image_processing` (e.g. ImageProcessingConfig) holds overlap_ratio /
        # stitching_mode etc. ModelConfig does not, so a pure `_cfg` lookup for
        # those keys silently falls back to defaults.  Prefer image_processing
        # when given, then fall through to the model config (some YAMLs put
        # overlap_ratio under `models:` via `extra="allow"`).
        def _ip(key: str, default: Any) -> Any:
            return _read(image_processing, key, _cfg(key, default))

        # 1. Positional encoding
        use_pe = bool(_cfg("use_projection_positional_encoding", True))
        if use_pe:
            self.projection_pe: Optional[PanoramaPositionalEncoding] = PanoramaPositionalEncoding(
                embed_dim=latent_dimension,
                view_encoding_type=_cfg("pe_view_encoding_type", "sinusoidal"),
                spatial_encoding_type=_cfg("pe_spatial_encoding_type", "sinusoidal"),
                enable_continuity=bool(_cfg("pe_enable_continuity", True)),
                overlap_ratio=float(_ip("overlap_ratio", 0.5)),
                temperature=float(_cfg("pe_temperature", 10000.0)),
                dropout=float(_cfg("pe_dropout", 0.0)),
            )
        else:
            self.projection_pe = None

        # 2. Linear projection
        self.linear = nn.Linear(latent_dimension, language_hidden_size)

        # 3. Stitching configuration (sourced from ImageProcessingConfig)
        self.overlap_ratio = float(_ip("overlap_ratio", 0.5))
        self.stitching_mode = str(_ip("stitching_mode", "stride_views"))
        self.stitch_stride_offset = int(_cfg("stitch_stride_offset", 0))
        self.stitch_target_cols = int(_cfg("stitch_target_cols", 0))
        self.stitch_target_to_view_width = bool(_ip("stitch_target_to_view_width", False))
        self.stitch_interp = str(_ip("stitch_interp", "linear"))

    def forward(
        self,
        resampled_features: torch.Tensor,
        batch_size: int,
        num_views: int,
        dtype_cache: Dict[str, torch.dtype],
        language_model: nn.Module,
    ) -> torch.Tensor:
        """Project and stitch resampled features into an LLM-ready tensor.

        Args:
            resampled_features: ``[B*V, S, D_latent]``.
            batch_size: Batch size *B*.
            num_views: Number of views *V*.
            dtype_cache: Mutable dtype cache dict.
            language_model: The LLM module (used for dtype alignment).

        Returns:
            ``[B, T, D_lm]`` where *T* depends on stitching mode.
        """
        # 1. Dtype alignment for projector
        proj_dtype = resolve_module_dtype(
            dtype_cache, "vision_to_language_projection", self.linear, resampled_features.dtype,
        )
        if proj_dtype is not None and resampled_features.dtype != proj_dtype:
            resampled_features = resampled_features.to(proj_dtype)

        # 2. Apply positional encoding
        if self.projection_pe is not None:
            resampled_features = self.projection_pe(resampled_features, batch_size, num_views)
            if proj_dtype is not None and resampled_features.dtype != proj_dtype:
                resampled_features = resampled_features.to(proj_dtype)

        # 3. Project + stitch
        BV, S, D = resampled_features.shape
        try:
            H, W = infer_hw(S)
            x5 = resampled_features.view(batch_size, num_views, H, W, D)
            y5 = self.linear(x5)
            combined = self._stitch_views(y5)
        except Exception:
            # Fallback: simple concat of all tokens
            x = resampled_features.view(batch_size, num_views * S, D)
            combined = self.linear(x)

        # 4. LM dtype alignment
        lm_dtype = resolve_module_dtype(dtype_cache, "language_model", language_model, combined.dtype)
        if lm_dtype is not None and combined.dtype != lm_dtype:
            combined = combined.to(lm_dtype)

        return combined

    # ------------------------------------------------------------------
    # View stitching
    # ------------------------------------------------------------------

    def _stitch_views(self, projected: torch.Tensor) -> torch.Tensor:
        """Stitch projected view features ``[B, V, H, W, D]`` → ``[B, T, D]``."""
        B, V, H, W, D = projected.shape
        ratio = self.overlap_ratio
        k = int(max(0, min(W, round(W * ratio))))
        mode = self.stitching_mode

        try:
            if mode == "concat":
                return projected.permute(0, 2, 1, 3, 4).contiguous().view(B, H * V * W, D)

            if mode == "resample":
                return self._resample_stitch(projected)

            # Default: stride_views – strip overlap columns
            if V <= 1 or k <= 0:
                return projected.permute(0, 2, 1, 3, 4).contiguous().view(B, H * V * W, D)

            parts = [projected[:, 0]]  # [B, H, W, D]
            for v in range(1, V):
                parts.append(projected[:, v, :, k:, :])  # [B, H, W-k, D]
            rowwise = torch.cat(parts, dim=2)  # cat along width
            return rowwise.reshape(B, -1, D)

        except Exception as e:
            logger.warning("Stitching failed, falling back to simple concat: %s", e)
            return projected.permute(0, 2, 1, 3, 4).contiguous().view(B, H * V * W, D)

    def _resample_stitch(self, projected: torch.Tensor) -> torch.Tensor:
        """Differentiable grid-sample blending of overlapping views."""
        B, V, H, W, D = projected.shape
        ratio = self.overlap_ratio
        k = int(max(0, min(W, round(W * ratio))))
        target_cols = self.stitch_target_cols

        if target_cols <= 0:
            unique_cols = W * V - max(0, V - 1) * k
        else:
            unique_cols = target_cols

        s_float = max(float(W - k), float(W)) if k <= 0 else float(W - k)
        T = int(W) if self.stitch_target_to_view_width else int(unique_cols)

        device = projected.device
        g = torch.linspace(0.0, float(unique_cols) - 1.0, steps=T, device=device)
        v_idx = torch.arange(V, device=device, dtype=torch.float32).view(V, 1)
        x_local = g.view(1, T) - (v_idx * s_float)

        xN = projected.permute(0, 1, 4, 2, 3).contiguous().view(B * V, D, H, W)

        y_lin = torch.linspace(-1.0, 1.0, steps=H, device=device) if H > 1 else torch.zeros(1, device=device)
        x_norm = (2.0 * (x_local / max(1.0, float(W - 1)))) - 1.0 if W > 1 else torch.zeros(V, T, device=device)

        x_grid = x_norm.view(V, 1, T).expand(V, H, T)
        y_grid = y_lin.view(1, H, 1).expand(V, H, T)
        grid = torch.stack([x_grid, y_grid], dim=-1)
        grid = grid.unsqueeze(0).expand(B, V, H, T, 2).contiguous().view(B * V, H, T, 2)

        gs_mode = "bilinear" if self.stitch_interp.lower() == "linear" else "nearest"
        sampled = F.grid_sample(xN, grid, mode=gs_mode, padding_mode="zeros", align_corners=True)

        ones = torch.ones(B * V, 1, H, W, device=device, dtype=projected.dtype)
        w_sample = F.grid_sample(ones, grid, mode=gs_mode, padding_mode="zeros", align_corners=True)

        sampled = sampled.view(B, V, D, H, T).permute(0, 1, 3, 4, 2)
        w_sample = w_sample.view(B, V, 1, H, T).permute(0, 1, 3, 4, 2)

        num = (sampled * w_sample).sum(dim=1)
        den = w_sample.sum(dim=1).clamp_min(1e-6)
        fused = num / den
        return fused.reshape(B, -1, D)
