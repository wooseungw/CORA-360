"""Resampler package: factory + all resampler types.

Use :func:`build_resampler` or :class:`ResamplerModule` to construct the
appropriate resampler from a config object.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cora.model.resampler.resamplers import (
    BaseResampler,
    IdentityResampler,
    AvgPoolResampler,
    ConvNeXtBlock,
    ConvResampler,
    QFormerResampler,
    MLPResampler,
)
from cora.model.resampler.perceiver import PerceiverResampler
from cora.model.resampler.spatial_pool import SpatialPoolResampler
from cora.model.resampler.masked_drop import MaskedDropResampler
from cora.model.resampler.c_abstractor import CAbstractorResampler

try:
    from cora.model.resampler.bimamba import BiMambaResampler, BidirectionalMambaResampler
    _BIMAMBA_AVAILABLE = True
except ImportError:
    BiMambaResampler = None  # type: ignore[assignment,misc]
    BidirectionalMambaResampler = None  # type: ignore[assignment,misc]
    _BIMAMBA_AVAILABLE = False

__all__ = [
    "CANONICAL_RESAMPLER_TYPES",
    "build_resampler",
    "ResamplerModule",
    "BaseResampler",
    "IdentityResampler",
    "AvgPoolResampler",
    "ConvNeXtBlock",
    "ConvResampler",
    "QFormerResampler",
    "MLPResampler",
    "PerceiverResampler",
    "SpatialPoolResampler",
    "MaskedDropResampler",
    "CAbstractorResampler",
]

logger = logging.getLogger(__name__)

CANONICAL_RESAMPLER_TYPES = {
    "mlp": "mlp",
    "bimamba": "bimamba",
    "bidirectional_mamba": "bimamba",
    "bi_mamba": "bimamba",
    "qformer": "qformer",
    "identity": "identity",
    "avg": "avg",
    "conv": "conv",
    "perceiver": "perceiver",
    "spatial_pool": "spatial_pool",
    "masked_drop": "masked_drop",
    "c_abstractor": "c_abstractor",
    "cabstractor": "c_abstractor",
}


def build_resampler(
    config: Any,
    vision_hidden_size: int,
) -> BaseResampler:
    """Construct a resampler from *config*.

    Reads ``resampler_type``, ``latent_dimension``, and ``resampler_config``
    from *config* (attribute access).
    """
    raw_type = getattr(config, "resampler_type", "mlp")
    rtype = CANONICAL_RESAMPLER_TYPES.get(raw_type, raw_type)
    in_dim = vision_hidden_size
    out_dim = int(getattr(config, "latent_dimension", 768))

    rcfg_raw = getattr(config, "resampler_config", {})
    if not isinstance(rcfg_raw, dict):
        rcfg: Dict[str, Any] = dict(rcfg_raw) if hasattr(rcfg_raw, "__iter__") else {}
    else:
        rcfg = rcfg_raw

    logger.info("Building resampler: type=%s, in=%d, out=%d", rtype, in_dim, out_dim)

    if rtype == "mlp":
        return MLPResampler(
            vision_dim=in_dim,
            latent_dim=out_dim,
            hidden_dim=rcfg.get("hidden_dim"),
            depth=int(rcfg.get("depth", 3)),
            use_ln=rcfg.get("use_ln", True),
            pool_tokens=rcfg.get("pool_tokens"),
            pool_type=str(rcfg.get("pool_type", "avg")),
        )

    if rtype == "bimamba":
        if not _BIMAMBA_AVAILABLE:
            logger.warning("BiMamba requested but unavailable. Falling back to MLP.")
            return MLPResampler(vision_dim=in_dim, latent_dim=out_dim, depth=int(rcfg.get("num_layers", rcfg.get("depth", 3))))
        return BiMambaResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=rcfg.get("hidden_dim"),
            num_layers=int(rcfg.get("num_layers", rcfg.get("depth", 4))),
            d_state=int(rcfg.get("d_state", 64)),
            d_conv=int(rcfg.get("d_conv", 4)),
            expand=float(rcfg.get("expand", 2.0)),
            dropout=float(rcfg.get("dropout", 0.0)),
            norm_first=bool(rcfg.get("norm_first", True)),
        )

    if rtype == "qformer":
        return QFormerResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            num_query=int(rcfg.get("num_query_tokens", 64)),
            num_hidden_layers=int(rcfg.get("num_hidden_layers", 6)),
            num_attention_heads=int(rcfg.get("num_attention_heads", 8)),
        )

    if rtype == "identity":
        return IdentityResampler(in_dim, out_dim)

    if rtype == "avg":
        return AvgPoolResampler(in_dim, out_dim)

    if rtype == "conv":
        return ConvResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            num_tokens=int(rcfg.get("num_tokens", 144)),
            depths=rcfg.get("depths", [2, 2]),
            drop_path_rate=float(rcfg.get("drop_path_rate", 0.0)),
        )

    if rtype == "perceiver":
        return PerceiverResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            num_latents=int(rcfg.get("num_latents", 64)),
            depth=int(rcfg.get("depth", 6)),
            heads=int(rcfg.get("heads", 8)),
            dim_head=int(rcfg.get("dim_head", 64)),
            ff_mult=int(rcfg.get("ff_mult", 4)),
        )

    if rtype == "spatial_pool":
        return SpatialPoolResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            pool_size=int(rcfg.get("pool_size", 4)),
            pool_mode=str(rcfg.get("pool_mode", "average")),
            depth=int(rcfg.get("depth", 2)),
        )

    if rtype == "masked_drop":
        return MaskedDropResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            mask_ratio=float(rcfg.get("mask_ratio", 0.5)),
            mask_mode=str(rcfg.get("mask_mode", "fixed")),
            skip_percentage=float(rcfg.get("skip_percentage", 0.0)),
            ratio_lower=float(rcfg.get("ratio_lower", 0.3)),
            ratio_upper=float(rcfg.get("ratio_upper", 0.7)),
            depth=int(rcfg.get("depth", 2)),
        )

    if rtype == "c_abstractor":
        return CAbstractorResampler(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=rcfg.get("hidden_dim"),
            depth=int(rcfg.get("depth", 3)),
            num_queries=int(rcfg.get("num_queries", 0)),
            kernel_size=int(rcfg.get("kernel_size", 7)),
            expand=int(rcfg.get("expand", 4)),
            se_reduction=int(rcfg.get("se_reduction", 4)),
            mlp_depth=int(rcfg.get("mlp_depth", 2)),
            drop_path=float(rcfg.get("drop_path", 0.0)),
        )

    raise ValueError(
        f"Unknown resampler_type: {rtype!r}. "
        f"Supported: {', '.join(sorted(set(CANONICAL_RESAMPLER_TYPES.values())))}"
    )


class ResamplerModule(nn.Module):
    """Thin wrapper that builds and holds a resampler from config.

    Provides ``forward(vision_features, target_dtype=None)`` and exposes
    :attr:`output_dim`.
    """

    def __init__(self, config: Any, input_dim: int) -> None:
        super().__init__()
        self.resampler = build_resampler(config, input_dim)
        self.output_dim = int(getattr(config, "latent_dimension", 768))

    def forward(
        self,
        vision_features: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if target_dtype is not None and vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        return self.resampler(vision_features)
