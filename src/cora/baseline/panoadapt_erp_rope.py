"""ERP-RoPE-style adapter (Dense360 §4.1 re-implementation) for Qwen2.5-VL.

This module is a **rebuttal-only** experimental adapter for the C1 head-to-head
comparison requested by R2 (cv7f) of CORA. It re-implements the position-ID
remapping described in Dense360 (arXiv:2506.14471) §4.1 eqs 1–4 on top of the
unmodified Qwen2.5-VL M-RoPE width axis.

It is *not* wired into the default training path. To use it, register the class
in `panoadapt._ADAPTER_REGISTRY` and select `model_type: qwen_vl_erp_rope` in a
config that also routes the image processor through `processors/erp_native.py`
(to preserve the 2:1 ERP aspect ratio).

Faithfulness disclaimer (matches docs/REBUTTAL_DRAFT.md M2):
    Dense360 §4.1 does not specify (a) whether γ·f(w) is integer-cast, (b) which
    M-RoPE axis carries the remap, or (c) f(w) indexing convention. We chose:
        - position_ids[2] (width) carries γ·f(w);
        - integer floor cast for compatibility with Qwen's RoPE consumer;
        - 1-based fold f(w) = [1, 2, …, ⌈W/2⌉, ⌈W/2⌉, …, 2, 1] (length W).
    These choices are documented in the paper appendix.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from cora.baseline.panoadapt import VLMAdapter


def _erprope_gamma(grid_h: int) -> float:
    """Area-preserving global scalar γ = H / Σ_θ cos(θ).

    θ_h = (h + 0.5) * π / H − π/2 (centered latitude per row, 0 at equator,
    ±π/2 at poles). We use the centered-cell convention so the top row
    is not exactly at θ = −π/2 (which would zero out the cosine).
    """
    if grid_h <= 0:
        return 1.0
    h_idx = torch.arange(grid_h, dtype=torch.float64)
    theta = (h_idx + 0.5) * math.pi / grid_h - math.pi / 2.0
    denom = torch.cos(theta).sum().item()
    if denom <= 1e-9:
        return 1.0
    return grid_h / denom


def _erprope_fold(grid_w: int) -> torch.Tensor:
    """Symmetric width fold f(w) of length W, peaks at the center, ends fold to 1.

    Even W:  [1, 2, …, W/2, W/2, …, 2, 1]
    Odd  W:  [1, 2, …, (W+1)/2, …, 2, 1]
    """
    if grid_w <= 0:
        return torch.zeros(0, dtype=torch.long)
    half = (grid_w + 1) // 2
    rising = torch.arange(1, half + 1, dtype=torch.long)
    if grid_w % 2 == 0:
        # [1..W/2, W/2..1]
        falling = rising.flip(0)
    else:
        # [1..(W+1)/2, ((W-1)/2)..1] — drop the duplicate of the peak
        falling = rising[:-1].flip(0)
    return torch.cat([rising, falling])


class ERPRoPEQwenAdapter(VLMAdapter):
    """Dense360-style ERP-RoPE for Qwen2.5-VL on a SINGLE ERP image input.

    Assumes the data path produces ``image_grid_thw`` of shape ``[1, T_grid, H_grid, W_grid]``
    with a single (T=1) entry. PanoRoPE / multi-view shifts do not apply here; this
    adapter REPLACES the M-RoPE width axis with ⌊γ · f(w)⌋ for image tokens, and leaves
    text tokens alone.
    """

    def compute_rope_inputs(
        self, model: nn.Module, inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        position_ids, rope_deltas = model.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        return {"position_ids": position_ids, "rope_deltas": rope_deltas}

    def modify_position_ids(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_info: Any,  # image_grid_thw
        model: nn.Module,
    ) -> torch.Tensor:
        config = model.config
        image_token_id: int = config.image_token_id
        spatial_merge: int = config.vision_config.spatial_merge_size
        image_grid_thw: torch.Tensor = image_grid_info  # [num_images, 3]

        position_ids = position_ids.clone()
        device = position_ids.device

        for batch_idx in range(input_ids.shape[0]):
            is_image = input_ids[batch_idx] == image_token_id
            if not is_image.any():
                continue

            image_positions = is_image.nonzero(as_tuple=True)[0]
            pos = 0
            for view_idx in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[view_idx].tolist()
                grid_h_local = h // spatial_merge
                grid_w_local = w // spatial_merge
                n_tokens = int(t * grid_h_local * grid_w_local)
                if pos + n_tokens > len(image_positions):
                    break

                view_positions = image_positions[pos: pos + n_tokens]

                # Compute γ · f(w) for this image's grid (all in float64 for stability).
                gamma = _erprope_gamma(grid_h_local)
                fold = _erprope_fold(grid_w_local).to(device=device)  # [W]
                # Build a [grid_h_local, grid_w_local] map of ⌊γ·f(w)⌋, broadcast across t.
                wpe = torch.floor(gamma * fold.to(torch.float64)).to(torch.long)
                # Layout in get_rope_index iterates t → h → w; one image has T=1 typically.
                width_pids = wpe.unsqueeze(0).expand(grid_h_local, -1)         # [H, W]
                width_pids = width_pids.unsqueeze(0).expand(t, -1, -1)         # [T, H, W]
                width_pids = width_pids.reshape(-1)                            # [T*H*W]

                position_ids[2, batch_idx, view_positions] = width_pids[: view_positions.numel()]

                # Optional: leave height (position_ids[1]) untouched — defaults to h-axis index
                # (matches Dense360's silence on the height axis).

                pos += n_tokens

        return position_ids

    def get_vision_hook_target(self) -> str:
        # Same hook target as plain Qwen — we only change PIDs, not vision features.
        return "merger"

    def get_image_token_id(self, model: nn.Module) -> int:
        return int(model.config.image_token_id)

    def get_spatial_merge_size(self, model: nn.Module) -> int:
        return int(model.config.vision_config.spatial_merge_size)


# ---------------------------------------------------------------------------
# Registration helper — call this once at import time of the experimental config.
# ---------------------------------------------------------------------------

def register() -> None:
    """Add ``qwen_vl_erp_rope`` to the panoadapt registry. Idempotent."""
    from cora.baseline import panoadapt as _pa

    if "qwen_vl_erp_rope" not in _pa._ADAPTER_REGISTRY:
        _pa._ADAPTER_REGISTRY["qwen_vl_erp_rope"] = ERPRoPEQwenAdapter
