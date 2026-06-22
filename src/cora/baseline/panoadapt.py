"""PanoAdapt module for panoramic adaptation of Qwen2.5-VL.

Provides three components for adapting Qwen2.5-VL to panoramic inputs:

1. :func:`compute_panoramic_position_ids` — M-RoPE position IDs with
   overlap-aware width continuity so adjacent panoramic views share
   consistent spatial encodings.

2. :class:`VisionFeatureHook` — Forward hook to extract intermediate
   vision features from Qwen2.5-VL's PatchMerger for auxiliary loss.

3. :class:`PanoAdaptDenseCLLoss` — DenseCL-style InfoNCE loss computed on
   overlapping regions between adjacent panoramic views, adapted for
   post-PatchMerger features.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cora.baseline.config import PanoAdaptConfig

__all__ = [
    "compute_panoramic_position_ids",
    "VisionFeatureHook",
    "PanoAdaptDenseCLLoss",
    "PanoAdaptVICRegLoss",
    "create_panoadapt_loss",
    "VLMAdapter",
    "QwenVLAdapter",
    "InternVLAdapter",
    "Gemma3Adapter",
    "create_vlm_adapter",
]


# ---------------------------------------------------------------------------
# Component 1: Panoramic M-RoPE Position IDs
# ---------------------------------------------------------------------------


def compute_panoramic_position_ids(
    num_views: int,
    grid_h: int,
    grid_w: int,
    overlap_ratio: float = 0.5,
    spatial_merge_size: int = 2,
    wrap_around: bool = True,
) -> torch.Tensor:
    """Compute M-RoPE position IDs with overlap-aware continuity on width axis.

    Qwen2.5-VL uses 3-axis M-RoPE: (temporal, height, width).
    This function modifies the **width** axis so that overlapping regions
    between adjacent views share nearby position IDs, encouraging the
    model to learn spatial continuity across panoramic tile boundaries.

    The core overlap stride pattern is adapted from CORA's
    ``PanoramaPositionalEncoding._yaw_encoding()`` (positional.py:122-145)::

        s = 1.0 - overlap_ratio        # stride
        global_col = v * s * grid_w + c

    This ensures that view *v*'s rightmost ``k`` columns (where
    ``k = int(grid_w * overlap_ratio)``) receive the same position IDs
    as view *(v+1)*'s leftmost ``k`` columns.

    Args:
        num_views: Number of panoramic views (e.g. 8 yaw tiles).
        grid_h: Height of the patch grid per view after PatchMerger
            (e.g. 12 for 672×672 input with patch=14, merge=2).
        grid_w: Width of the patch grid per view after PatchMerger.
        overlap_ratio: Fraction of width that overlaps between adjacent views.
            Must be in ``[0, 1)``. Default 0.5 (50% horizontal overlap).
        spatial_merge_size: PatchMerger pooling factor (default 2 for
            Qwen2.5-VL). Provided for reference; does not affect the
            position ID computation since *grid_h*/*grid_w* already
            account for merging.
        wrap_around: If ``True``, the last view's right edge wraps to the
            first view's left edge (360° panorama continuity).

    Returns:
        ``torch.LongTensor`` of shape ``[3, total_vision_tokens]`` where
        ``total_vision_tokens = num_views * grid_h * grid_w``.
        Row 0 = temporal, row 1 = height, row 2 = width.
    """
    V, H, W = num_views, grid_h, grid_w

    # --- Temporal axis: view index for all tokens in that view ---
    temporal = (
        torch.arange(V)
        .view(V, 1, 1)
        .expand(V, H, W)
        .reshape(-1)
    )

    # --- Height axis: per-row index within each view (0 .. H-1) ---
    height = (
        torch.arange(H)
        .view(1, H, 1)
        .expand(V, H, W)
        .reshape(-1)
    )

    # --- Width axis: overlap-aware global column positions ---
    # Pattern from CORA positional.py:133-138:
    #   s = 1.0 - overlap_ratio
    #   g = v_idx * s + x  (continuous → here we use integer grid)
    stride = 1.0 - overlap_ratio
    v_idx = torch.arange(V, dtype=torch.float32).view(V, 1)
    c_idx = torch.arange(W, dtype=torch.float32).view(1, W)
    global_col = v_idx * stride * W + c_idx  # [V, W]

    if wrap_around:
        # Total effective width for 360° wrapping.
        # With V views and stride s, the panorama spans V*s*W unique columns
        # before the pattern repeats.
        total_width = float(V) * stride * W
        global_col = torch.fmod(global_col, max(total_width, 1.0))

    width = (
        global_col
        .round()
        .long()
        .view(V, 1, W)
        .expand(V, H, W)
        .reshape(-1)
    )

    # Stack to [3, V*H*W]
    position_ids = torch.stack([temporal.long(), height.long(), width], dim=0)
    return position_ids


# ---------------------------------------------------------------------------
# Component 2: Vision Feature Hook
# ---------------------------------------------------------------------------


class VisionFeatureHook:
    """Forward hook to extract vision features from Qwen2.5-VL's PatchMerger.

    The PatchMerger performs 2×2 spatial pooling followed by linear
    projection on patch embeddings.  By attaching a hook on its output we
    can extract intermediate vision features for auxiliary loss computation
    (e.g. DenseCL) without modifying the model code itself.

    Usage::

        hook = VisionFeatureHook()
        hook.register(model)  # model = Qwen2_5_VLForConditionalGeneration
        # ... forward pass ...
        features = hook.get_features()  # [num_tokens, hidden_dim]
        hook.clear()
        # ... when done ...
        hook.remove()
    """

    def __init__(self) -> None:
        self._features: Optional[torch.Tensor] = None
        self._hook_handle: Optional[Any] = None  # torch.utils.hooks.RemovableHandle

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,  # noqa: A002 — PyTorch hook convention
        output: Any,
    ) -> None:
        """Store the forward output of the hooked module."""
        self._features = output

    def register(self, model: nn.Module, hook_target_name: Optional[str] = None) -> None:
        """Attach a forward hook to the model's vision projection module.

        Resolution order:

        1. Explicit *hook_target_name* (from :class:`VLMAdapter`).
        2. Canonical Qwen path ``model.visual.merger``.
        3. Fallback substring search for ``"merger"`` or
           ``"multi_modal_projector"``.

        Args:
            model: A VLM model instance (Qwen2.5-VL, InternVL, Gemma3, …).
            hook_target_name: Optional module-name hint from a
                :class:`VLMAdapter`.  Matched via ``named_modules()``.

        Raises:
            ValueError: If no suitable vision projection module is found.
        """
        self.remove()

        target: Optional[nn.Module] = None

        if hook_target_name is not None:
            for name, module in model.named_modules():
                if name == hook_target_name or name.endswith(hook_target_name):
                    target = module
                    logger.info("VisionFeatureHook: matched explicit target '%s'", name)
                    break

        if target is None:
            try:
                visual = getattr(model, "visual", None)
                if visual is not None:
                    target = getattr(visual, "merger", None)
            except Exception:  # noqa: BLE001
                pass

        if target is None:
            _SEARCH_TERMS = ("merger", "multi_modal_projector")
            for name, module in model.named_modules():
                name_lower = name.lower()
                if any(term in name_lower for term in _SEARCH_TERMS):
                    target = module
                    logger.info("VisionFeatureHook: found target at '%s'", name)
                    break

        if target is None:
            raise ValueError(
                "Could not find a vision projection module. "
                "Expected 'model.visual.merger' (Qwen), "
                "'multi_modal_projector' (InternVL/Gemma3), "
                "or pass hook_target_name explicitly."
            )

        self._hook_handle = target.register_forward_hook(self._hook_fn)
        logger.info("VisionFeatureHook registered on %s", type(target).__name__)

    def get_features(self, detach: bool = False) -> Optional[torch.Tensor]:
        """Return stored features from the last forward pass.

        Args:
            detach: If ``True``, return a detached copy (no gradient).

        Returns:
            Tensor of shape ``[num_tokens, hidden_dim]`` or ``None`` if
            no forward pass has been executed since the last ``clear()``.
        """
        if self._features is None:
            return None
        return self._features.detach() if detach else self._features

    def clear(self) -> None:
        """Reset stored features to ``None``."""
        self._features = None

    def remove(self) -> None:
        """Remove the forward hook and clear stored features."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.clear()

    @property
    def has_features(self) -> bool:
        """Whether features are available from a previous forward pass."""
        return self._features is not None


# ---------------------------------------------------------------------------
# Component 3: PanoAdapt DenseCL Loss
# ---------------------------------------------------------------------------


class PanoAdaptDenseCLLoss(nn.Module):
    """DenseCL loss adapter for VLM features.

    Computes symmetric InfoNCE loss on overlapping regions between adjacent
    panoramic views, similar to CORA's
    :class:`~cora.training.losses.DenseCLLoss` but adapted for
    post-PatchMerger features from :class:`VisionFeatureHook`.

    The loss encourages spatially corresponding patches in overlap zones
    to have similar representations, providing a self-supervised panoramic
    consistency signal during LoRA finetuning.

    Key difference from CORA's ``DenseCLLoss``:

    - Operates on ``[V, H, W, D]`` features (no batch dimension) since
      VisionFeatureHook captures features for the *entire* batch at once.
      The batch dimension is handled by the caller (collate/trainer).
    - Also accepts flat ``[total_tokens, D]`` format with explicit grid
      dimensions for flexibility.

    Parameters
    ----------
    overlap_ratio : float
        Fraction of spatial width that overlaps between adjacent tiles.
        Should match the image processing overlap setting.
    temperature : float
        Softmax temperature for InfoNCE.  Lower → sharper similarity peak.
        Default 0.07, empirically validated in CORA experiments.
    """

    def __init__(
        self,
        overlap_ratio: float = 0.5,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.overlap_ratio = overlap_ratio
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        num_views: int,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE on overlap regions.

        Accepts features in two formats:

        1. Pre-shaped ``[V, H, W, D]`` — used directly.
        2. Flat ``[total_tokens, D]`` — reshaped using *num_views*,
           *grid_h*, *grid_w* (grid dims inferred as square if omitted).

        Args:
            features: Vision features, either ``[V, H, W, D]`` or
                ``[total_tokens, D]``.
            num_views: Number of views (*V*).
            grid_h: Patch grid height (inferred if features are 4-D).
            grid_w: Patch grid width (inferred if features are 4-D).
            overlap_ratio: Per-call override for the phys overlap fraction.
                When None, falls back to ``self.overlap_ratio``. Used by the
                random-aug path so each batch slices the same fraction of
                columns its tiles actually share.

        Returns:
            Scalar loss tensor (gradient-carrying).
        """
        if num_views <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # --- Reshape to [V, H, W, D] ---
        if features.ndim == 2:
            D = features.shape[-1]
            total = features.shape[0]
            if grid_h is not None and grid_w is not None:
                H, W = grid_h, grid_w
            else:
                tokens_per_view = total // num_views
                H = W = int(math.isqrt(tokens_per_view))
            grid = features.view(num_views, H, W, D)
        elif features.ndim == 3:
            V, Tokens, D = features.shape
            if grid_h is not None and grid_w is not None:
                H, W = grid_h, grid_w
            else:
                H = W = int(math.isqrt(Tokens))
            grid = features.view(V, H, W, D)
        elif features.ndim == 4:
            grid = features
        else:
            raise ValueError(
                f"Expected features with 2, 3 or 4 dims, got {features.ndim}"
            )

        orig_dtype = grid.dtype
        grid = grid.float()
        V, H, W, D = grid.shape

        ratio = float(overlap_ratio) if overlap_ratio is not None else float(self.overlap_ratio)
        k = max(1, int(W * ratio))
        if k == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Extract overlap regions:
        #   right k cols of view v  ↔  left k cols of view v+1
        # torch.roll with dims=0 gives circular wrapping (last → first view)
        # Pattern from CORA DenseCLLoss (losses.py:579-581)
        curr_right = grid[:, :, -k:, :]                           # [V, H, k, D]
        next_left = torch.roll(grid, -1, dims=0)[:, :, :k, :]    # [V, H, k, D]

        N = H * k  # number of spatial positions per view-pair

        # Reshape to [V, N, D] and L2-normalise
        Q = F.normalize(curr_right.reshape(V, N, D), dim=-1)
        K = F.normalize(next_left.reshape(V, N, D), dim=-1)

        # Batched similarity [V, N, N]
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.temperature

        labels = torch.arange(N, device=features.device).unsqueeze(0).expand(V, -1)

        # Symmetric InfoNCE: q→k + k→q
        VN = V * N
        loss_q2k = F.cross_entropy(logits.reshape(VN, N), labels.reshape(VN))
        loss_k2q = F.cross_entropy(
            logits.transpose(1, 2).contiguous().reshape(VN, N),
            labels.reshape(VN),
        )

        return (0.5 * (loss_q2k + loss_k2q)).to(orig_dtype)


class PanoAdaptVICRegLoss(nn.Module):
    """VICReg loss adapter for overlap regions between adjacent panoramic views.

    Adapts CORA's overlap VICReg formulation to PanoAdapt's interface where
    features are extracted from Qwen2.5-VL PatchMerger without an explicit batch
    dimension.

    Supports two variance/covariance modes:

    - ``"batchwise"``: aggregate statistics over all overlap tokens.
    - ``"pairwise"``: compute statistics independently per adjacent view pair.
    """

    def __init__(
        self,
        mode: str = "batchwise",
        overlap_ratio: float = 0.5,
        similarity_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if mode not in {"batchwise", "pairwise"}:
            raise ValueError(f"Unsupported VICReg mode: {mode}")

        self.mode = mode
        self.overlap_ratio = overlap_ratio
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        num_views: int,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute overlap VICReg loss on adjacent panoramic view pairs.

        ``overlap_ratio`` overrides ``self.overlap_ratio`` for a single call —
        used by the random (hfov, overlap) augmentation path so the slice
        fraction tracks each batch's realized geometry.
        """
        if num_views <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        orig_dtype = features.dtype
        features = features.float()

        # --- Reshape to [V, H, W, D] ---
        if features.ndim == 2:
            dim = features.shape[-1]
            total = features.shape[0]
            if grid_h is not None and grid_w is not None:
                grid_h_val, grid_w_val = grid_h, grid_w
            else:
                tokens_per_view = total // num_views
                grid_h_val = grid_w_val = int(math.isqrt(tokens_per_view))
            grid = features.view(num_views, grid_h_val, grid_w_val, dim)
        elif features.ndim == 3:
            num_view_tokens, num_tokens_per_view, dim = features.shape
            if grid_h is not None and grid_w is not None:
                grid_h_val, grid_w_val = grid_h, grid_w
            else:
                grid_h_val = grid_w_val = int(math.isqrt(num_tokens_per_view))
            grid = features.view(num_view_tokens, grid_h_val, grid_w_val, dim)
        elif features.ndim == 4:
            grid = features
        else:
            raise ValueError(f"Expected features with 2, 3, or 4 dims, got {features.ndim}")

        num_views_local, grid_h_local, grid_w_local, dim = grid.shape

        ratio = float(overlap_ratio) if overlap_ratio is not None else float(self.overlap_ratio)
        k = max(1, int(grid_w_local * ratio))

        curr_right = grid[:, :, -k:, :]  # [V, H, k, D]
        next_left = torch.roll(grid, -1, dims=0)[:, :, :k, :]  # [V, H, k, D]

        num_pairs = num_views_local
        tokens_per_pair = grid_h_local * k
        curr = curr_right.contiguous().view(num_pairs, tokens_per_pair, dim)
        nxt = next_left.contiguous().view(num_pairs, tokens_per_pair, dim)

        inv_pair = F.mse_loss(curr, nxt, reduction="none").mean(dim=(1, 2))

        if self.mode == "batchwise":
            combined = torch.cat([curr, nxt], dim=0).reshape(-1, dim)
            std_all = torch.sqrt(combined.var(dim=0, unbiased=False) + self.eps)
            var_pair = F.relu(self.gamma - std_all).mean().expand(num_pairs)
        else:
            std_c = torch.sqrt(curr.var(dim=1, unbiased=False) + self.eps)
            std_n = torch.sqrt(nxt.var(dim=1, unbiased=False) + self.eps)
            var_pair = 0.5 * (
                F.relu(self.gamma - std_c).mean(dim=1)
                + F.relu(self.gamma - std_n).mean(dim=1)
            )

        if self.mode == "batchwise":
            combined = torch.cat([curr, nxt], dim=0).reshape(-1, dim)
            cc = combined - combined.mean(dim=0, keepdim=True)
            cov = (cc.T @ cc) / max(combined.size(0) - 1, 1)
            cov_clone = cov.clone()
            cov_clone.diagonal().zero_()
            cov_pair = ((cov_clone ** 2).sum() / dim).expand(num_pairs)
        else:
            curr_c = curr - curr.mean(dim=1, keepdim=True)
            nxt_c = nxt - nxt.mean(dim=1, keepdim=True)
            denom = max(tokens_per_pair - 1, 1)

            def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
                C = torch.bmm(xc.transpose(1, 2), xc) / denom
                C2 = (C ** 2).sum(dim=(1, 2))
                diag2 = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                return (C2 - diag2) / dim

            cov_pair = 0.5 * (_cov_offdiag_sq_mean(curr_c) + _cov_offdiag_sq_mean(nxt_c))

        per_pair = (
            self.similarity_weight * inv_pair
            + self.variance_weight * var_pair
            + self.covariance_weight * cov_pair
        )
        total = per_pair.mean()

        return torch.clamp(total, max=1e6).to(orig_dtype)


def create_panoadapt_loss(config: "PanoAdaptConfig") -> torch.nn.Module:
    """Create overlap loss module from PanoAdapt configuration."""
    loss_type = config.overlap_loss_type.lower()

    if loss_type == "densecl":
        return PanoAdaptDenseCLLoss(
            overlap_ratio=config.overlap_ratio,
            temperature=config.overlap_loss_temperature,
        )
    if loss_type == "vicreg_batchwise":
        return PanoAdaptVICRegLoss(
            mode="batchwise",
            overlap_ratio=config.overlap_ratio,
            similarity_weight=config.vicreg_sim_weight,
            variance_weight=config.vicreg_var_weight,
            covariance_weight=config.vicreg_cov_weight,
        )
    if loss_type == "vicreg_pairwise":
        return PanoAdaptVICRegLoss(
            mode="pairwise",
            overlap_ratio=config.overlap_ratio,
            similarity_weight=config.vicreg_sim_weight,
            variance_weight=config.vicreg_var_weight,
            covariance_weight=config.vicreg_cov_weight,
        )

    raise ValueError(f"Unsupported panoadapt overlap_loss_type: {config.overlap_loss_type}")


# ---------------------------------------------------------------------------
# Component 4: VLM Adapter Abstraction
# ---------------------------------------------------------------------------


def _split_consecutive_groups(positions: torch.Tensor) -> List[torch.Tensor]:
    """Split a sorted 1-D index tensor into groups of consecutive indices."""
    if positions.numel() == 0:
        return []
    diffs = positions[1:] - positions[:-1]
    split_at = (diffs > 1).nonzero(as_tuple=True)[0] + 1
    return list(torch.tensor_split(positions, split_at.tolist()))


class VLMAdapter(ABC):
    """Dispatch layer abstracting model-specific PanoAdapt operations.

    Five abstract methods decouple PanoAdapt from Qwen-specific internals
    so InternVL and Gemma3 can reuse the same trainer path.
    """

    def __init__(self, overlap_ratio: float = 0.5, include_global: bool = True) -> None:
        self.overlap_ratio = overlap_ratio
        self.include_global = include_global

    def reorder_vision_features(
        self,
        features: torch.Tensor,
        model: nn.Module,
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return hooked vision features in row-major ``[tile, row, col]`` order.

        Default identity: InternVL/Gemma ``multi_modal_projector`` outputs are
        already row-major, so the overlap/VICReg edge slicing lines up.
        Overridden by :class:`QwenVLAdapter` to undo Qwen2.5-VL's
        window-attention permutation (its merger hook fires *before*
        ``visual.forward`` reapplies ``reverse_indices``).
        """
        return features

    @abstractmethod
    def compute_rope_inputs(
        self, model: nn.Module, inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def modify_position_ids(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_info: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def get_vision_hook_target(self) -> str:
        ...

    @abstractmethod
    def get_image_token_id(self, model: nn.Module) -> int:
        ...

    @abstractmethod
    def get_spatial_merge_size(self, model: nn.Module) -> int:
        ...


# ---------------------------------------------------------------------------
# QwenVLAdapter — wraps existing 3-D M-RoPE logic (backward-compatible)
# ---------------------------------------------------------------------------


class QwenVLAdapter(VLMAdapter):
    """Qwen2.5-VL / Qwen2-VL adapter (3-D M-RoPE width-axis shift)."""

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
        image_grid_info: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        config = model.config
        image_token_id: int = config.image_token_id
        spatial_merge: int = config.vision_config.spatial_merge_size
        image_grid_thw: torch.Tensor = image_grid_info

        position_ids = position_ids.clone()

        for batch_idx in range(input_ids.shape[0]):
            is_image = input_ids[batch_idx] == image_token_id
            if not is_image.any():
                continue

            image_positions = is_image.nonzero(as_tuple=True)[0]
            pos = 0
            for view_idx in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[view_idx].tolist()
                llm_w = w // spatial_merge
                n_tokens = int(t * (h // spatial_merge) * llm_w)
                if pos + n_tokens > len(image_positions):
                    break

                view_positions = image_positions[pos: pos + n_tokens]

                if self.include_global and view_idx == 0:
                    pos += n_tokens
                    continue

                tile_idx = view_idx - (1 if self.include_global else 0)
                # Qwen `get_rope_index` lays out adjacent images with width-axis gap
                # ≈ llm_w. To create overlap_ratio overlap, shrink that gap to
                # stride * llm_w by shifting tile i back by overlap * llm_w * i.
                pano_shift = -int(round(tile_idx * self.overlap_ratio * llm_w))
                position_ids[2, batch_idx, view_positions] += pano_shift

                pos += n_tokens

        return position_ids

    def get_vision_hook_target(self) -> str:
        return "merger"

    def reorder_vision_features(
        self,
        features: torch.Tensor,
        model: nn.Module,
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Undo Qwen2.5-VL's window-attention token permutation.

        The hook captures ``visual.merger``'s output, but in Qwen2.5-VL the
        merger runs on window-permuted tokens; ``visual.forward`` only restores
        row-major order *after* the merger via
        ``hidden_states[argsort(window_index)]`` (modeling_qwen2_5_vl.py). So the
        hooked features are window-permuted and the overlap loss would slice
        non-corresponding patches. We reproduce that final reorder here using the
        model's own ``get_window_index`` (exact, version-robust). Qwen2-VL has no
        window attention and no ``get_window_index`` → falls through unchanged.
        """
        if features is None or image_grid_thw is None:
            return features
        visual = getattr(model, "visual", None)
        if visual is None:
            inner = getattr(model, "model", None)
            visual = getattr(inner, "visual", None) if inner is not None else None
        if visual is None or not hasattr(visual, "get_window_index"):
            return features
        try:
            window_index, _ = visual.get_window_index(image_grid_thw)
        except Exception:  # noqa: BLE001 — never let a diagnostic helper kill training
            logger.warning("QwenVLAdapter.reorder_vision_features: get_window_index failed; using raw order")
            return features
        if window_index.shape[0] != features.shape[0]:
            # Merged-token counts must match for the reorder to be valid.
            logger.warning(
                "QwenVLAdapter.reorder_vision_features: window_index len %d != features %d; skipping reorder",
                int(window_index.shape[0]), int(features.shape[0]),
            )
            return features
        reverse_indices = torch.argsort(window_index).to(features.device)
        return features.index_select(0, reverse_indices)

    def get_image_token_id(self, model: nn.Module) -> int:
        return int(model.config.image_token_id)

    def get_spatial_merge_size(self, model: nn.Module) -> int:
        return int(model.config.vision_config.spatial_merge_size)


# ---------------------------------------------------------------------------
# PanoRoPE-1D base for InternVL / Gemma3
# ---------------------------------------------------------------------------


class _PanoRoPE1DAdapter(VLMAdapter):
    """Shared PanoRoPE-1D logic for models with standard 1-D position IDs.

    Subclasses set ``_IMAGE_TOKEN_ID`` and ``_SPATIAL_MERGE_SIZE`` and
    override ``get_vision_hook_target`` / ``get_image_token_id`` as needed.
    """

    _IMAGE_TOKEN_ID: int = 0
    _SPATIAL_MERGE_SIZE: int = 1

    def compute_rope_inputs(
        self, model: nn.Module, inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).clone()
        return {"position_ids": position_ids}

    def modify_position_ids(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_info: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        position_ids = position_ids.clone()
        image_token_id = self.get_image_token_id(model)

        for batch_idx in range(input_ids.shape[0]):
            is_image = input_ids[batch_idx] == image_token_id
            if not is_image.any():
                continue

            image_positions = is_image.nonzero(as_tuple=True)[0]
            views = _split_consecutive_groups(image_positions)

            for view_idx, view_pos in enumerate(views):
                if self.include_global and view_idx == 0:
                    continue

                tile_idx = view_idx - (1 if self.include_global else 0)
                tokens_per_view = len(view_pos)
                # Default arange-PIDs place adjacent views at gap = tokens_per_view.
                # Panorama goal: adjacent views overlap by overlap_ratio fraction,
                # i.e. shrink the gap to (1 - overlap_ratio) * tokens_per_view.
                # Shift = (target_gap - default_gap) * tile_idx = -overlap * T * tile_idx.
                pano_shift = -int(round(tile_idx * self.overlap_ratio * tokens_per_view))
                position_ids[batch_idx, view_pos] += pano_shift

        return position_ids

    def get_spatial_merge_size(self, model: nn.Module) -> int:
        return self._SPATIAL_MERGE_SIZE


# ---------------------------------------------------------------------------
# InternVLAdapter
# ---------------------------------------------------------------------------


class InternVLAdapter(_PanoRoPE1DAdapter):
    """InternVL3.5 adapter — PanoRoPE-1D, hook on ``multi_modal_projector``."""

    _IMAGE_TOKEN_ID = 151671  # <IMG_CONTEXT>
    _SPATIAL_MERGE_SIZE = 1

    def get_vision_hook_target(self) -> str:
        return "multi_modal_projector"

    def get_image_token_id(self, model: nn.Module) -> int:
        ctx_id = getattr(model, "img_context_token_id", None)
        if ctx_id is not None:
            return int(ctx_id)
        return self._IMAGE_TOKEN_ID


# ---------------------------------------------------------------------------
# Gemma3Adapter
# ---------------------------------------------------------------------------


class Gemma3Adapter(_PanoRoPE1DAdapter):
    """Gemma3 adapter — PanoRoPE-1D, hook on ``multi_modal_projector``."""

    _IMAGE_TOKEN_ID = 262144
    _SPATIAL_MERGE_SIZE = 1

    def get_vision_hook_target(self) -> str:
        return "multi_modal_projector"

    def get_image_token_id(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("image_token_id", "image_token_index"):
                val = getattr(config, attr, None)
                if val is not None:
                    return int(val)
        return self._IMAGE_TOKEN_ID


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, type] = {
    "qwen_vl": QwenVLAdapter,
    "qwen25_vl": QwenVLAdapter,
    "qwenvl": QwenVLAdapter,
    "qwen2_vl": QwenVLAdapter,
    "internvl": InternVLAdapter,
    "internvl_chat": InternVLAdapter,
    "gemma3": Gemma3Adapter,
    "gemma-3": Gemma3Adapter,
}


def create_vlm_adapter(
    model_type: str,
    overlap_ratio: float = 0.5,
    include_global: bool = True,
) -> VLMAdapter:
    """Create the appropriate :class:`VLMAdapter` for *model_type*."""
    key = model_type.lower()
    if key == "qwen_vl_erp_rope" and key not in _ADAPTER_REGISTRY:
        # Lazy import — Dense360 §4.1 ERP-RoPE adapter (rebuttal-only).
        from cora.baseline.panoadapt_erp_rope import register as _register_erp_rope

        _register_erp_rope()
    cls = _ADAPTER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No VLMAdapter for model_type={model_type!r}. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return cls(overlap_ratio=overlap_ratio, include_global=include_global)
