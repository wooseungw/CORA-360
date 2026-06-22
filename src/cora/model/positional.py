"""Panorama-aware positional encoding with yaw-continuity support.

Provides sinusoidal and Fourier-based spherical encodings that respect the
360° wrap-around property of panoramic images, ensuring adjacent views share
consistent positional signals in overlapping regions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "PanoramaPositionalEncoding",
    "PanoramaPositionalEncoding2",
    "PanoramaYawRoPE",
    "attach_yaw_rope_hook",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def infer_hw(num_patches: int) -> Tuple[int, int]:
    """Infer (height, width) from a total patch count assuming near-square grid.

    Tries the exact integer square root first, then expands minimally so that
    ``h * w >= num_patches``.
    """
    h = w = int(np.sqrt(num_patches))
    while h * w < num_patches:
        if h <= w:
            h += 1
        else:
            w += 1
    return h, w


# ---------------------------------------------------------------------------
# Main positional encoding
# ---------------------------------------------------------------------------

class PanoramaPositionalEncoding(nn.Module):
    """Panorama-aware positional encoding enforcing yaw continuity across views.

    Two encoding components are summed and added to the input:

    * **Yaw encoding** – global longitude-aware sinusoidal encoding that wraps
      continuously over ``2π``, correctly accounting for overlap between
      adjacent views.
    * **Spatial encoding** – per-view 2-D grid sinusoidal encoding.

    Args:
        embed_dim: Dimensionality of the input (and output) embeddings.
        view_encoding_type: Type of view-level encoding (``"sinusoidal"``).
        spatial_encoding_type: Type of spatial encoding (``"sinusoidal"`` or
            ``"none"``).
        enable_continuity: If ``True``, global yaw positions account for the
            overlap ratio so that adjacent views share consistent encodings
            in their overlapping columns.
        overlap_ratio: Fraction of horizontal overlap between consecutive views
            (clamped to ``[0, 0.999]``).
        temperature: Base temperature for sinusoidal frequencies.
        dropout: Dropout probability applied after adding the encoding.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        view_encoding_type: str = "sinusoidal",
        spatial_encoding_type: str = "sinusoidal",
        enable_continuity: bool = True,
        overlap_ratio: float = 0.0,
        temperature: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.view_encoding_type = view_encoding_type
        self.spatial_encoding_type = spatial_encoding_type
        self.enable_continuity = bool(enable_continuity)
        self.temperature = float(temperature)
        self.overlap_ratio = max(0.0, min(float(overlap_ratio), 0.999))
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Core sinusoidal builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sinusoidal(
        pos: torch.Tensor,
        dim: int,
        temperature: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a sinusoidal positional encoding from arbitrary position values.

        Computation is performed in float32 and cast back to *dtype* at the end
        to avoid precision issues with half-precision training.
        """
        device = pos.device
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        pos_f = pos.to(compute_dtype)
        half = dim // 2
        if half == 0:
            return torch.zeros(*pos.shape, dim, device=device, dtype=dtype)
        idx = torch.arange(half, device=device, dtype=compute_dtype)
        div = torch.exp(-math.log(temperature) * (2 * idx / max(1, dim)))
        ang = pos_f.unsqueeze(-1) * div
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < dim:
            pad = torch.zeros(*emb.shape[:-1], dim - emb.shape[-1], device=device, dtype=compute_dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.to(dtype)

    # ------------------------------------------------------------------
    # Component encodings
    # ------------------------------------------------------------------

    def _yaw_encoding(
        self,
        num_views: int,
        batch_size: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute yaw (longitude) encoding with optional continuity."""
        V, D = num_views, self.embed_dim
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)
        x = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))
        g = v_idx * s + x.unsqueeze(0)  # [V, W] global column positions

        L_total = V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)
        phi = (2.0 * math.pi) * (g / max(1e-6, float(L_total)))
        yaw_vw = self._build_sinusoidal(phi, D, self.temperature, base_dtype)
        # Broadcast: [1, V, 1, W, D] -> [B, V, H, W, D]
        yaw = yaw_vw.view(1, V, 1, W, D).expand(batch_size, V, H, W, D)
        return yaw.to(dtype)

    def _spatial_encoding(
        self,
        H: int,
        W: int,
        V: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute 2-D spatial grid encoding (row + column)."""
        if self.spatial_encoding_type != "sinusoidal":
            return torch.zeros(V, H, W, self.embed_dim, device=device, dtype=dtype)
        base_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        y = torch.arange(H, device=device, dtype=base_dtype)
        y_emb = self._build_sinusoidal(y, self.embed_dim, self.temperature, base_dtype)

        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=base_dtype).view(V, 1)
        x_local = torch.arange(W, device=device, dtype=base_dtype) / max(1.0, float(W))
        g = v_idx * s + x_local.unsqueeze(0)
        x_emb = self._build_sinusoidal(g, self.embed_dim, self.temperature, base_dtype)

        grid = y_emb.view(1, H, 1, self.embed_dim) + x_emb.view(V, 1, W, self.embed_dim)
        return grid.to(dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Add panorama-aware positional encoding to resampled features.

        Args:
            x: Tensor of shape ``[B*V, S, D]`` where *S* is the spatial token
                count and *D* must equal :attr:`embed_dim`.
            batch_size: Batch size *B*.
            num_views: Number of views *V*.

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        BV, S, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {D}, "
                f"but PanoramaPositionalEncoding was configured with {self.embed_dim}"
            )
        H, W = infer_hw(S)
        xv = x.view(batch_size, num_views, H, W, D)

        yaw_pe = self._yaw_encoding(num_views, batch_size, H, W, x.device, x.dtype)
        spat_pe = self._spatial_encoding(H, W, num_views, x.device, x.dtype)

        pe = yaw_pe + spat_pe  # broadcast [B,V,H,W,D] + [V,H,W,D]
        out = (xv + pe).view(BV, S, D)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Spherical 3D positional encoding (Fourier-based)
# ---------------------------------------------------------------------------


class PanoramaYawRoPE(nn.Module):
    """Vision-side learnable yaw RoPE for panoramic tile tokens.

    Applies RoPE-style 2-D rotation to the first ``rope_dim`` (=D/2 by default)
    channels of each token using the token's true yaw position (radians).
    The rotation frequency table is a learnable ``nn.Parameter`` so the
    backbone can shape its yaw-sensitivity scale during fine-tuning. Row
    (height) spatial information is added as a fixed sinusoidal encoding to
    the *remaining* D/2 channels (an additive signal, not rotated) so vertical
    structure is still encoded.

    Geometry-aware: each forward call needs to know which yaw the token sits
    at. The Stage 0 metadata pipeline already produces
    ``pano_meta = {"hfov_deg", "yaw_centers_deg", ...}`` per sample, so we
    expose ``set_meta(meta_list)`` for the trainer to stash that info just
    before calling the model. ``forward`` then reads ``self._meta`` and falls
    back to uniform yaw spacing if the stash is empty (eval / non-pano path).

    Args:
        embed_dim: Token feature width D.
        rope_dim: Dims that receive the yaw rotation. Must be even and
            ``<= embed_dim``. Default ``embed_dim // 2``.
        init_temperature: Initial RoPE base (10000 = LLaMA default).
        row_temperature: Sinusoidal temperature for the row PE.
        dropout: Dropout applied after combining.
        include_global: Whether view 0 is the global image (kept as-is).
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        rope_dim: Optional[int] = None,
        init_temperature: float = 10000.0,
        row_temperature: float = 10000.0,
        dropout: float = 0.0,
        include_global: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        if rope_dim is None:
            rope_dim = self.embed_dim // 2
        rope_dim = int(rope_dim)
        if rope_dim <= 0 or rope_dim > self.embed_dim:
            raise ValueError(
                f"rope_dim ({rope_dim}) must be in (0, embed_dim={self.embed_dim}]"
            )
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) must be even for RoPE pairs")
        self.rope_dim = rope_dim
        self.row_dim = self.embed_dim - rope_dim
        self.row_temperature = float(row_temperature)
        self.include_global = bool(include_global)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # Learnable RoPE frequencies — initialized log-spaced like vanilla RoPE.
        # We parameterize log_inv_freq so frequencies stay positive after exp().
        half = rope_dim // 2
        idx = torch.arange(half, dtype=torch.float32)
        init_log_inv_freq = -math.log(init_temperature) * (2.0 * idx / max(1, rope_dim))
        self.log_inv_freq = nn.Parameter(init_log_inv_freq)

        # Per-call meta stash. Populated by the trainer right before the forward
        # pass. None ⇒ fall back to uniform yaw spacing (eval / sanity).
        self._meta: Optional[List[Dict[str, object]]] = None

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def set_meta(self, meta_list: Optional[List[Dict[str, object]]]) -> None:
        """Stash per-sample pano_meta for the next forward()."""
        self._meta = meta_list

    def clear_meta(self) -> None:
        self._meta = None

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """LLaMA-style half rotation: [x1, x2] → [-x2, x1]."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _yaw_per_token(
        self,
        num_views: int,
        H: int,
        W: int,
        meta: Optional[Dict[str, object]],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute yaw (radians) for every (v, h, w) token. Shape: [V, H, W].

        Stage 0 guarantees ``meta["yaw_centers_deg"]`` has length ``n_tiles``;
        the global view (when ``include_global``) takes slot 0 with yaw 0.
        Length mismatches indicate a wiring bug — fail loud, not silent.
        """
        if meta is None or meta.get("yaw_centers_deg") is None:
            # No meta — uniform spacing fallback. Used by standalone unit tests
            # or any eval path that bypasses the Stage 0 pipeline.
            step = 2.0 * math.pi / max(1, num_views)
            yaw_centers = torch.arange(num_views, device=device, dtype=torch.float32) * step
            hfov_rad = step
        else:
            yaw_tiles = torch.as_tensor(
                list(meta["yaw_centers_deg"]), dtype=torch.float32, device=device,
            ) * (math.pi / 180.0)
            hfov_rad = float(meta["hfov_deg"]) * math.pi / 180.0
            if self.include_global:
                expected = yaw_tiles.numel() + 1
                if expected != num_views:
                    raise ValueError(
                        f"yaw_centers_deg length {yaw_tiles.numel()} +1 (global) "
                        f"!= num_views {num_views}"
                    )
                yaw_centers = torch.cat(
                    [torch.zeros(1, device=device, dtype=torch.float32), yaw_tiles]
                )
            else:
                if yaw_tiles.numel() != num_views:
                    raise ValueError(
                        f"yaw_centers_deg length {yaw_tiles.numel()} "
                        f"!= num_views {num_views} (include_global=False)"
                    )
                yaw_centers = yaw_tiles

        # Column offsets within a tile: each tile's hfov spans W columns.
        col_idx = torch.arange(W, device=device, dtype=torch.float32)
        col_offset = (col_idx - (W - 1) / 2.0) * (hfov_rad / W)
        yaw_vw = yaw_centers.view(num_views, 1) + col_offset.view(1, W)
        return yaw_vw.unsqueeze(1).expand(num_views, H, W)

    def _row_pe(self, H: int, W: int, V: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Additive sinusoidal row encoding ``[V, H, W, row_dim]``."""
        if self.row_dim == 0:
            return torch.zeros(V, H, W, 0, device=device, dtype=dtype)
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        y = torch.arange(H, device=device, dtype=compute_dtype)
        half = self.row_dim // 2
        if half == 0:
            return torch.zeros(V, H, W, self.row_dim, device=device, dtype=dtype)
        idx = torch.arange(half, device=device, dtype=compute_dtype)
        div = torch.exp(-math.log(self.row_temperature) * (2 * idx / max(1, self.row_dim)))
        ang = y.unsqueeze(-1) * div  # [H, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [H, 2*half]
        if emb.shape[-1] < self.row_dim:
            pad = torch.zeros(H, self.row_dim - emb.shape[-1], device=device, dtype=compute_dtype)
            emb = torch.cat([emb, pad], dim=-1)
        emb = emb.to(dtype)
        return emb.view(1, H, 1, self.row_dim).expand(V, H, W, self.row_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Apply yaw RoPE + row additive PE to ``x: [B*V, S, D]``."""
        BV, S, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {D}, "
                f"but PanoramaYawRoPE was configured with {self.embed_dim}"
            )
        H, W = infer_hw(S)
        xv = x.view(batch_size, num_views, H, W, D)
        compute_dtype = torch.float32 if xv.dtype in (torch.float16, torch.bfloat16) else xv.dtype

        # Build yaw[V, H, W] for each sample. With batch_size=1 (our recipe)
        # we just take meta[0]; otherwise we collapse to a default by sample.
        # We compute on float32 then cast back to xv.dtype to avoid bf16 RoPE drift.
        outs = []
        for b in range(batch_size):
            meta_b = None
            if self._meta is not None and b < len(self._meta):
                meta_b = self._meta[b]
            yaw_vhw = self._yaw_per_token(num_views, H, W, meta_b, xv.device)  # [V, H, W]

            inv_freq = torch.exp(self.log_inv_freq.to(compute_dtype))  # [rope_dim/2]
            # angle: [V, H, W, rope_dim/2] = yaw * inv_freq
            angle = yaw_vhw.unsqueeze(-1) * inv_freq  # broadcast
            cos = torch.cos(angle)  # [V, H, W, rope_dim/2]
            sin = torch.sin(angle)
            # Duplicate to full rope_dim: standard LLaMA-style expects cos/sin
            # of shape rope_dim with the second half being the same values.
            cos_full = torch.cat([cos, cos], dim=-1)  # [V, H, W, rope_dim]
            sin_full = torch.cat([sin, sin], dim=-1)

            xb = xv[b]  # [V, H, W, D]
            x_rope = xb[..., :self.rope_dim].to(compute_dtype)
            x_keep = xb[..., self.rope_dim:].to(compute_dtype)

            rotated = x_rope * cos_full + self._rotate_half(x_rope) * sin_full

            if self.row_dim > 0:
                row = self._row_pe(H, W, num_views, xv.device, compute_dtype)  # [V, H, W, row_dim]
                kept = x_keep + row
            else:
                kept = x_keep

            yb = torch.cat([rotated, kept], dim=-1).to(xv.dtype)  # [V, H, W, D]
            outs.append(yb)

        out = torch.stack(outs, dim=0).view(BV, S, D)
        return self.dropout(out)


def attach_yaw_rope_hook(
    *,
    model: nn.Module,
    target_module: nn.Module,
    rope_dim: Optional[int] = None,
    init_temperature: float = 10000.0,
    include_global: bool = True,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple["PanoramaYawRoPE", object]:
    """Build a ``PanoramaYawRoPE``, attach it to ``model``, register a
    forward hook on the vision projector.

    The hook reads geometry from the module's own ``_meta`` stash, so the
    train and eval paths share one source of truth — the caller sets
    ``yaw_rope.set_meta([...])`` immediately before each forward pass.

    Args:
        model: The full VLM model (PeftModel or base). The yaw RoPE is
            attached as ``model.panoadapt_yaw_rope`` so device moves /
            HF Trainer optimizer / state-dict iteration pick it up.
        target_module: The vision projector whose forward output is rotated
            (``multi_modal_projector`` / ``merger``).
        rope_dim: Channels to rotate (default = embed_dim // 2).
        init_temperature: RoPE freq base at init.
        include_global: Whether view 0 is a global image (yaw=0 anchor).
        state_dict: Optional pre-trained ``yaw_rope.state_dict()`` to load
            before attaching — used by the eval path.

    Returns:
        (yaw_rope, hook_handle).
    """
    # Prefer last Linear's out_features (works for InternVL / Qwen mergers).
    # Fall back to the LM's hidden_size from model.config when the projector
    # has no Linear at all (e.g. Gemma3MultiModalProjector is just RMSNorm +
    # AvgPool, with the hidden_size→LM mapping implicit elsewhere).
    embed_dim: Optional[int] = None
    for sub in target_module.modules():
        if isinstance(sub, nn.Linear):
            embed_dim = sub.out_features
    if embed_dim is None:
        cfg = getattr(model, "config", None) or getattr(getattr(model, "base_model", None), "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        for cand in (text_cfg, cfg):
            if cand is not None and hasattr(cand, "hidden_size"):
                embed_dim = int(cand.hidden_size)
                break
    if embed_dim is None:
        raise RuntimeError(
            f"attach_yaw_rope_hook: could not infer projector out width from "
            f"{type(target_module).__name__} and no LM hidden_size on model.config"
        )

    yaw_rope = PanoramaYawRoPE(
        embed_dim=embed_dim,
        rope_dim=rope_dim,
        init_temperature=init_temperature,
        include_global=include_global,
    )
    if state_dict is not None:
        yaw_rope.load_state_dict(state_dict)

    # Move to the projector's device, BUT keep params fp32 even when the model
    # is fp16. torch.amp.GradScaler can only unscale fp32 grads; casting yaw_rope
    # to fp16 caused "Attempting to unscale FP16 gradients" on Qwen2.5-VL fp16
    # baselines. PanoramaYawRoPE.forward already upcasts inputs to float32 for
    # the rotation math and downcasts back, so an fp32 param is correct here.
    sample_param = next(target_module.parameters(), None)
    if sample_param is not None:
        yaw_rope.to(device=sample_param.device)
    model.add_module("panoadapt_yaw_rope", yaw_rope)

    def _hook_fn(module, inputs, output):  # noqa: ARG001
        meta_list = yaw_rope._meta
        if not meta_list:
            # Eval/probe path without a meta stash — module is dormant.
            return output
        if len(meta_list) != 1:
            raise RuntimeError(
                f"yaw_rope hook only supports batch_size=1 (got {len(meta_list)})"
            )
        meta = meta_list[0]
        if meta is None:
            return output

        num_views = int(meta["num_views"])
        if output.ndim == 3:
            reshaped = output
        elif output.ndim == 2:
            total, dim = output.shape
            if total % num_views != 0:
                raise RuntimeError(
                    f"yaw_rope hook: flat projector output {total} not divisible "
                    f"by num_views {num_views}"
                )
            reshaped = output.view(num_views, total // num_views, dim)
        else:
            raise RuntimeError(
                f"yaw_rope hook: unexpected projector output ndim={output.ndim}"
            )

        rotated = yaw_rope(reshaped, batch_size=1, num_views=num_views)
        return rotated.view(output.shape)

    handle = target_module.register_forward_hook(_hook_fn)
    return yaw_rope, handle


class PanoramaPositionalEncoding2(nn.Module):
    """Spherical 3-D Fourier positional encoding for panoramic tokens.

    Unlike :class:`PanoramaPositionalEncoding` which uses additive sinusoidal
    encodings, this variant maps each token to a 3-D point on the unit sphere
    via latitude/longitude, applies multi-band Fourier features, and projects
    back to the embedding dimension via a learned linear layer.

    This encoding naturally captures the spherical geometry of equirectangular
    panoramas and smoothly wraps around the 360 degree azimuth.

    Args:
        embed_dim: Dimensionality of the input (and output) embeddings.
        view_encoding_type: Unused (kept for API compatibility).
        spatial_encoding_type: Unused (kept for API compatibility).
        enable_continuity: If ``True``, yaw positions account for overlap.
        overlap_ratio: Fraction of horizontal overlap between consecutive views.
        temperature: Unused (kept for API compatibility).
        dropout: Dropout probability applied after adding the encoding.
        num_fourier_bands: Number of Fourier frequency bands.
        include_input_xyz: Whether to include raw (x, y, z) coordinates in
            the feature vector before projection.
        pe_scale: Frequency scaling factor for the Fourier features.
        phi_offset_rad: Offset (in radians) applied to the global longitude.
        lat_center_rad: Centre latitude (in radians) for the coverage window.
        lat_coverage_ratio: Fraction of the full latitude range to cover.
        project_bias: Whether the output linear projection includes a bias.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        view_encoding_type: str = "sinusoidal",
        spatial_encoding_type: str = "sinusoidal",
        enable_continuity: bool = True,
        overlap_ratio: float = 0.0,
        temperature: float = 10000.0,
        dropout: float = 0.0,
        num_fourier_bands: int = 8,
        include_input_xyz: bool = True,
        pe_scale: float = math.pi,
        phi_offset_rad: float = 0.0,
        lat_center_rad: float = 0.0,
        lat_coverage_ratio: float = 1.0,
        project_bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.enable_continuity = bool(enable_continuity)
        self.overlap_ratio = float(max(0.0, min(overlap_ratio, 0.999)))
        self.dropout = (
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        )
        self.num_fourier_bands = int(num_fourier_bands)
        self.include_input_xyz = bool(include_input_xyz)
        self.pe_scale = float(pe_scale)
        self.phi_offset_rad = float(phi_offset_rad)
        self.lat_center_rad = float(lat_center_rad)
        self.lat_coverage_ratio = float(lat_coverage_ratio)
        raw_dim = self._raw_feature_dim()
        self._proj = nn.Linear(raw_dim, embed_dim, bias=project_bias)

    def _raw_feature_dim(self) -> int:
        """Compute the intermediate feature dimension before projection."""
        xyz_dim = 3 if self.include_input_xyz else 0
        return xyz_dim + (3 * (2 * self.num_fourier_bands))

    @staticmethod
    def _infer_hw(tokens: int) -> Tuple[int, int]:
        """Delegate to module-level :func:`infer_hw`."""
        return infer_hw(tokens)

    # ------------------------------------------------------------------
    # Coordinate computation
    # ------------------------------------------------------------------

    def _global_longitude(
        self, V: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Compute global longitude (phi) for each view/column ``[V, W]``."""
        s = 1.0 - self.overlap_ratio if self.enable_continuity else 1.0
        v_idx = torch.arange(V, device=device, dtype=torch.float32).view(V, 1)
        x = torch.arange(W, device=device, dtype=torch.float32) / max(1.0, float(W))
        g = v_idx * s + x.unsqueeze(0)
        L_total = max(
            float(V - (V - 1) * (self.overlap_ratio if self.enable_continuity else 0.0)),
            1e-6,
        )
        phi = (2.0 * math.pi) * (g / L_total) + self.phi_offset_rad
        return phi

    def _latitude_from_rows(self, H: int, device: torch.device) -> torch.Tensor:
        """Compute latitude (theta) for each row ``[H]``."""
        y = torch.arange(H, device=device, dtype=torch.float32)
        u = (y + 0.5) / max(1.0, float(H))
        theta_raw = (u - 0.5) * math.pi
        theta = self.lat_center_rad + (self.lat_coverage_ratio * theta_raw)
        return theta

    @staticmethod
    def _spherical_xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Convert latitude/longitude to unit-sphere Cartesian coordinates."""
        cos_theta = torch.cos(theta)
        x = cos_theta * torch.cos(phi)
        y = torch.sin(theta)
        z = cos_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=-1)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply multi-band Fourier encoding to 3-D coordinates.

        Args:
            coords: ``[..., 3]`` Cartesian coordinates.

        Returns:
            ``[..., 3 * 2 * num_fourier_bands]`` sin/cos features.
        """
        bands = self.num_fourier_bands
        if bands <= 0:
            raise ValueError("num_fourier_bands must be positive")
        freq = (
            2.0 ** torch.arange(bands, device=coords.device, dtype=coords.dtype)
        ) * self.pe_scale
        ang = coords.unsqueeze(-1) * freq  # [..., 3, bands]
        out = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [..., 3, 2*bands]
        return out.view(*coords.shape[:-1], 3 * (2 * bands))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Add spherical positional encoding to resampled features.

        Args:
            x: ``[B*V, S, D]`` resampled features.
            batch_size: Batch size *B*.
            num_views: Number of views *V*.

        Returns:
            Tensor of the same shape with spherical encoding added.
        """
        BV, S, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {D}, "
                f"but PanoramaPositionalEncoding2 was configured with {self.embed_dim}"
            )
        H, W = self._infer_hw(S)
        device = x.device
        xv = x.view(batch_size, num_views, H, W, D)

        # Compute spherical coordinates
        phi_vw = self._global_longitude(num_views, W, device)  # [V, W]
        theta_h = self._latitude_from_rows(H, device)  # [H]

        phi = phi_vw.view(1, num_views, 1, W).expand(batch_size, num_views, H, W)
        theta = theta_h.view(1, 1, H, 1).expand(batch_size, num_views, H, W)

        xyz = self._spherical_xyz(theta, phi)  # [B, V, H, W, 3]

        feats = []
        if self.include_input_xyz:
            feats.append(xyz)
        feats.append(self._fourier_encode(xyz))

        pe_raw = torch.cat(feats, dim=-1)  # [B, V, H, W, raw_dim]
        pe = self._proj(pe_raw)  # [B, V, H, W, D]

        out = xv + pe
        out = out.view(BV, S, D)
        return self.dropout(out)
