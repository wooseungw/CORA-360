"""Loss functions for CORA training.

Ports:
  - legacy/CORA/cora/training/losses.py          → VICRegLoss (overlap-based)
  - legacy/src_panovlm/losses/vicreg_overlap.py   → compute_vicreg_overlap_loss
  - legacy/src_panovlm/losses/vicreg_projector.py → VICRegProjector
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: infer spatial grid from patch count (no external dep)
# ---------------------------------------------------------------------------

def _infer_hw(num_patches: int) -> Tuple[int, int]:
    """Infer (H, W) grid from total patch count; assumes ~square layout."""
    h = w = int(math.isqrt(num_patches))
    while h * w < num_patches:
        if h <= w:
            h += 1
        else:
            w += 1
    return h, w


# ---------------------------------------------------------------------------
# VICReg Projector  (ported from legacy/src_panovlm/losses/vicreg_projector.py)
# ---------------------------------------------------------------------------

class VICRegProjector(nn.Module):
    """Token-wise MLP projector for VICReg features.

    Accepts ``[B·V, S, D_in]`` and returns ``[B·V, S, D_out]``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        hd = hidden_dim or max(in_dim, out_dim)
        layers: list[nn.Module] = []
        d_prev = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_prev, hd))
            layers.append(nn.LayerNorm(hd) if use_ln else nn.Identity())
            layers.append(nn.GELU())
            d_prev = hd
        layers.append(nn.Linear(d_prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[BV, S, D_in]`` → ``[BV, S, D_out]``."""
        BV, S, D = x.shape
        out = self.mlp(x.reshape(-1, D))
        return out.view(BV, S, -1)


# ---------------------------------------------------------------------------
# VICReg Loss  (ported from legacy CORA losses + src_panovlm vicreg_overlap)
# ---------------------------------------------------------------------------

class GlobalLocalLoss(nn.Module):
    """Global-local consistency loss between global ERP view and E2P tiles.

    Encourages each tile's mean representation to align with the global view's
    mean representation, teaching tiles their position within the full panorama
    (analogous to DINO's global-crop / local-crop consistency).
    """

    def __init__(self, loss_type: str = "cosine", temperature: float = 0.1) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        global_features: torch.Tensor,
        tile_features: torch.Tensor,
        batch_size: int,
        num_tiles: int,
    ) -> torch.Tensor:
        """Compute global-local consistency loss.

        Args:
            global_features: ``[B, S, D]`` — resampled features of global ERP view.
            tile_features: ``[B*T, S, D]`` — resampled features of E2P tiles.
            batch_size: B.
            num_tiles: T (num_views - 1, excluding global).
        """
        g_pooled = global_features.mean(dim=1)  # [B, D]
        t_pooled = tile_features.mean(dim=1)  # [B*T, D]
        t_per_image = t_pooled.view(batch_size, num_tiles, -1).mean(dim=1)  # [B, D]

        if self.loss_type == "mse":
            return F.mse_loss(g_pooled, t_per_image)

        g_norm = F.normalize(g_pooled, dim=-1)
        t_norm = F.normalize(t_per_image, dim=-1)
        return (1.0 - (g_norm * t_norm).sum(dim=-1)).mean()


class VICRegLoss(nn.Module):
    """VICReg overlap loss for panoramic multi-view features.

    Operates on features shaped ``[B·V, S, D]`` where adjacent views share an
    overlap region determined by ``overlap_ratio``.

    Supports two modes via ``vicreg_mode``:
      - **pairwise** (default, panorama-oriented): variance / covariance
        computed per (current, next) pair.
      - **batchwise** (original VICReg): variance / covariance computed across
        the entire mini-batch.

    Also supports a *flat* mode when ``batch_size`` and ``num_views`` are both
    set to 0: treats the two input tensors as pre-extracted embedding matrices
    (useful for unit tests with ``torch.randn``).
    """

    def __init__(
        self,
        similarity_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        gamma: float = 1.0,
        overlap_ratio: float = 0.5,
        vicreg_mode: str = "pairwise",
        pair_chunk: int | None = 8,
        eps: float = 1e-4,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.gamma = gamma
        self.overlap_ratio = overlap_ratio
        self.vicreg_mode = vicreg_mode
        self.pair_chunk = pair_chunk
        self.eps = eps
        self._debug = debug

    # -- public API -----------------------------------------------------------

    def forward(
        self,
        z_a: torch.Tensor,
        z_b_or_batch_size: torch.Tensor | int,
        num_views: int = 0,
    ) -> torch.Tensor:
        """Compute VICReg loss.

        Two calling conventions:

        1. **Overlap mode** (panoramic multi-view):
           ``forward(vision_features, batch_size, num_views)``
           where ``vision_features`` has shape ``[B·V, S, D]``.

        2. **Flat / embedding mode** (unit-test friendly):
           ``forward(z_a, z_b)``
           where both tensors have shape ``[N, D]`` (2-D).
        """
        # Dispatch: flat mode when z_b is a tensor
        if isinstance(z_b_or_batch_size, torch.Tensor):
            return self._forward_flat(z_a, z_b_or_batch_size)

        # Overlap mode
        batch_size: int = int(z_b_or_batch_size)
        return self._forward_overlap(z_a, batch_size, num_views)

    # -- flat / embedding mode ------------------------------------------------

    def _forward_flat(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """VICReg on two embedding matrices ``[N, D]``."""
        assert z_a.ndim == 2 and z_b.ndim == 2, (
            f"Flat mode expects 2-D tensors, got {z_a.shape} and {z_b.shape}"
        )
        orig_dtype = z_a.dtype
        z_a = z_a.float()
        z_b = z_b.float()
        N, D = z_a.shape

        inv_loss = F.mse_loss(z_a, z_b)

        std_a = torch.sqrt(z_a.var(dim=0, unbiased=False) + self.eps)
        std_b = torch.sqrt(z_b.var(dim=0, unbiased=False) + self.eps)
        var_loss = 0.5 * (
            F.relu(self.gamma - std_a).mean() + F.relu(self.gamma - std_b).mean()
        )

        def _cov_off_diag(z: torch.Tensor) -> torch.Tensor:
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.T @ zc) / max(N - 1, 1)
            off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            return off / D

        cov_loss = 0.5 * (_cov_off_diag(z_a) + _cov_off_diag(z_b))

        total = (
            self.similarity_weight * inv_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )

        if self._debug:
            logger.info(
                "VICReg(flat): inv=%.4f var=%.4f cov=%.4f total=%.4f",
                inv_loss.item(), var_loss.item(), cov_loss.item(), total.item(),
            )
        return total.to(orig_dtype)

    # -- overlap / panoramic mode --------------------------------------------

    def _forward_overlap(
        self,
        vision_features: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Overlap-based VICReg for panoramic multi-view features.

        ``vision_features``: ``[B·V, S, D]``.
        All var/cov/sqrt ops are computed in FP32 to prevent FP16 overflow.
        """
        if num_views <= 1:
            return vision_features.sum() * 0.0

        orig_dtype = vision_features.dtype
        vision_features = vision_features.float()

        BV, S, D = vision_features.shape
        grid_h, grid_w = _infer_hw(S)

        features_grid = vision_features.view(batch_size, num_views, grid_h, grid_w, D)

        k = max(1, int(grid_w * self.overlap_ratio))
        curr_overlap = features_grid[:, :, :, -k:, :]
        next_overlap = torch.roll(features_grid, -1, dims=1)[:, :, :, :k, :]

        P = batch_size * num_views
        L = grid_h * k
        curr = curr_overlap.contiguous().view(P, L, D)
        nxt = next_overlap.contiguous().view(P, L, D)

        inv_pair = F.mse_loss(curr, nxt, reduction="none").mean(dim=(1, 2))

        if self.vicreg_mode == "batchwise":
            combined = torch.cat([curr, nxt], dim=0).reshape(-1, D)
            std_all = torch.sqrt(combined.var(dim=0, unbiased=False) + self.eps)
            var_pair = F.relu(self.gamma - std_all).mean().expand(P)
        else:
            std_c = torch.sqrt(curr.var(dim=1, unbiased=False) + self.eps)
            std_n = torch.sqrt(nxt.var(dim=1, unbiased=False) + self.eps)
            var_pair = 0.5 * (
                F.relu(self.gamma - std_c).mean(dim=1)
                + F.relu(self.gamma - std_n).mean(dim=1)
            )

        if self.vicreg_mode == "batchwise":
            combined = torch.cat([curr, nxt], dim=0).reshape(-1, D)
            cc = combined - combined.mean(dim=0, keepdim=True)
            cov = (cc.T @ cc) / max(combined.size(0) - 1, 1)
            cov_clone = cov.clone()
            cov_clone.diagonal().zero_()
            cov_pair = ((cov_clone ** 2).sum() / D).expand(P)
        else:
            curr_c = curr - curr.mean(dim=1, keepdim=True)
            nxt_c = nxt - nxt.mean(dim=1, keepdim=True)
            denom = max(L - 1, 1)

            def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
                P_local = xc.size(0)
                chunk = self.pair_chunk
                if chunk is None or chunk >= P_local:
                    C = torch.bmm(xc.transpose(1, 2), xc) / denom
                    C2 = (C ** 2).sum(dim=(1, 2))
                    diag2 = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                    return (C2 - diag2) / D
                parts: list[torch.Tensor] = []
                for s in range(0, P_local, chunk):
                    e = min(s + chunk, P_local)
                    C = torch.bmm(xc[s:e].transpose(1, 2), xc[s:e]) / denom
                    C2 = (C ** 2).sum(dim=(1, 2))
                    diag2 = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                    parts.append((C2 - diag2) / D)
                    del C
                return torch.cat(parts, dim=0)

            cov_pair = 0.5 * (
                _cov_offdiag_sq_mean(curr_c) + _cov_offdiag_sq_mean(nxt_c)
            )

        per_pair = (
            self.similarity_weight * inv_pair
            + self.variance_weight * var_pair
            + self.covariance_weight * cov_pair
        )
        total = per_pair.mean()

        if self._debug:
            logger.info(
                "VICReg(overlap): inv=%.4f var=%.4f cov=%.4f total=%.4f",
                inv_pair.mean().item(),
                var_pair.mean().item(),
                cov_pair.mean().item(),
                total.item(),
            )

        return torch.clamp(total, max=1e6).to(orig_dtype)


# ---------------------------------------------------------------------------
# Panoramic Contrastive Loss  (replaces VICReg for Stage 1)
# ---------------------------------------------------------------------------

class PanoContrastiveLoss(nn.Module):
    """Dense patch-level contrastive loss for panoramic tile overlap alignment.

    Replaces VICReg to avoid the invariance-variance equilibrium trap.
    Uses **symmetric InfoNCE** with two components:

    1. **Overlap alignment** — for each adjacent tile pair ``(v, v+1 mod V)``,
       the right *k* columns of tile *v* (from projection view 1) are matched
       against the left *k* columns of tile *v+1* (from projection view 2).
       Positives: same ``(batch, row, col)`` — same physical location.
       Negatives: all other ``(batch, row, col)`` combinations.

    2. **Within-tile discrimination** — every patch in a tile is treated as
       a positive with its dropout-augmented counterpart from the second
       projection view.  All other patches in the **same tile** are negatives.
       This gives gradient signal to ALL spatial positions, including the
       non-overlap columns that receive zero gradient from the overlap term.

    Why it avoids VICReg's equilibrium
    -----------------------------------
    VICReg has two directly opposing forces on overlap features:
    invariance (make identical) vs variance (make diverse).  They cancel
    at a fixed point.  Here there is no explicit variance term; repulsion
    comes from contrastive negatives, which is *unstable* under collapse
    (identical embeddings → positives indistinguishable from negatives →
    loss stays high → gradients push apart).

    Parameters
    ----------
    overlap_ratio : float
        Fraction of spatial width that overlaps between adjacent tiles.
        Should match the **physical** overlap (``image_processing.overlap_ratio``),
        not the reduced ``vicreg_overlap_ratio`` (which was only needed for VICReg).
    tau_overlap : float
        Temperature for overlap InfoNCE.  Lower → sharper similarity.
    tau_tile : float
        Temperature for within-tile InfoNCE.  Keep higher than ``tau_overlap``
        to tolerate repetitive textures (reduces false-negative issues).
    tile_loss_weight : float
        Relative weight ``λ_tile`` of within-tile loss.
    """

    def __init__(
        self,
        overlap_ratio: float = 0.5,
        tau_overlap: float = 0.07,
        tau_tile: float = 0.2,
        tile_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.overlap_ratio = overlap_ratio
        self.tau_overlap = tau_overlap
        self.tau_tile = tau_tile
        self.tile_loss_weight = tile_loss_weight

    # -- public API -----------------------------------------------------------

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute total contrastive loss.

        Args:
            z1: ``[B*V, S, D]`` — first L2-normalised projection.
            z2: ``[B*V, S, D]`` — second L2-normalised projection
                (different dropout mask).
            batch_size: *B*.
            num_views: *V* (tile count, excluding global).

        Returns:
            Dict with ``loss`` (gradient-carrying), ``overlap_loss``,
            ``tile_loss`` (both detached, for logging).
        """
        if num_views <= 1:
            zero = z1.sum() * 0.0
            return {"loss": zero, "overlap_loss": zero, "tile_loss": zero}

        orig_dtype = z1.dtype
        z1 = z1.float()
        z2 = z2.float()

        L_overlap = self._overlap_infonce(z1, z2, batch_size, num_views)

        if self.tile_loss_weight > 0:
            L_tile = self._tile_infonce(z1, z2, batch_size, num_views)
        else:
            L_tile = z1.new_zeros(())

        total = L_overlap + self.tile_loss_weight * L_tile

        return {
            "loss": total.to(orig_dtype),
            "overlap_loss": L_overlap.detach(),
            "tile_loss": L_tile.detach(),
        }

    # -- overlap InfoNCE ------------------------------------------------------

    def _overlap_infonce(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Symmetric InfoNCE between overlap strips of adjacent tiles.

        Uses **circular** adjacency (tile V-1 → tile 0) for 360° continuity,
        matching the existing VICReg pairing convention (``torch.roll``).
        """
        S, D = z1.shape[1], z1.shape[2]
        H = W = int(math.isqrt(S))

        g1 = z1.view(B, V, H, W, D)
        g2 = z2.view(B, V, H, W, D)

        k = max(1, int(W * self.overlap_ratio))
        N = B * H * k  # number of positive pairs per tile-pair

        # Right k columns of each tile (from z1)
        curr_right = g1[:, :, :, -k:, :]                          # [B, V, H, k, D]
        # Left k columns of next tile with circular wrap (from z2)
        next_left = torch.roll(g2, -1, dims=1)[:, :, :, :k, :]   # [B, V, H, k, D]

        # Reshape to [V, N, D] for batched matmul
        Q = curr_right.permute(1, 0, 2, 3, 4).reshape(V, N, D)
        K = next_left.permute(1, 0, 2, 3, 4).reshape(V, N, D)

        # Batched similarity [V, N, N]
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.tau_overlap

        labels = torch.arange(N, device=z1.device).unsqueeze(0).expand(V, -1)

        # Symmetric InfoNCE: q→k and k→q
        loss_q2k = F.cross_entropy(
            logits.reshape(V * N, N), labels.reshape(V * N),
        )
        loss_k2q = F.cross_entropy(
            logits.transpose(1, 2).contiguous().reshape(V * N, N),
            labels.reshape(V * N),
        )
        return 0.5 * (loss_q2k + loss_k2q)

    # -- within-tile InfoNCE --------------------------------------------------

    def _tile_infonce(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Symmetric InfoNCE between dropout views of the same tile.

        For each tile ``(b, v)``, every patch is a positive with its
        dropout-augmented counterpart.  All other patches in the same tile
        are negatives.  Provides gradient to ALL spatial positions.
        """
        BV, S, D = z1.shape

        # Batched similarity [BV, S, S]
        logits = torch.bmm(z1, z2.transpose(1, 2)) / self.tau_tile

        labels = torch.arange(S, device=z1.device).unsqueeze(0).expand(BV, -1)

        loss_q2k = F.cross_entropy(
            logits.reshape(BV * S, S), labels.reshape(BV * S),
        )
        loss_k2q = F.cross_entropy(
            logits.transpose(1, 2).contiguous().reshape(BV * S, S),
            labels.reshape(BV * S),
        )
        return 0.5 * (loss_q2k + loss_k2q)


# ---------------------------------------------------------------------------
# DenseCL Loss  (dense patch-level contrastive on overlap zones)
# ---------------------------------------------------------------------------

class DenseCLLoss(nn.Module):
    """Dense Contrastive Learning loss for panoramic tile overlap alignment.

    Adapted from DenseCL (CVPR 2021) for panoramic overlap correspondence.
    Uses **known spatial correspondence** in overlap zones — no feature-matching
    step is needed because the equirectangular grid defines exact pixel pairs.

    Key differences from :class:`PanoContrastiveLoss`:

    - **Single projection**: operates on ``vicreg_features`` directly.
      No dropout-augmented second view; the two "views" are the physically
      overlapping strips of adjacent tiles (different input → same location).
    - **Overlap only**: no within-tile discrimination term.
      Gradients flow only to overlap columns; non-overlap columns are
      regularised implicitly through the shared backbone / resampler.

    Parameters
    ----------
    overlap_ratio : float
        Fraction of spatial width that overlaps between adjacent tiles.
        Should match the **VICReg** overlap (``vicreg_overlap_ratio`` if set,
        else ``image_processing.overlap_ratio``).
    temperature : float
        Softmax temperature for InfoNCE.  Lower → sharper similarity peak.
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
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE on overlap regions.

        Args:
            features: ``[B*V, S, D]`` — projected tile features (single view).
            batch_size: *B*.
            num_views: *V* (tile count, excluding global).

        Returns:
            Scalar loss (gradient-carrying).
        """
        if num_views <= 1:
            return features.sum() * 0.0

        orig_dtype = features.dtype
        features = features.float()

        BV, S, D = features.shape
        H = W = int(math.isqrt(S))

        grid = features.view(batch_size, num_views, H, W, D)

        k = max(1, int(W * self.overlap_ratio))
        N = batch_size * H * k  # positives per tile-pair

        # Right k columns of each tile  →  left k columns of next tile (circular)
        curr_right = grid[:, :, :, -k:, :]                         # [B, V, H, k, D]
        next_left = torch.roll(grid, -1, dims=1)[:, :, :, :k, :]  # [B, V, H, k, D]

        # Reshape to [V, N, D] and L2-normalise
        Q = F.normalize(
            curr_right.permute(1, 0, 2, 3, 4).reshape(num_views, N, D), dim=-1,
        )
        K = F.normalize(
            next_left.permute(1, 0, 2, 3, 4).reshape(num_views, N, D), dim=-1,
        )

        # Batched similarity  [V, N, N]
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.temperature

        labels = torch.arange(N, device=features.device).unsqueeze(0).expand(
            num_views, -1,
        )

        # Symmetric InfoNCE: q→k  +  k→q
        VN = num_views * N
        loss_q2k = F.cross_entropy(logits.reshape(VN, N), labels.reshape(VN))
        loss_k2q = F.cross_entropy(
            logits.transpose(1, 2).contiguous().reshape(VN, N),
            labels.reshape(VN),
        )

        return (0.5 * (loss_q2k + loss_k2q)).to(orig_dtype)


# ---------------------------------------------------------------------------
# Functional API  (ported from legacy/src_panovlm/losses/vicreg_overlap.py)
# ---------------------------------------------------------------------------

def compute_vicreg_overlap_loss(
    vision_output: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    overlap_ratio: float = 0.5,
    pair_chunk: int | None = 8,
    vicreg_loss_module: VICRegLoss | nn.Module,
    has_cls_token: bool,
    vicreg_mode: str = "pairwise",
) -> torch.Tensor:
    """Compute VICReg loss on overlapping regions between adjacent panoramic views.

    Standalone functional entry-point that reads weight hyper-parameters from
    *vicreg_loss_module* and handles optional CLS-token stripping.

    Parameters
    ----------
    vision_output : Tensor [BV, S, D]
        Raw vision encoder / projector output (may include CLS token).
    batch_size, num_views : int
    overlap_ratio : float
        Fraction of spatial width that overlaps between adjacent views.
    pair_chunk : int | None
        Chunk size for covariance computation (memory optimisation).
    vicreg_loss_module : VICRegLoss | nn.Module
        Source of weight hyper-parameters (similarity_weight, etc.).
    has_cls_token : bool
        If True, strips the first token before reshaping to spatial grid.
    vicreg_mode : str
        ``"pairwise"`` or ``"batchwise"``.
    """
    if num_views <= 1:
        return torch.zeros((), device=vision_output.device)

    patch_features = vision_output[:, 1:] if has_cls_token else vision_output
    num_patches = patch_features.size(1)
    grid_h, grid_w = _infer_hw(num_patches)
    D = patch_features.size(-1)
    patch_features = patch_features.view(batch_size, num_views, grid_h, grid_w, D)

    k = max(1, int(grid_w * overlap_ratio))
    curr = patch_features[:, :, :, -k:, :]
    nxt = torch.roll(patch_features, shifts=-1, dims=1)[:, :, :, :k, :]

    P = batch_size * num_views
    L = grid_h * k
    curr = curr.contiguous().view(P, L, D)
    nxt = nxt.contiguous().view(P, L, D)

    inv_pair = F.mse_loss(curr, nxt, reduction="none").mean(dim=(1, 2))

    eps = 1e-4
    gamma = getattr(vicreg_loss_module, "gamma", 1.0)

    if vicreg_mode == "batchwise":
        combined = torch.cat([curr, nxt], dim=0).reshape(-1, D)
        std_combined = torch.sqrt(combined.var(dim=0, unbiased=False) + eps)
        var_pair = F.relu(gamma - std_combined).mean().expand(P)
    else:
        std_x = torch.sqrt(curr.var(dim=1, unbiased=False) + eps)
        std_y = torch.sqrt(nxt.var(dim=1, unbiased=False) + eps)
        var_pair = 0.5 * (
            F.relu(gamma - std_x).mean(dim=1) + F.relu(gamma - std_y).mean(dim=1)
        )

    if vicreg_mode == "batchwise":
        combined = torch.cat([curr, nxt], dim=0).reshape(-1, D)
        combined_c = combined - combined.mean(dim=0, keepdim=True)
        denom_batch = max(combined.size(0) - 1, 1)
        C = (combined_c.T @ combined_c) / denom_batch
        C_offdiag = C.clone()
        C_offdiag.diagonal().zero_()
        cov_pair = ((C_offdiag ** 2).sum() / D).expand(P)
    else:
        curr_c = curr - curr.mean(dim=1, keepdim=True)
        nxt_c = nxt - nxt.mean(dim=1, keepdim=True)
        denom = max(L - 1, 1)

        def _cov_offdiag_sq_mean(xc: torch.Tensor) -> torch.Tensor:
            P_local = xc.size(0)
            if pair_chunk is None or pair_chunk >= P_local:
                C = torch.bmm(xc.transpose(1, 2), xc) / denom
                C2_sum = (C ** 2).sum(dim=(1, 2))
                diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                return (C2_sum - diag_sq) / D
            parts: list[torch.Tensor] = []
            for s in range(0, P_local, pair_chunk):
                e = min(s + pair_chunk, P_local)
                C = torch.bmm(xc[s:e].transpose(1, 2), xc[s:e]) / denom
                C2_sum = (C ** 2).sum(dim=(1, 2))
                diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
                parts.append((C2_sum - diag_sq) / D)
                del C
            return torch.cat(parts, dim=0)

        cov_pair = 0.5 * (_cov_offdiag_sq_mean(curr_c) + _cov_offdiag_sq_mean(nxt_c))

    w_inv = getattr(vicreg_loss_module, "similarity_weight", 25.0)
    w_var = getattr(vicreg_loss_module, "variance_weight", 25.0)
    w_cov = getattr(vicreg_loss_module, "covariance_weight", 1.0)

    per_pair = w_inv * inv_pair + w_var * var_pair + w_cov * cov_pair
    total = per_pair.mean()

    if getattr(vicreg_loss_module, "_debug", False):
        logger.info(
            "[VICReg func] inv=%.4f var=%.4f cov=%.4f total=%.2f",
            inv_pair.mean().item(), var_pair.mean().item(),
            cov_pair.mean().item(), total.item(),
        )

    return torch.clamp(total, max=1e6)
