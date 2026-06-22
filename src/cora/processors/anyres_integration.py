"""
AnyRes VICReg integration: tile pairing strategies for spatial consistency loss.

Computes VICReg loss between overlapping tiles in a 2D yaw×pitch grid,
respecting the spherical geometry of panoramic images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TileRelationship:
    """Spatial relationship between two tiles."""

    tile_idx1: int
    tile_idx2: int
    overlap_ratio: float
    yaw_distance: float
    pitch_distance: float
    relationship_type: str  # 'horizontal', 'vertical', 'diagonal', 'none'


class AnyResVICRegPairing:
    """Build VICReg training pairs from AnyRes tile grids."""

    def __init__(
        self,
        hfov_deg: float = 90.0,
        vfov_deg: float = 90.0,
        overlap_ratio: float = 0.5,
        pairing_strategy: str = "adjacent",
        distance_threshold: float = 120.0,
    ) -> None:
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.overlap_ratio = overlap_ratio
        self.pairing_strategy = pairing_strategy
        self.distance_threshold = distance_threshold

    # ── Geometry helpers ────────────────────────────────────────────

    @staticmethod
    def angular_distance(yaw1: float, pitch1: float, yaw2: float, pitch2: float) -> float:
        y1, p1 = np.radians(yaw1), np.radians(pitch1)
        y2, p2 = np.radians(yaw2), np.radians(pitch2)
        dlat, dlon = p2 - p1, y2 - y1
        a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
        return float(np.degrees(2 * np.arcsin(np.sqrt(a))))

    @staticmethod
    def compute_overlap_area(
        yaw1: float, pitch1: float, hfov1: float, vfov1: float,
        yaw2: float, pitch2: float, hfov2: float, vfov2: float,
    ) -> float:
        yaw_diff = abs(yaw1 - yaw2)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        h_overlap = max(0, (hfov1 + hfov2) / 2 - yaw_diff) / min(hfov1, hfov2)
        pitch_diff = abs(pitch1 - pitch2)
        v_overlap = max(0, (vfov1 + vfov2) / 2 - pitch_diff) / min(vfov1, vfov2)
        return float(np.clip(h_overlap * v_overlap, 0.0, 1.0))

    # ── Pair builders ───────────────────────────────────────────────

    def build_tile_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        if self.pairing_strategy == "adjacent":
            return self._build_adjacent_pairs(tile_metas)
        if self.pairing_strategy == "distance_based":
            return self._build_distance_pairs(tile_metas)
        if self.pairing_strategy == "all_pairs":
            return self._build_all_pairs(tile_metas)
        raise ValueError(f"Unknown pairing strategy: {self.pairing_strategy}")

    def _build_adjacent_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []

        # Group by pitch
        pitch_groups: Dict[float, List[int]] = {}
        for i, meta in enumerate(tile_metas):
            pitch = round(meta["pitch_deg"], 1)
            pitch_groups.setdefault(pitch, []).append(i)

        # Horizontal adjacent pairs within each pitch level
        for pitch, indices in pitch_groups.items():
            sorted_idx = sorted(indices, key=lambda j: tile_metas[j]["yaw_deg"])
            for j in range(len(sorted_idx) - 1):
                i1, i2 = sorted_idx[j], sorted_idx[j + 1]
                ov = self.compute_overlap_area(
                    tile_metas[i1]["yaw_deg"], tile_metas[i1]["pitch_deg"],
                    tile_metas[i1]["hfov_deg"], tile_metas[i1]["vfov_deg"],
                    tile_metas[i2]["yaw_deg"], tile_metas[i2]["pitch_deg"],
                    tile_metas[i2]["hfov_deg"], tile_metas[i2]["vfov_deg"],
                )
                if ov > 0.01:
                    pairs.append((i1, i2, ov))

            # Wrap-around pair
            if len(sorted_idx) > 2:
                i1, i2 = sorted_idx[-1], sorted_idx[0]
                yaw_diff = abs(tile_metas[i1]["yaw_deg"] - tile_metas[i2]["yaw_deg"])
                if yaw_diff > 180:
                    ov = self.compute_overlap_area(
                        tile_metas[i1]["yaw_deg"], tile_metas[i1]["pitch_deg"],
                        tile_metas[i1]["hfov_deg"], tile_metas[i1]["vfov_deg"],
                        tile_metas[i2]["yaw_deg"], tile_metas[i2]["pitch_deg"],
                        tile_metas[i2]["hfov_deg"], tile_metas[i2]["vfov_deg"],
                    )
                    if ov > 0.01:
                        pairs.append((i1, i2, ov))

        # Vertical adjacent pairs between pitch levels
        pitch_levels = sorted(pitch_groups.keys())
        for k in range(len(pitch_levels) - 1):
            p1, p2 = pitch_levels[k], pitch_levels[k + 1]
            for i1 in pitch_groups[p1]:
                for i2 in pitch_groups[p2]:
                    yaw_diff = abs(tile_metas[i1]["yaw_deg"] - tile_metas[i2]["yaw_deg"])
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    if yaw_diff < self.hfov_deg:
                        ov = self.compute_overlap_area(
                            tile_metas[i1]["yaw_deg"], tile_metas[i1]["pitch_deg"],
                            tile_metas[i1]["hfov_deg"], tile_metas[i1]["vfov_deg"],
                            tile_metas[i2]["yaw_deg"], tile_metas[i2]["pitch_deg"],
                            tile_metas[i2]["hfov_deg"], tile_metas[i2]["vfov_deg"],
                        )
                        if ov > 0.01:
                            pairs.append((i1, i2, ov))
        return pairs

    def _build_distance_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        n = len(tile_metas)
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.angular_distance(
                    tile_metas[i]["yaw_deg"], tile_metas[i]["pitch_deg"],
                    tile_metas[j]["yaw_deg"], tile_metas[j]["pitch_deg"],
                )
                if dist < self.distance_threshold:
                    ov = self.compute_overlap_area(
                        tile_metas[i]["yaw_deg"], tile_metas[i]["pitch_deg"],
                        tile_metas[i]["hfov_deg"], tile_metas[i]["vfov_deg"],
                        tile_metas[j]["yaw_deg"], tile_metas[j]["pitch_deg"],
                        tile_metas[j]["hfov_deg"], tile_metas[j]["vfov_deg"],
                    )
                    if ov > 0.01:
                        pairs.append((i, j, ov))
        return pairs

    def _build_all_pairs(self, tile_metas: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        n = len(tile_metas)
        for i in range(n):
            for j in range(i + 1, n):
                ov = self.compute_overlap_area(
                    tile_metas[i]["yaw_deg"], tile_metas[i]["pitch_deg"],
                    tile_metas[i]["hfov_deg"], tile_metas[i]["vfov_deg"],
                    tile_metas[j]["yaw_deg"], tile_metas[j]["pitch_deg"],
                    tile_metas[j]["hfov_deg"], tile_metas[j]["vfov_deg"],
                )
                if ov > 0.01:
                    pairs.append((i, j, ov))
        return pairs


# ── Loss computation ────────────────────────────────────────────────

def compute_vicreg_anyres_loss(
    vision_features: torch.Tensor,
    batch_size: int,
    tile_metas_batch: List[List[Dict[str, Any]]],
    vicreg_loss_module: Any,
    pairing_strategy: str = "adjacent",
    hfov_deg: float = 90.0,
    pair_chunk: int = 8,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """Compute VICReg loss over overlapping AnyRes tile pairs."""

    num_views = vision_features.size(0) // batch_size

    patch_features = vision_features[:, 1:] if has_cls_token else vision_features
    num_patches = patch_features.size(1)
    D = patch_features.size(-1)

    grid_h = int(np.sqrt(num_patches))
    grid_w = num_patches // grid_h

    patch_features_grid = patch_features.view(batch_size, num_views, grid_h, grid_w, D)

    all_curr: list[torch.Tensor] = []
    all_next: list[torch.Tensor] = []

    for b in range(batch_size):
        pairing = AnyResVICRegPairing(
            hfov_deg=hfov_deg, vfov_deg=hfov_deg,
            overlap_ratio=0.5, pairing_strategy=pairing_strategy,
        )
        pairs = pairing.build_tile_pairs(tile_metas_batch[b])
        if not pairs:
            continue

        for idx1, idx2, overlap_ratio in pairs:
            if tile_metas_batch[b][idx1]["tile_id"] == 0 or tile_metas_batch[b][idx2]["tile_id"] == 0:
                continue

            feat1 = patch_features_grid[b, idx1]
            feat2 = patch_features_grid[b, idx2]

            if overlap_ratio > 0.1:
                k = max(1, int(grid_w * overlap_ratio))
                all_curr.append(feat1[:, -k:, :].reshape(-1, D))
                all_next.append(feat2[:, :k, :].reshape(-1, D))

    if not all_curr:
        return torch.zeros((), device=vision_features.device)

    curr = torch.stack(all_curr)
    nxt = torch.stack(all_next)

    eps = 1e-4
    gamma = getattr(vicreg_loss_module, "gamma", 1.0)
    w_inv = getattr(vicreg_loss_module, "similarity_weight", 25.0)
    w_var = getattr(vicreg_loss_module, "variance_weight", 25.0)
    w_cov = getattr(vicreg_loss_module, "covariance_weight", 1.0)

    P, L = curr.shape[0], curr.shape[1]

    inv_pair = F.mse_loss(curr, nxt, reduction="none").mean(dim=(1, 2))
    std_x = torch.sqrt(curr.var(dim=1, unbiased=False) + eps)
    std_y = torch.sqrt(nxt.var(dim=1, unbiased=False) + eps)
    var_pair = 0.5 * (F.relu(gamma - std_x).mean(dim=1) + F.relu(gamma - std_y).mean(dim=1))

    curr_c = curr - curr.mean(dim=1, keepdim=True)
    nxt_c = nxt - nxt.mean(dim=1, keepdim=True)
    denom = max(L - 1, 1)

    def _cov_offdiag(xc: torch.Tensor) -> torch.Tensor:
        chunks = range(0, xc.size(0), pair_chunk) if pair_chunk < xc.size(0) else [0]
        parts = []
        for s in chunks:
            e = min(s + pair_chunk, xc.size(0))
            C = torch.bmm(xc[s:e].transpose(1, 2), xc[s:e]) / denom
            C2 = (C**2).sum(dim=(1, 2))
            diag_sq = torch.square(torch.diagonal(C, dim1=1, dim2=2)).sum(dim=1)
            parts.append((C2 - diag_sq) / D)
            del C
        return torch.cat(parts)

    cov_pair = 0.5 * (_cov_offdiag(curr_c) + _cov_offdiag(nxt_c))
    total = (w_inv * inv_pair + w_var * var_pair + w_cov * cov_pair).mean()
    return torch.clamp(total, max=1e6)
