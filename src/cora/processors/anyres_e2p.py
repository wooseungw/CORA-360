"""
ERP (Equirectangular) -> Pinhole tile generation for AnyRes-style preprocessing.

Generates a global context image + grid of perspective tiles from a 360-degree
equirectangular panorama. Supports closed-loop yaw division for seamless coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from py360convert import e2p as erp_to_persp

    _HAS_PY360 = True
except Exception:
    _HAS_PY360 = False
    erp_to_persp = None  # type: ignore[assignment]


# ── Utilities ───────────────────────────────────────────────────────

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def yaw_pitch_to_xyz(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """Convert yaw/pitch (degrees) to unit vector (x, y, z)."""
    yr, pr = deg2rad(yaw_deg), deg2rad(pitch_deg)
    cp = math.cos(pr)
    return (cp * math.cos(yr), math.sin(pr), cp * math.sin(yr))


def letterbox_square(img: Image.Image, size: int, fill: Tuple[int, ...] = (0, 0, 0)) -> Image.Image:
    """Resize longest edge to *size* and pad shorter edge to make a square."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_to_vit(t: torch.Tensor, vit_size: Optional[int]) -> torch.Tensor:
    if vit_size is None:
        return t
    return torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(vit_size, vit_size), mode="bilinear", align_corners=False
    ).squeeze(0)


# ── Yaw / Pitch center computation ─────────────────────────────────

def _norm_angle_180(x: float) -> float:
    """Normalize angle to [-180, 180)."""
    return ((x + 180.0) % 360.0) - 180.0


def make_yaw_centers_standard(
    hfov_deg: float,
    overlap: float,
    yaw_start: float = -180.0,
    yaw_end: float = 180.0,
    phase_deg: float = 0.0,
) -> List[float]:
    """Open-interval yaw centers (default mode)."""
    assert 0.0 <= overlap < 1.0
    step = hfov_deg * (1.0 - overlap)
    if step <= 0:
        raise ValueError("Invalid overlap leading to non-positive step.")
    centers: List[float] = []
    cur = yaw_start + hfov_deg / 2.0 + phase_deg
    while cur < yaw_end - hfov_deg / 2.0 + 1e-9:
        centers.append(_norm_angle_180(cur))
        cur += step
    return centers


def make_yaw_centers_closed_loop(
    hfov_deg: float,
    overlap: float,
    start_deg: float = -180.0,
    seam_phase_deg: float = 0.0,
) -> List[float]:
    """Closed-loop yaw division: uniform spacing that exactly wraps 360 degrees."""
    assert 0.0 <= overlap < 1.0
    step_raw = hfov_deg * (1.0 - overlap)
    n = max(1, math.ceil(360.0 / step_raw))
    step = 360.0 / n

    centers: List[float] = []
    cur = start_deg + seam_phase_deg
    for _ in range(n):
        centers.append(_norm_angle_180(cur))
        cur += step

    uniq: List[float] = []
    for c in centers:
        if all(abs(_norm_angle_180(c - u)) > 1e-6 for u in uniq):
            uniq.append(c)
    return sorted(uniq, key=lambda x: ((x + 360.0) % 360.0))


def resolve_yaw_geometry(
    hfov_deg: float,
    render_overlap: float,
    closed_loop: bool = True,
) -> Tuple[int, float]:
    """Resolve the *realized* yaw-tile geometry from the rendering knobs.

    Returns ``(n_tiles, phys_overlap)`` where ``phys_overlap`` is the fraction
    of a tile's horizontal extent that is physically shared with each adjacent
    tile, as actually produced by the renderer.

    This mirrors :func:`make_yaw_centers_closed_loop` /
    :func:`make_yaw_centers_standard` exactly so it cannot drift. The closed-loop
    path quantizes the tile count (``n = ceil(360 / step_raw)``), so the realized
    physical overlap generally differs from ``render_overlap`` — which is why the
    overlap loss must derive its ``overlap_ratio`` from *this* value rather than
    from the configured ``overlap`` or a hard-coded constant.
    """
    if hfov_deg <= 0.0:
        return 1, 0.0
    render_overlap = float(max(0.0, min(render_overlap, 0.999999)))
    step_raw = hfov_deg * (1.0 - render_overlap)
    if step_raw <= 0.0:
        return 1, 0.0
    if closed_loop:
        n = max(1, math.ceil(360.0 / step_raw))
        step = 360.0 / n
    else:
        n = max(1, int(math.floor(360.0 / step_raw)))
        step = step_raw
    phys_overlap = max(0.0, 1.0 - step / hfov_deg)
    return n, float(phys_overlap)


def maybe_add_seam_center(centers: List[float], seam_center: float = -180.0) -> List[float]:
    """Force-add a center at the seam (±180°) if not already present."""

    def ang_diff(a: float, b: float) -> float:
        return abs(_norm_angle_180(a - b))

    if all(ang_diff(c, seam_center) > 1e-6 for c in centers):
        return [_norm_angle_180(seam_center)] + centers
    return centers


def make_pitch_centers(
    vfov_deg: float,
    overlap: float,
    pitch_min: float,
    pitch_max: float,
) -> List[float]:
    """Sliding pitch centers within a vertical range."""
    if abs(pitch_max - pitch_min) < 1e-6:
        return [pitch_min]
    assert pitch_min < pitch_max
    step = vfov_deg * (1.0 - overlap)
    if step <= 0:
        raise ValueError("Invalid overlap leading to non-positive step.")
    centers: List[float] = []
    cur = pitch_min + vfov_deg / 2.0
    while cur <= pitch_max - vfov_deg / 2.0 + 1e-9:
        centers.append(cur)
        cur += step
    return centers


# ── Tile generation ─────────────────────────────────────────────────

@dataclass
class TileMeta:
    tile_id: int
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    center_xyz: Tuple[float, float, float]


@dataclass
class AnyResPack:
    global_image: torch.Tensor  # (3, G, G)
    tiles: torch.Tensor  # (N, 3, T, T)
    metas: List[TileMeta]
    global_meta: Dict[str, object]
    yaw_geometry: Dict[str, object] = field(default_factory=dict)
    # yaw_geometry keys (filled by build_anyres_from_erp):
    #   hfov_deg, render_overlap, phys_overlap, n_tiles,
    #   yaw_centers_deg (List[float]), closed_loop_yaw (bool), include_global (bool implied by caller)


def compute_vfov_from_hfov(hfov_deg: float, out_size: int) -> float:
    """For square output, vFOV == hFOV."""
    return hfov_deg


def erp_to_pinhole_tile(
    erp: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    out_size: int,
) -> Image.Image:
    if not _HAS_PY360:
        raise ImportError("py360convert required: `pip install py360convert opencv-python`")
    vfov = compute_vfov_from_hfov(hfov_deg, out_size)

    if erp.dtype != np.uint8:
        erp_u8 = (np.clip(erp, 0, 1) * 255).astype(np.uint8) if erp.max() <= 1.0 else erp.astype(np.uint8)
    else:
        erp_u8 = erp
    bgr = erp_u8[:, :, ::-1]
    persp = erp_to_persp(bgr, hfov_deg, yaw_deg, pitch_deg, out_hw=(out_size, out_size))
    rgb = persp[:, :, ::-1]
    return Image.fromarray(rgb)


def build_anyres_from_erp(
    erp_img: Image.Image,
    base_size: int = 336,
    tile_render_size: int = 672,
    vit_size: Optional[int] = None,
    hfov_deg: float = 90.0,
    overlap: float = 0.2,
    closed_loop_yaw: bool = False,
    yaw_phase_deg: float = 0.0,
    include_seam_center: bool = False,
    pitch_min: float = -45.0,
    pitch_max: float = 45.0,
    pitch_full_span: bool = False,
    cap_eps: float = 0.5,
) -> AnyResPack:
    """Generate global context + perspective tiles from an equirectangular panorama."""

    # Global image (context)
    global_img = letterbox_square(erp_img.convert("RGB"), base_size)
    g_tensor = resize_to_vit(pil_to_tensor(global_img), vit_size)

    # Pitch boundaries
    if pitch_full_span:
        pitch_min = max(pitch_min, -90.0 + cap_eps)
        pitch_max = min(pitch_max, 90.0 - cap_eps)

    erp_np = np.array(erp_img.convert("RGB"))
    vfov_deg = compute_vfov_from_hfov(hfov_deg, tile_render_size)

    # Yaw centers
    if closed_loop_yaw:
        yaws = make_yaw_centers_closed_loop(hfov_deg, overlap, start_deg=-180.0, seam_phase_deg=yaw_phase_deg)
    else:
        yaws = make_yaw_centers_standard(hfov_deg, overlap, yaw_start=-180.0, yaw_end=180.0, phase_deg=yaw_phase_deg)

    if include_seam_center:
        yaws = maybe_add_seam_center(yaws, seam_center=-180.0)

    pitches = make_pitch_centers(vfov_deg, overlap, pitch_min=pitch_min, pitch_max=pitch_max)

    # Tile generation
    tiles_list: List[torch.Tensor] = []
    metas: List[TileMeta] = []
    tid = 0
    for p in pitches:
        for y in yaws:
            tile = erp_to_pinhole_tile(erp_np, yaw_deg=y, pitch_deg=p, hfov_deg=hfov_deg, out_size=tile_render_size)
            t_tensor = resize_to_vit(pil_to_tensor(tile), vit_size)
            tiles_list.append(t_tensor)
            metas.append(
                TileMeta(
                    tile_id=tid,
                    yaw_deg=y,
                    pitch_deg=p,
                    hfov_deg=hfov_deg,
                    vfov_deg=vfov_deg,
                    center_xyz=yaw_pitch_to_xyz(y, p),
                )
            )
            tid += 1

    tiles = torch.stack(tiles_list, dim=0) if tiles_list else torch.empty(0, 3, tile_render_size, tile_render_size)

    gmeta: Dict[str, object] = {
        "kind": "global_letterbox",
        "base_size": base_size,
        "vit_size": vit_size,
    }
    _, phys_overlap = resolve_yaw_geometry(hfov_deg, overlap, closed_loop=closed_loop_yaw)
    yaw_geometry: Dict[str, object] = {
        "hfov_deg": float(hfov_deg),
        "render_overlap": float(overlap),
        "phys_overlap": float(phys_overlap),
        "n_tiles": int(len(yaws)),
        "yaw_centers_deg": [float(y) for y in yaws],
        "closed_loop_yaw": bool(closed_loop_yaw),
    }
    return AnyResPack(
        global_image=g_tensor,
        tiles=tiles,
        metas=metas,
        global_meta=gmeta,
        yaw_geometry=yaw_geometry,
    )
