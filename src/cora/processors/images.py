"""
Panorama image processor: ERP -> multi-view tensor conversion.

Supports strategies: sliding_window, e2p, cubemap, anyres, anyres_max, anyres_e2p, resize.
Optionally delegates to HuggingFace AutoProcessor for normalization/resizing.
"""

from __future__ import annotations

import logging
import math
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms

from py360convert import e2p, e2c

from cora.processors.anyres_e2p import (
    AnyResPack,
    TileMeta,
    build_anyres_from_erp,
)

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

VALID_STRATEGIES = frozenset({
    "sliding_window", "e2p", "cubemap",
    "anyres", "anyres_max", "anyres_e2p", "resize",
})

DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]
SIGLIP_IMAGE_MEAN = [0.5, 0.5, 0.5]
SIGLIP_IMAGE_STD = [0.5, 0.5, 0.5]
CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

CENTRAL_CROP_RATIO = 0.9
POLAR_MARGIN_RATIO = 12
HORIZON_MARGIN_RATIO = 4
E2P_CROP_RATIO = 0.8
CUBEMAP_FACE_SIZE = 256
ASPECT_RATIO_TOLERANCE = 0.1


# ── Main processor ──────────────────────────────────────────────────

class PanoramaImageProcessor:
    """Convert panoramic images to multi-view tensors."""

    def __init__(
        self,
        image_size: Optional[Tuple[int, int]] = None,
        crop_strategy: str = "anyres_e2p",
        fov_deg: float = 90.0,
        overlap_ratio: float = 0.5,
        normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        anyres_patch_size: Optional[int] = None,
        anyres_max_patches: int = 12,
        anyres_image_grid_pinpoints: Optional[List[Tuple[int, int]]] = None,
        # AnyRes ERP params
        anyres_e2p_base_size: int = 336,
        anyres_e2p_tile_size: int = 672,
        anyres_e2p_vit_size: Optional[int] = None,
        anyres_e2p_closed_loop: bool = True,
        anyres_e2p_pitch_range: Tuple[float, float] = (-45.0, 45.0),
        # Vision processor delegation
        use_vision_processor: bool = False,
        vision_model_name: Optional[str] = None,
    ) -> None:
        if image_size is None:
            image_size = self._infer_image_size(vision_model_name)

        if crop_strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid crop_strategy: {crop_strategy}. Must be one of {sorted(VALID_STRATEGIES)}")

        self.image_size = image_size
        self.crop_strategy = crop_strategy
        self.fov_deg = fov_deg
        self.overlap_ratio = overlap_ratio

        # AnyRes standard
        self.anyres_patch_size = anyres_patch_size or min(image_size)
        self.anyres_max_patches = anyres_max_patches
        self.anyres_image_grid_pinpoints = anyres_image_grid_pinpoints or [
            (336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008),
            (672, 1008), (1008, 672), (336, 1344), (1344, 336), (1008, 1008),
        ]

        # AnyRes ERP
        self.anyres_e2p_base_size = anyres_e2p_base_size
        self.anyres_e2p_tile_size = anyres_e2p_tile_size
        self.anyres_e2p_vit_size = anyres_e2p_vit_size
        self.anyres_e2p_closed_loop = anyres_e2p_closed_loop
        self.anyres_e2p_pitch_min, self.anyres_e2p_pitch_max = anyres_e2p_pitch_range

        # Vision processor
        self.use_vision_processor = use_vision_processor
        self.vision_model_name = vision_model_name
        self._vision_processor: Any = None

        # Normalization
        self.image_mean = image_mean or self._infer_normalization_mean(vision_model_name)
        self.image_std = image_std or self._infer_normalization_std(vision_model_name)

        # Transforms
        tf: list[Any] = [transforms.ToTensor()]
        if normalize and not use_vision_processor:
            tf.append(transforms.Normalize(self.image_mean, self.image_std))
        tf.append(transforms.Lambda(lambda t: t.contiguous()))
        self.to_tensor = transforms.Compose(tf)

        # Separate tensor normalizer for strategies that bypass to_tensor
        # (e.g. anyres_e2p which returns raw [0,1] tensors from build_anyres_from_erp)
        self._tensor_normalize: Optional[transforms.Normalize] = (
            transforms.Normalize(self.image_mean, self.image_std)
            if normalize and not use_vision_processor
            else None
        )

        self.num_views = self._calculate_num_views()
        self.view_metadata: List[Dict[str, Any]] = []
        self.tile_metas: List[Dict[str, Any]] = []

    # ── Static helpers ──────────────────────────────────────────────

    @staticmethod
    def _infer_image_size(vision_model_name: Optional[str]) -> Tuple[int, int]:
        default = (224, 224)
        if not vision_model_name:
            return default
        try:
            from transformers import AutoImageProcessor
            iproc = AutoImageProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
            for key in ("size", "crop_size", "image_size"):
                val = getattr(iproc, key, None)
                if isinstance(val, dict):
                    if "height" in val and "width" in val:
                        return (val["height"], val["width"])
                    if "shortest_edge" in val:
                        s = val["shortest_edge"]
                        return (s, s)
                elif isinstance(val, int):
                    return (val, val)
        except Exception:
            pass
        return default

    @staticmethod
    def _infer_normalization_mean(vision_model_name: Optional[str]) -> List[float]:
        if not vision_model_name:
            return list(DEFAULT_IMAGE_MEAN)
        try:
            from transformers import AutoImageProcessor
            iproc = AutoImageProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
            mean = getattr(iproc, "image_mean", None)
            if mean is not None:
                return list(map(float, mean))
        except Exception:
            pass
        return list(DEFAULT_IMAGE_MEAN)

    @staticmethod
    def _infer_normalization_std(vision_model_name: Optional[str]) -> List[float]:
        if not vision_model_name:
            return list(DEFAULT_IMAGE_STD)
        try:
            from transformers import AutoImageProcessor
            iproc = AutoImageProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
            std = getattr(iproc, "image_std", None)
            if std is not None:
                return list(map(float, std))
        except Exception:
            pass
        return list(DEFAULT_IMAGE_STD)

    @staticmethod
    def _to_pil(x: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            return Image.fromarray(x).convert("RGB")
        if isinstance(x, str):
            if x.startswith(("http://", "https://")):
                r = requests.get(x, timeout=10)
                r.raise_for_status()
                return Image.open(BytesIO(r.content)).convert("RGB")
            return Image.open(x).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(x)}")

    # ── View count ──────────────────────────────────────────────────

    def _calculate_num_views(self) -> int:
        if self.crop_strategy in ("sliding_window", "e2p"):
            stride = self.fov_deg * (1 - self.overlap_ratio)
            return math.ceil(360 / stride)
        if self.crop_strategy == "cubemap":
            return 4
        if self.crop_strategy == "resize":
            return 1
        if self.crop_strategy in ("anyres", "anyres_max"):
            return 1 + self.anyres_max_patches
        if self.crop_strategy == "anyres_e2p":
            stride = self.fov_deg * (1 - self.overlap_ratio)
            n_yaw = math.ceil(360 / stride)
            # Match make_pitch_centers logic exactly:
            #   - equatorial case (pitch_min == pitch_max) → always 1 pitch row
            #   - otherwise: count centers from pitch_min+vfov/2 to pitch_max-vfov/2
            if abs(self.anyres_e2p_pitch_max - self.anyres_e2p_pitch_min) < 1e-6:
                n_pitch = 1  # equatorial: single row at pitch_min
            else:
                vfov = self.fov_deg  # square tiles → vfov == hfov
                first_p = self.anyres_e2p_pitch_min + vfov / 2.0
                last_p = self.anyres_e2p_pitch_max - vfov / 2.0
                n_pitch = max(1, int((last_p - first_p) / stride) + 1) if last_p >= first_p else 0
            return 1 + n_yaw * n_pitch
        return 1

    # ── Vision processor (lazy) ─────────────────────────────────────

    @property
    def vision_processor(self) -> Any:
        if self.use_vision_processor and self._vision_processor is None:
            from transformers import AutoProcessor
            self._vision_processor = AutoProcessor.from_pretrained(self.vision_model_name, use_fast=True)
        return self._vision_processor

    # ── Public interface ────────────────────────────────────────────

    def __call__(
        self,
        x: Union[str, Image.Image, np.ndarray, List[Image.Image]],
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """Process image(s) into multi-view tensor [B, V, C, H, W] or [V, C, H, W]."""
        if isinstance(x, list):
            tensors = [self._process_single(img) for img in x]
            views = torch.stack(tensors)
        else:
            views = self._process_single(x).unsqueeze(0)

        if return_metadata:
            return views, self.view_metadata
        return views

    def _process_single(self, x: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        pil = self._to_pil(x)
        self.view_metadata = []

        if self.use_vision_processor:
            pil_views = self._extract_pil_views(pil)
            return self._process_with_vision_processor(pil_views)

        dispatch = {
            "sliding_window": self._sliding,
            "e2p": self._e2p,
            "cubemap": self._cubemap4,
            "resize": self._resize,
            "anyres": self._anyres,
            "anyres_max": self._anyres_max,
            "anyres_e2p": self._anyres_e2p,
        }
        return dispatch[self.crop_strategy](pil)

    # ── PIL extraction (for vision processor mode) ──────────────────

    def _extract_pil_views(self, pil: Image.Image) -> List[Image.Image]:
        dispatch = {
            "sliding_window": self._sliding_pil,
            "e2p": self._e2p_pil,
            "cubemap": self._cubemap4_pil,
            "resize": lambda img: [img],
            "anyres": self._anyres_pil,
            "anyres_max": self._anyres_max_pil,
            "anyres_e2p": self._anyres_e2p_pil,
        }
        return dispatch[self.crop_strategy](pil)

    def _process_with_vision_processor(self, pil_views: List[Image.Image]) -> torch.Tensor:
        if not pil_views:
            return torch.zeros(0, 3, *self.image_size)
        pixel_values: torch.Tensor = self.vision_processor(images=pil_views, return_tensors="pt")["pixel_values"]
        return pixel_values

    # ── E2P ─────────────────────────────────────────────────────────

    def _generate_e2p_views(self, img: Image.Image) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        stride = self.fov_deg * (1 - self.overlap_ratio)
        yaws = ((np.arange(self.num_views) * stride) % 360).astype(float)
        yaws = np.where(yaws > 180, yaws - 360, yaws)

        keep = CENTRAL_CROP_RATIO
        img_arr = np.array(img)
        pil_views: List[Image.Image] = []
        metadata: List[Dict[str, Any]] = []

        for yaw in yaws:
            npv = e2p(img_arr, fov_deg=self.fov_deg, u_deg=float(yaw), v_deg=0, out_hw=self.image_size, mode="bilinear")
            h = npv.shape[0]
            cut = int(h * (1 - keep) / 2)
            npv = npv[cut : h - cut]
            eff_fov = float(2 * np.degrees(np.arctan(keep * np.tan(np.radians(self.fov_deg / 2)))))
            metadata.append({"yaw": float(yaw), "pitch": 0.0, "effective_fov": eff_fov, "view_index": len(pil_views)})
            pil_views.append(Image.fromarray(npv.astype(np.uint8)))

        return pil_views, metadata

    def _e2p(self, img: Image.Image) -> torch.Tensor:
        pil_views, metadata = self._generate_e2p_views(img)
        self.view_metadata = metadata
        views = torch.empty(len(pil_views), 3, *self.image_size, dtype=torch.float32)
        for i, pil in enumerate(pil_views):
            views[i] = self.to_tensor(pil.resize(self.image_size[::-1], Image.Resampling.LANCZOS))
        return views

    def _e2p_pil(self, img: Image.Image) -> List[Image.Image]:
        pil_views, metadata = self._generate_e2p_views(img)
        self.view_metadata = metadata
        return pil_views

    # ── Sliding window ──────────────────────────────────────────────

    def _generate_sliding_views(self, img: Image.Image) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        W, H = img.size
        vw = int(W * self.fov_deg / 360)
        stride = int(vw * (1 - self.overlap_ratio))
        polar_margin = H // POLAR_MARGIN_RATIO
        crop_top = max(0, polar_margin)
        crop_bottom = min(H, H - polar_margin)

        pil_views: List[Image.Image] = []
        metadata: List[Dict[str, Any]] = []

        for i in range(self.num_views):
            s = i * stride
            if s + vw <= W:
                patch = img.crop((s, 0, s + vw, H))
            else:
                patch = Image.new("RGB", (vw, H))
                w1 = W - s
                patch.paste(img.crop((s, 0, W, H)), (0, 0))
                patch.paste(img.crop((0, 0, vw - w1, H)), (w1, 0))

            smart = patch.crop((0, crop_top, vw, crop_bottom))
            yaw = (i * stride * 360.0 / W) % 360.0
            if yaw > 180:
                yaw -= 360
            metadata.append({"yaw": yaw, "pitch": 0.0, "view_index": i})
            pil_views.append(smart)

        return pil_views, metadata

    def _sliding(self, img: Image.Image) -> torch.Tensor:
        pil_views, metadata = self._generate_sliding_views(img)
        self.view_metadata = metadata
        views = torch.empty(len(pil_views), 3, *self.image_size, dtype=torch.float32)
        for i, pil in enumerate(pil_views):
            views[i] = self.to_tensor(pil.resize(self.image_size[::-1], Image.Resampling.LANCZOS))
        return views

    def _sliding_pil(self, img: Image.Image) -> List[Image.Image]:
        pil_views, metadata = self._generate_sliding_views(img)
        self.view_metadata = metadata
        return pil_views

    # ── Cubemap ─────────────────────────────────────────────────────

    def _cubemap4(self, img: Image.Image) -> torch.Tensor:
        face_w = self.image_size[1]
        faces = e2c(np.array(img), face_w=face_w, cube_format="dict")
        order = ["F", "R", "B", "L"]
        views = torch.empty(4, 3, *self.image_size, dtype=torch.float32)
        for i, k in enumerate(order):
            face_img = Image.fromarray(faces[k].astype(np.uint8)).resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views[i] = self.to_tensor(face_img)
        return views

    def _cubemap4_pil(self, img: Image.Image) -> List[Image.Image]:
        faces = e2c(np.array(img), face_w=CUBEMAP_FACE_SIZE, cube_format="dict")
        return [Image.fromarray(faces[k].astype(np.uint8)) for k in ("F", "R", "B", "L")]

    # ── Resize ──────────────────────────────────────────────────────

    def _resize(self, img: Image.Image) -> torch.Tensor:
        resized = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        return self.to_tensor(resized).unsqueeze(0)

    # ── AnyRes (standard grid) ──────────────────────────────────────

    def _anyres(self, img: Image.Image) -> torch.Tensor:
        views: list[torch.Tensor] = []
        W, H = img.size

        # Global view
        views.append(self.to_tensor(img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))

        # Horizon patches
        center_h = H // 2
        margin = H // HORIZON_MARGIN_RATIO
        ht, hb = max(0, center_h - margin), min(H, center_h + margin)
        n_horiz = min(6, (self.num_views - 1) // 2)
        pw = W / n_horiz

        for i in range(n_horiz):
            if len(views) >= self.num_views:
                break
            patch = img.crop((int(i * pw), ht, int(min((i + 1) * pw, W)), hb))
            views.append(self.to_tensor(patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))

        # Full-height patches
        remaining = self.num_views - len(views)
        if remaining > 0:
            vp = min(remaining, n_horiz)
            pw2 = W / vp
            for i in range(vp):
                if len(views) >= self.num_views:
                    break
                patch = img.crop((int(i * pw2), 0, int(min((i + 1) * pw2, W)), H))
                views.append(self.to_tensor(patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))

        # E2P fill
        remaining = self.num_views - len(views)
        if remaining > 0:
            for yaw in np.linspace(0, 315, remaining, endpoint=False):
                if len(views) >= self.num_views:
                    break
                try:
                    npv = e2p(np.array(img), fov_deg=self.fov_deg, u_deg=float(yaw), v_deg=0, out_hw=self.image_size, mode="bilinear")
                    h = npv.shape[0]
                    m = int(h * (1 - E2P_CROP_RATIO) / 2)
                    npv = npv[m : h - m]
                    views.append(self.to_tensor(Image.fromarray(npv.astype(np.uint8)).resize(self.image_size[::-1], Image.Resampling.LANCZOS)))
                except Exception:
                    views.append(views[-1].clone() if views else torch.zeros(3, *self.image_size))

        while len(views) < self.num_views:
            views.append(views[-1].clone() if views else torch.zeros(3, *self.image_size))

        return torch.stack(views[: self.num_views])

    def _anyres_pil(self, img: Image.Image) -> List[Image.Image]:
        pil_views: list[Image.Image] = [img]
        W, H = img.size
        center_h = H // 2
        margin = H // HORIZON_MARGIN_RATIO
        ht, hb = max(0, center_h - margin), min(H, center_h + margin)
        n_horiz = min(6, (self.num_views - 1) // 2)
        pw = W / n_horiz
        for i in range(n_horiz):
            if len(pil_views) >= self.num_views:
                break
            pil_views.append(img.crop((int(i * pw), ht, int(min((i + 1) * pw, W)), hb)))
        return pil_views[: self.num_views]

    # ── AnyRes Max ──────────────────────────────────────────────────

    def _anyres_max(self, img: Image.Image) -> torch.Tensor:
        views: list[torch.Tensor] = []
        W, H = img.size

        views.append(self.to_tensor(img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))

        h_splits = min(6, self.anyres_max_patches // 2)
        v_splits = min(3, self.anyres_max_patches // h_splits)
        pw, ph = W // h_splits, H // v_splits

        for row in range(v_splits):
            for col in range(h_splits):
                if len(views) >= self.num_views:
                    break
                patch = img.crop((col * pw, row * ph, min((col + 1) * pw, W), min((row + 1) * ph, H)))
                views.append(self.to_tensor(patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))
            if len(views) >= self.num_views:
                break

        # E2P fill for remaining
        if len(views) < self.num_views:
            remaining = self.num_views - len(views)
            step = 360 / remaining
            for i in range(remaining):
                yaw = i * step
                if yaw > 180:
                    yaw -= 360
                try:
                    npv = e2p(np.array(img), fov_deg=self.fov_deg, u_deg=float(yaw), v_deg=0, out_hw=self.image_size, mode="bilinear")
                    views.append(self.to_tensor(Image.fromarray(npv.astype(np.uint8))))
                except Exception:
                    views.append(views[-1].clone() if views else torch.zeros(3, *self.image_size))

        while len(views) < self.num_views:
            views.append(views[-1].clone() if views else torch.zeros(3, *self.image_size))

        return torch.stack(views[: self.num_views])

    def _anyres_max_pil(self, img: Image.Image) -> List[Image.Image]:
        pil_views: list[Image.Image] = [img]
        W, H = img.size
        h_splits = min(6, self.anyres_max_patches // 2)
        v_splits = min(3, self.anyres_max_patches // h_splits)
        pw, ph = W // h_splits, H // v_splits
        for row in range(v_splits):
            for col in range(h_splits):
                if len(pil_views) >= self.num_views:
                    break
                pil_views.append(img.crop((col * pw, row * ph, min((col + 1) * pw, W), min((row + 1) * ph, H))))
            if len(pil_views) >= self.num_views:
                break
        return pil_views[: self.num_views]

    # ── AnyRes ERP ──────────────────────────────────────────────────

    def _resolve_vit_size(self) -> Optional[int]:
        """Return vit_size for build_anyres_from_erp.

        Falls back to ``min(self.image_size)`` so that global and tile tensors
        share the same spatial resolution (required for ``torch.cat``).
        """
        return self.anyres_e2p_vit_size or min(self.image_size)

    def _anyres_e2p(self, img: Image.Image) -> torch.Tensor:
        pack = build_anyres_from_erp(
            erp_img=img,
            base_size=self.anyres_e2p_base_size,
            tile_render_size=self.anyres_e2p_tile_size,
            vit_size=self._resolve_vit_size(),
            hfov_deg=self.fov_deg,
            overlap=self.overlap_ratio,
            closed_loop_yaw=self.anyres_e2p_closed_loop,
            pitch_min=self.anyres_e2p_pitch_min,
            pitch_max=self.anyres_e2p_pitch_max,
        )
        self.tile_metas = [
            {"tile_id": m.tile_id, "yaw_deg": m.yaw_deg, "pitch_deg": m.pitch_deg,
             "hfov_deg": m.hfov_deg, "vfov_deg": m.vfov_deg, "center_xyz": m.center_xyz}
            for m in pack.metas
        ]
        views = torch.cat([pack.global_image.unsqueeze(0), pack.tiles], dim=0)
        # build_anyres_from_erp returns [0,1] tensors without normalization;
        # apply the same normalization that to_tensor would for other strategies.
        if self._tensor_normalize is not None:
            views = torch.stack([self._tensor_normalize(v) for v in views])
        return views

    def _anyres_e2p_pil(self, img: Image.Image) -> List[Image.Image]:
        pack = build_anyres_from_erp(
            erp_img=img,
            base_size=self.anyres_e2p_base_size,
            tile_render_size=self.anyres_e2p_tile_size,
            vit_size=self._resolve_vit_size(),
            hfov_deg=self.fov_deg,
            overlap=self.overlap_ratio,
            closed_loop_yaw=self.anyres_e2p_closed_loop,
            pitch_min=self.anyres_e2p_pitch_min,
            pitch_max=self.anyres_e2p_pitch_max,
        )
        self.tile_metas = [
            {"tile_id": m.tile_id, "yaw_deg": m.yaw_deg, "pitch_deg": m.pitch_deg,
             "hfov_deg": m.hfov_deg, "vfov_deg": m.vfov_deg, "center_xyz": m.center_xyz}
            for m in pack.metas
        ]

        def _t2pil(t: torch.Tensor) -> Image.Image:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        pil_views = [_t2pil(pack.global_image)]
        for i in range(pack.tiles.size(0)):
            pil_views.append(_t2pil(pack.tiles[i]))
        return pil_views
