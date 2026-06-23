"""Baseline LoRA finetuning trainer for commercial VLMs.

Uses HuggingFace Trainer + PEFT LoRA on models loaded via
:class:`BaselineModelRegistry`. Ported from the
``LoRAAblationRunner`` in legacy/root_scripts/vlm_finetune_and_eval.py.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.utils.data
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from .config import BaselineConfig, PanoAdaptConfig, PanoViewConfig
from .models import BaselineModelRegistry

# Allow large panorama images without PIL decompression bomb warnings.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class _SplitQKV(torch.nn.Module):
    """Drop-in replacement for Qwen2.5-VL's fused vision ``attn.qkv``.

    Qwen2.5-VL vision attention uses one fused ``nn.Linear(dim, 3*dim)`` whose
    output is reshaped ``(seq, 3, num_heads, head_dim)`` → q, k, v (in that
    concatenation order). A single LoRA adapter on the fused matrix couples the
    q/k/v updates and is effectively low-rank-per-head. This wrapper exposes
    three independent ``q_proj``/``k_proj``/``v_proj`` Linears (initialised from
    the fused weight slices, so the forward is numerically identical at init) so
    PEFT can attach a separate LoRA to each — mirroring how InternVL/Gemma split
    vision attention is adapted by the default ``q_proj/k_proj/v_proj`` targets.
    """

    def __init__(self, fused: torch.nn.Linear) -> None:
        super().__init__()
        dim = fused.in_features
        out = fused.out_features
        if out != 3 * dim:
            raise ValueError(f"_SplitQKV expects out==3*in, got {out} != 3*{dim}")
        has_bias = fused.bias is not None
        self.q_proj = torch.nn.Linear(dim, dim, bias=has_bias)
        self.k_proj = torch.nn.Linear(dim, dim, bias=has_bias)
        self.v_proj = torch.nn.Linear(dim, dim, bias=has_bias)
        self.to(device=fused.weight.device, dtype=fused.weight.dtype)
        with torch.no_grad():
            w = fused.weight  # [3*dim, dim], rows: [q | k | v]
            self.q_proj.weight.copy_(w[0:dim])
            self.k_proj.weight.copy_(w[dim:2 * dim])
            self.v_proj.weight.copy_(w[2 * dim:3 * dim])
            if has_bias:
                b = fused.bias
                self.q_proj.bias.copy_(b[0:dim])
                self.k_proj.bias.copy_(b[dim:2 * dim])
                self.v_proj.bias.copy_(b[2 * dim:3 * dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)


def _split_qwen_vision_qkv(model: torch.nn.Module) -> int:
    """Replace every Qwen2.5-VL vision ``attn.qkv`` with a :class:`_SplitQKV`.

    Returns the number of attention blocks split. No-op (returns 0) for models
    without ``Qwen2_5_VLVisionAttention`` modules. Must run BEFORE
    ``get_peft_model`` so LoRA attaches to the split q/k/v Linears.
    """
    targets = [
        m for _, m in model.named_modules()
        if m.__class__.__name__ in ("Qwen2_5_VLVisionAttention", "Qwen2VLVisionAttention")
        and isinstance(getattr(m, "qkv", None), torch.nn.Linear)
    ]
    for m in targets:
        m.qkv = _SplitQKV(m.qkv)
    if targets:
        logger.info("split_vision_qkv: replaced %d fused vision attn.qkv → split q/k/v Linears", len(targets))
    return len(targets)


def build_generation_inputs(
    cfg: BaselineConfig,
    processor: Any,
    erp_image: Image.Image,
    prompt: str,
) -> Dict[str, Any]:
    """Build inference inputs using the exact baseline evaluation path."""
    return build_generation_inputs_with_meta(cfg, processor, erp_image, prompt)["inputs"]


def _maybe_erp_resize(cfg: BaselineConfig, image: Image.Image) -> Image.Image:
    """Pre-resize ERP to model.erp_resize_width/height if configured.

    Used by ERP-RoPE width-matched single-image experiments to force a
    deterministic post-tokenization grid (e.g. grid_w that matches what
    AnyRes-E2P would produce). When height is omitted, defaults to W//2
    (preserves ERP 2:1 aspect).
    """
    w = cfg.model.erp_resize_width
    h = cfg.model.erp_resize_height
    if not w:
        return image
    if not h:
        h = w // 2
    if image.size == (w, h):
        return image
    return image.resize((w, h), Image.BICUBIC)


def build_generation_inputs_with_meta(
    cfg: BaselineConfig,
    processor: Any,
    erp_image: Image.Image,
    prompt: str,
) -> Dict[str, Any]:
    """Build inference inputs plus metadata about the pre-processor input geometry."""
    pv = cfg.effective_pano_view
    is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}
    if pv is None and not is_seq2seq:
        erp_image = _maybe_erp_resize(cfg, erp_image)

    if is_seq2seq:
        inputs = processor(images=erp_image, text=prompt, return_tensors="pt")
        return {
            "inputs": inputs,
            "meta": {
                "num_views": 1,
                "raw_view_sizes": [erp_image.size],
                "raw_view_pixels": erp_image.size[0] * erp_image.size[1],
                "pe_mode": "seq2seq",
            },
        }

    if pv is not None:
        from cora.processors.anyres_e2p import build_anyres_from_erp
        import numpy as np

        if pv.strategy.lower() == "cubemap":
            hfov, overlap = 90.0, 0.0
        else:
            hfov, overlap = pv.hfov_deg, pv.overlap

        pack = build_anyres_from_erp(
            erp_img=erp_image,
            base_size=pv.base_size,
            tile_render_size=pv.tile_render_size,
            vit_size=pv.vit_size,
            hfov_deg=hfov,
            overlap=overlap,
            closed_loop_yaw=pv.closed_loop_yaw,
            pitch_min=pv.pitch_min,
            pitch_max=pv.pitch_max,
        )

        def _t2pil(t: torch.Tensor) -> Image.Image:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        eval_images: List[Image.Image] = []
        if pv.include_global:
            eval_images.append(_t2pil(pack.global_image))
        for ti in range(pack.tiles.size(0)):
            eval_images.append(_t2pil(pack.tiles[ti]))

        image_entries: List[Dict[str, str]] = [{"type": "image"} for _ in eval_images]
        messages = [
            {
                "role": "user",
                "content": image_entries + [{"type": "text", "text": prompt}],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=eval_images, return_tensors="pt", padding=True,
        )
        raw_view_sizes = [img.size for img in eval_images]
        meta = {
            "num_views": len(eval_images),
            "raw_view_sizes": raw_view_sizes,
            "raw_view_pixels": sum(w * h for w, h in raw_view_sizes),
            "pe_mode": "multiview",
            # Yaw geometry mirrors the train-time pano_meta so PanoramaYawRoPE
            # sees the same keys it expects.
            "include_global": bool(pv.include_global),
        }
        meta.update(pack.yaw_geometry)
        return {"inputs": inputs, "meta": meta}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[erp_image], return_tensors="pt", padding=True,
    )
    return {
        "inputs": inputs,
        "meta": {
            "num_views": 1,
            "raw_view_sizes": [erp_image.size],
            "raw_view_pixels": erp_image.size[0] * erp_image.size[1],
            "pe_mode": "single_view",
        },
    }


def estimate_visual_token_counts(
    cfg: BaselineConfig,
    processor: Any,
    inputs: Dict[str, Any],
) -> Dict[str, int]:
    """Estimate pre-encoder patches and post-compression visual token counts.

    `vision_patches` is the count immediately before the vision encoder.
    `visual_tokens` is the count after any model-specific merge/compression and
    corresponds to the LLM-side visual token count.
    """
    model_type = cfg.model.model_type.lower()
    ip = getattr(processor, "image_processor", None)
    total_tokens = int(inputs.get("input_ids", torch.tensor([])).shape[-1])

    if model_type in {"qwen_vl", "qwen25_vl", "qwen2_vl", "qwenvl"}:
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return {"vision_patches": 0, "visual_tokens": 0, "prompt_tokens": total_tokens, "total_tokens": total_tokens}
        patch_total = 0
        for t, h, w in grid.tolist():
            patch_total += int(t) * int(h) * int(w)
        merge_size = int(getattr(ip, "merge_size", 2)) if ip is not None else 2
        visual_tokens = patch_total // (merge_size * merge_size)
        return {
            "vision_patches": patch_total,
            "visual_tokens": visual_tokens,
            "prompt_tokens": max(0, total_tokens - visual_tokens),
            "total_tokens": total_tokens,
        }

    if model_type in {"internvl", "internvl_chat", "internvl25", "internvl_legacy"}:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return {"vision_patches": 0, "visual_tokens": 0, "prompt_tokens": total_tokens, "total_tokens": total_tokens}
        num_views = int(pixel_values.shape[0])
        vision_cfg = getattr(getattr(processor, "image_processor", None), "size", {})
        image_size = vision_cfg.get("height", 448) if isinstance(vision_cfg, dict) else 448
        patch_size = 14
        patches_per_view = (image_size // patch_size) * (image_size // patch_size)
        patch_total = num_views * patches_per_view
        image_seq_length = int(getattr(processor, "image_seq_length", 256))
        visual_tokens = num_views * image_seq_length
        return {
            "vision_patches": patch_total,
            "visual_tokens": visual_tokens,
            "prompt_tokens": max(0, total_tokens - visual_tokens),
            "total_tokens": total_tokens,
        }

    if model_type in {"gemma3", "gemma-3", "gemma_3"}:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return {"vision_patches": 0, "visual_tokens": 0, "prompt_tokens": total_tokens, "total_tokens": total_tokens}
        num_views = int(pixel_values.shape[1]) if pixel_values.ndim == 5 else int(pixel_values.shape[0])
        size = getattr(ip, "size", {"height": 896, "width": 896}) if ip is not None else {"height": 896, "width": 896}
        height = int(size.get("height", 896))
        width = int(size.get("width", 896))
        vision_cfg = getattr(getattr(processor, "image_processor", None), "size", size)
        _ = vision_cfg  # keep logic explicit; patch size comes from model config in Transformers.
        patch_size = 14
        patches_per_view = (height // patch_size) * (width // patch_size)
        patch_total = num_views * patches_per_view
        image_seq_length = int(getattr(processor, "image_seq_length", 256))
        visual_tokens = num_views * image_seq_length
        return {
            "vision_patches": patch_total,
            "visual_tokens": visual_tokens,
            "prompt_tokens": max(0, total_tokens - visual_tokens),
            "total_tokens": total_tokens,
        }

    return {
        "vision_patches": 0,
        "visual_tokens": 0,
        "prompt_tokens": total_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Auto batch-size (YOLO-style GPU memory profiling)
# ---------------------------------------------------------------------------

def autobatch(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn,
    fraction: float = 0.60,
    default_batch_size: int = 1,
) -> int:
    if not torch.cuda.is_available():
        logger.warning("AutoBatch: CUDA not available, using batch_size=%d", default_batch_size)
        return default_batch_size

    import numpy as np

    device = torch.device("cuda")
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / gb

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.to(device).train()

    allocated_after_model = torch.cuda.memory_allocated(device) / gb
    free = total - allocated_after_model

    logger.info(
        "AutoBatch: %s %.1fG total, %.1fG model, %.1fG free (%.0f%% target)",
        props.name, total, allocated_after_model, free, fraction * 100,
    )

    batch_sizes = [1, 2, 4, 8, 16]
    mem_usage = []

    for bs in batch_sizes:
        if bs > len(dataset):
            break
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            batch = collate_fn([dataset[i % len(dataset)] for i in range(bs)])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss.backward()
            peak = torch.cuda.max_memory_allocated(device) / gb
            mem_usage.append((bs, peak))
            model.zero_grad(set_to_none=True)
            del batch, outputs, loss
            torch.cuda.empty_cache()
            logger.info("  batch %2d -> %.2fG peak", bs, peak)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                logger.info("  batch %2d -> OOM", bs)
                break
            raise

    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.cuda.empty_cache()

    if len(mem_usage) < 2:
        logger.warning("AutoBatch: insufficient data, using batch_size=%d", default_batch_size)
        return default_batch_size

    xs, ys = zip(*mem_usage)
    p = np.polyfit(xs, ys, deg=1)
    optimal = int((free * fraction - p[1]) / p[0])

    oom_limit = batch_sizes[len(mem_usage)] if len(mem_usage) < len(batch_sizes) else batch_sizes[-1] * 2
    optimal = min(optimal, oom_limit - 1)
    optimal = max(optimal, 1)

    logger.info("AutoBatch: optimal batch_size = %d (%.1fG / %.1fG, %.0f%%)", optimal, np.polyval(p, optimal), total, fraction * 100)
    return optimal


def autobatch_generate(
    model: torch.nn.Module,
    sample_inputs_fn,
    max_new_tokens: int = 128,
    fraction: float = 0.85,
    default_batch_size: int = 1,
) -> int:
    if not torch.cuda.is_available():
        return default_batch_size

    import numpy as np

    device = next(model.parameters()).device
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / gb

    torch.cuda.empty_cache()
    allocated_model = torch.cuda.memory_allocated(device) / gb
    free = total - allocated_model

    logger.info(
        "AutoBatch (eval): %s %.1fG total, %.1fG free (%.0f%% target)",
        props.name, total, free, fraction * 100,
    )

    batch_sizes = [1, 2, 4, 8, 16]
    mem_usage = []

    for bs in batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            inputs = sample_inputs_fn(bs)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=min(max_new_tokens, 16), do_sample=False)
            peak = torch.cuda.max_memory_allocated(device) / gb
            mem_usage.append((bs, peak))
            del inputs
            torch.cuda.empty_cache()
            logger.info("  batch %2d -> %.2fG peak", bs, peak)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.info("  batch %2d -> OOM", bs)
                break
            raise

    if len(mem_usage) < 2:
        return default_batch_size

    xs, ys = zip(*mem_usage)
    p = np.polyfit(xs, ys, deg=1)
    optimal = int((free * fraction - p[1]) / p[0])

    oom_limit = batch_sizes[len(mem_usage)] if len(mem_usage) < len(batch_sizes) else batch_sizes[-1] * 2
    optimal = min(optimal, oom_limit - 1)
    optimal = max(optimal, 1)

    logger.info("AutoBatch (eval): optimal batch_size = %d", optimal)
    return optimal


# ---------------------------------------------------------------------------
# bf16 availability check (transformers compat shim)
# ---------------------------------------------------------------------------

try:
    from transformers.utils import is_torch_bf16_gpu_available as _is_bf16_supported
except ImportError:
    try:
        from transformers.utils import is_bfloat16_supported as _is_bf16_supported  # type: ignore[assignment]
    except ImportError:

        def _is_bf16_supported() -> bool:  # type: ignore[misc]
            if not torch.cuda.is_available():
                return False
            return getattr(torch.cuda, "is_bf16_supported", lambda: False)()


# ---------------------------------------------------------------------------
# VLM Dataset
# ---------------------------------------------------------------------------

# Upper bound on yaw tiles drawn by random (hfov, overlap) augmentation. With
# 256 tokens/view (InternVL @ 448px) and a global view, 12 tiles → 13 views →
# 3328 image tokens, which leaves headroom under training.max_length=4096 for
# the prompt + response. See _load_multi_images augment clamp.
_MAX_AUG_TILES = 12


class VLMDataset(torch.utils.data.Dataset):
    """CSV-backed dataset that yields processor-ready dicts.

    Supports single-image (default) and multi-image (pano_view) modes.
    When ``pano_view_config`` is provided, each sample produces multiple
    perspective tile views from the ERP panorama image based on the
    configured strategy (anyres_e2p, cubemap, or pinhole).
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        tokenizer: Any,
        model_type: str,
        image_column: str = "url",
        instruction_column: str = "instruction",
        response_column: str = "response",
        max_samples: Optional[int] = None,
        anyres_config: Optional["PanoViewConfig"] = None,
        pano_view_config: Optional["PanoViewConfig"] = None,
        erp_resize: Optional[tuple] = None,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.head(max_samples)

        self.processor = processor
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.pano_view_config = pano_view_config or anyres_config
        # (W, H) tuple to force-resize ERP before processor — used by ERP-RoPE
        # width-matched single-image runs. Only applied to the single-image path.
        self.erp_resize = erp_resize
        # Train-time random (hfov, overlap) augmentation switch. Honored only
        # when the pano_view_config provides *_range fields.
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.df)

    # ---- helpers ----------------------------------------------------------

    def _load_image(self, row: pd.Series) -> Image.Image:
        path = row.get(self.image_column)
        if path is None:
            raise ValueError(f"CSV missing column '{self.image_column}'")
        img_path = Path(str(path))
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.erp_resize is not None and not self.is_multi_image:
            w, h = self.erp_resize
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)
        return img

    def _load_multi_images(
        self, row: pd.Series,
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generate perspective views from ERP panorama based on strategy.

        Returns (views, pano_meta) where pano_meta carries yaw geometry
        (hfov_deg, phys_overlap, n_tiles, yaw_centers_deg, include_global, ...)
        for downstream PE/loss modules. The meta is a plain dict so it can flow
        through the HF dataloader/collate path.
        """
        from cora.processors.anyres_e2p import build_anyres_from_erp

        erp_img = self._load_image(row)
        cfg = self.pano_view_config
        assert cfg is not None

        strategy = cfg.strategy.lower()

        if strategy == "cubemap":
            hfov, overlap = 90.0, 0.0
        else:
            hfov, overlap = cfg.hfov_deg, cfg.overlap
            if self.augment:
                import random as _rnd
                if getattr(cfg, "hfov_deg_range", None):
                    lo, hi = float(cfg.hfov_deg_range[0]), float(cfg.hfov_deg_range[1])
                    hfov = _rnd.uniform(lo, hi)
                if getattr(cfg, "overlap_range", None):
                    lo, hi = float(cfg.overlap_range[0]), float(cfg.overlap_range[1])
                    overlap = _rnd.uniform(lo, hi)
                # Hard cap on realized n_tiles. The per-view image-token count
                # (n_tiles+global)*tokens_per_view must fit max_length, else the
                # collate truncation severs the image-token block and InternVL/
                # Qwen raise "image features and image tokens do not match".
                # The config overlap range is already chosen to satisfy this, but
                # clamp here too so a future range edit can't silently reintroduce
                # the crash. Reducing overlap (not hfov) keeps the requested FOV.
                from cora.processors.anyres_e2p import resolve_yaw_geometry
                n_t, _ = resolve_yaw_geometry(hfov, overlap, cfg.closed_loop_yaw)
                while n_t > _MAX_AUG_TILES and overlap > 0.0:
                    overlap = max(0.0, overlap - 0.05)
                    n_t, _ = resolve_yaw_geometry(hfov, overlap, cfg.closed_loop_yaw)

        pack = build_anyres_from_erp(
            erp_img=erp_img,
            base_size=cfg.base_size,
            tile_render_size=cfg.tile_render_size,
            vit_size=cfg.vit_size,
            hfov_deg=hfov,
            overlap=overlap,
            closed_loop_yaw=cfg.closed_loop_yaw,
            pitch_min=cfg.pitch_min,
            pitch_max=cfg.pitch_max,
        )

        import numpy as np

        def _t2pil(t: torch.Tensor) -> Image.Image:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        pil_views: List[Image.Image] = []
        if cfg.include_global:
            pil_views.append(_t2pil(pack.global_image))
        for i in range(pack.tiles.size(0)):
            pil_views.append(_t2pil(pack.tiles[i]))

        pano_meta: Dict[str, Any] = dict(pack.yaw_geometry)
        pano_meta["include_global"] = bool(cfg.include_global)
        pano_meta["num_views"] = len(pil_views)
        return pil_views, pano_meta

    def _get_text(self, row: pd.Series, column: str, default: str = "") -> str:
        val = str(row.get(column, default))
        if not val or val.lower() == "nan":
            return default
        return val

    @property
    def is_multi_image(self) -> bool:
        return self.pano_view_config is not None

    # ---- per-sample processing --------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt_idx in range(len(self.df)):
            try:
                real_idx = (idx + attempt_idx) % len(self.df)
                row = self.df.iloc[real_idx]
                prompt = self._get_text(row, self.instruction_column, "Describe the image.")
                response = self._get_text(row, self.response_column, "")

                if self.model_type in {"blip2", "blip-2"}:
                    image = self._load_image(row)
                    return self._prepare_seq2seq(image, prompt, response)

                if self.is_multi_image:
                    images, pano_meta = self._load_multi_images(row)
                    return self._prepare_causal_multi(images, prompt, response, pano_meta)

                image = self._load_image(row)
                return self._prepare_causal(image, prompt, response)
            except Exception as e:
                if attempt_idx == 0:
                    logger.warning(f"Skipping idx {idx}: {e}")
                continue
        raise RuntimeError(f"No valid sample found after {len(self.df)} attempts")

    def _prepare_causal(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Prepare a single-image causal-LM sample."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )

        return {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "image": image,
            "reference_text": response,
        }

    def _prepare_causal_multi(
        self,
        images: List[Image.Image],
        prompt: str,
        response: str,
        pano_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare a multi-image causal-LM sample (anyres-e2p tiles)."""
        image_entries: List[Dict[str, str]] = [{"type": "image"} for _ in images]
        messages = [
            {
                "role": "user",
                "content": image_entries + [{"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )

        sample: Dict[str, Any] = {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "images": images,
            "reference_text": response,
        }
        if pano_meta is not None:
            sample["pano_meta"] = pano_meta
        return sample

    def _prepare_seq2seq(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """Prepare an encoder-decoder sample (BLIP-2)."""
        enc = self.processor(images=image, text=response, return_tensors="pt")
        labels = enc["input_ids"].squeeze(0).clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        result: Dict[str, Any] = {"labels": labels}
        for key, value in enc.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)
            elif value is not None:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _resolve_image_token_id(model: Any, processor: Any) -> Optional[int]:
    """Best-effort lookup of the image-placeholder token id.

    Different VLM families expose it under different names / nesting. We probe
    the model config (and sub-configs) then the processor. Returns None if not
    found, in which case the collate falls back to plain truncation.
    """
    cfg = getattr(model, "config", None)
    cfgs = [cfg]
    for attr in ("text_config", "vision_config"):
        sub = getattr(cfg, attr, None)
        if sub is not None:
            cfgs.append(sub)
    for c in cfgs:
        for name in ("image_token_id", "image_token_index"):
            tid = getattr(c, name, None)
            if isinstance(tid, int):
                return tid
    tid = getattr(processor, "image_token_id", None)
    if isinstance(tid, int):
        return tid
    return None


def _collate_causal(
    processor: Any,
    tokenizer: Any,
    max_length: Optional[int] = None,
    image_token_id: Optional[int] = None,
) -> Any:
    """Return a collate function for causal VLMs.

    Handles both single-image (``feature["image"]``) and multi-image
    (``feature["images"]``) samples transparently.

    ``image_token_id`` (when provided) makes the post-processing truncation
    image-aware: it never cuts through the image-placeholder token block, which
    would desync the text image-token count from the vision feature count and
    crash InternVL/Qwen (``image features and image tokens do not match``).
    """

    # NOTE: Do NOT pass max_length/truncation to the processor — it breaks
    # multimodal models (e.g. InternVL) whose processors validate image
    # token counts before truncation.  We truncate manually after processing.
    proc_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}

    def _get_images(f: Dict[str, Any]) -> List[Image.Image]:
        """Extract image(s) from a single feature dict."""
        if "images" in f:
            return f["images"]
        return [f["image"]]

    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        per_sample_images = [_get_images(f) for f in features]

        # Qwen2.5-VL processor expects a flat image list; it matches images
        # to text via the number of <|image_pad|> token blocks in each text.
        flat_images: List[Image.Image] = []
        for imgs in per_sample_images:
            flat_images.extend(imgs)

        full_inputs = processor(
            text=full_texts, images=flat_images, **proc_kwargs,
        )

        # --- Post-process truncation (BEFORE prompt masking) ---
        if max_length is not None and full_inputs["input_ids"].shape[1] > max_length:
            ids = full_inputs["input_ids"]
            cut = max_length
            # Never truncate through image placeholder tokens: the vision tower
            # already produced features for every view, so dropping image tokens
            # from the text desyncs the two counts and crashes the model. If the
            # image block extends past max_length, keep the whole block (response
            # text past it is lost → the prompt-mask loop below will warn that
            # the sample has no training signal, which is preferable to a crash).
            if image_token_id is not None:
                is_img = ids == image_token_id
                if is_img.any():
                    last_img = int(is_img.nonzero(as_tuple=True)[1].max())
                    if last_img >= max_length:
                        cut = last_img + 1
                        logger.warning(
                            "collate: image block ends at token %d > max_length %d; "
                            "keeping %d tokens to avoid image/feature desync.",
                            last_img, max_length, cut,
                        )
            full_inputs["input_ids"] = ids[:, :cut]
            full_inputs["attention_mask"] = full_inputs["attention_mask"][:, :cut]

        labels = full_inputs["input_ids"].clone()

        for i in range(len(features)):
            single = processor(
                text=[prompt_texts[i]], images=per_sample_images[i],
                return_tensors="pt", padding=False,
            )
            prompt_len = min(single["input_ids"].shape[1], labels.shape[1])
            if prompt_len >= labels.shape[1]:
                logger.warning(
                    "Sample %d: prompt_len (%d) >= max_length (%d). "
                    "No training signal for this sample.",
                    i, single["input_ids"].shape[1], labels.shape[1],
                )
            labels[i, :prompt_len] = -100

        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        result: Dict[str, Any] = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }
        for key in full_inputs:
            if key not in result:
                result[key] = full_inputs[key]

        pano_meta_list = [f.get("pano_meta") for f in features]
        if any(m is not None for m in pano_meta_list):
            result["pano_meta_list"] = pano_meta_list
        return result

    return collate_fn


def _collate_seq2seq(tokenizer: Any) -> Any:
    """Return a collate function for encoder-decoder VLMs (BLIP-2)."""

    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [
            f["attention_mask"] for f in features
            if f.get("attention_mask") is not None
        ]

        pad_dict: Dict[str, Any] = {"input_ids": input_ids_list}
        if attn_list:
            pad_dict["attention_mask"] = attn_list
        padded = tokenizer.pad(pad_dict, padding=True, return_tensors="pt")

        labels = tokenizer.pad(
            {"input_ids": [f["labels"] for f in features]},
            padding=True, return_tensors="pt",
        )["input_ids"]
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        batch: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": padded["input_ids"],
            "labels": labels,
        }
        if "attention_mask" in padded:
            batch["attention_mask"] = padded["attention_mask"]
        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# Precision helper
# ---------------------------------------------------------------------------


def _resolve_precision(mp: Optional[str]) -> Dict[str, bool]:
    if mp is None:
        return {"fp16": False, "bf16": False}
    key = mp.lower()
    if key == "bfp16":
        logger.warning("Typo 'bfp16' detected in mixed_precision config; auto-correcting to 'bf16'.")
        key = "bf16"
    if key == "bf16":
        if _is_bf16_supported():
            return {"fp16": False, "bf16": True}
        logger.warning("bf16 not supported; falling back to fp16.")
        return {"fp16": True, "bf16": False}
    if key == "fp16":
        return {"fp16": True, "bf16": False}
    logger.warning("Unknown mixed_precision '%s'; disabling.", mp)
    return {"fp16": False, "bf16": False}


# ---------------------------------------------------------------------------
# BaselineTrainer
# ---------------------------------------------------------------------------


class _SafeTrainer(Trainer):
    """Trainer subclass that handles kwargs incompatible with some model forward() signatures.

    Newer versions of HF Trainer pass ``num_items_in_batch`` to
    ``compute_loss`` which eventually reaches ``model.forward()``.
    Models like T5 (used in BLIP-2) do not accept this kwarg.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pano_meta_list = inputs.pop("pano_meta_list", None) if isinstance(inputs, dict) else None
        if pano_meta_list is not None:
            self._last_pano_meta_list = pano_meta_list
        yaw_rope = getattr(self, "_yaw_rope", None)
        if yaw_rope is not None and pano_meta_list is not None:
            yaw_rope.set_meta(pano_meta_list)
        try:
            try:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            except TypeError as exc:
                if "num_items_in_batch" in str(exc):
                    kwargs.pop("num_items_in_batch", None)
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
                raise
        finally:
            if yaw_rope is not None:
                yaw_rope.clear_meta()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # In-loop eval and generate() do NOT route through compute_loss, so we
        # also pop+stash meta here. Without this, the yaw_rope hook would fire
        # with stale meta from the last train step and either silently distort
        # or raise on num_views mismatch.
        pano_meta_list = inputs.pop("pano_meta_list", None) if isinstance(inputs, dict) else None
        yaw_rope = getattr(self, "_yaw_rope", None)
        if yaw_rope is not None and pano_meta_list is not None:
            yaw_rope.set_meta(pano_meta_list)
        try:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
            )
        finally:
            if yaw_rope is not None:
                yaw_rope.clear_meta()

    def create_optimizer(self):
        """Optionally decouple the vision-tower LR from everything else.

        When ``self._vision_lr`` is set (BaselineTrainingConfig.vision_lr),
        trainable params whose name contains ``.visual.`` get their own AdamW
        group at that LR while LM LoRA / yaw_rope keep ``args.learning_rate``.
        A heavily pretrained ViT (Qwen2.5-VL) then adapts gently instead of
        being over-written at the LM's hotter LR. Falls back to the stock
        single-group optimizer when ``vision_lr`` is unset.
        """
        vision_lr = getattr(self, "_vision_lr", None)
        if vision_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        opt_model = self.model
        decay_names = set(self.get_decay_parameter_names(opt_model))
        base_lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)
        groups: Dict[str, List[Any]] = {"vd": [], "vn": [], "rd": [], "rn": []}
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            key = ("v" if ".visual." in n else "r") + ("d" if n in decay_names else "n")
            groups[key].append(p)
        specs = [
            ("vd", float(vision_lr), wd), ("vn", float(vision_lr), 0.0),
            ("rd", base_lr, wd), ("rn", base_lr, 0.0),
        ]
        grouped = [
            {"params": groups[k], "lr": lr, "weight_decay": w}
            for k, lr, w in specs if groups[k]
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs.pop("lr", None)            # per-group lr set above
        optimizer_kwargs.pop("weight_decay", None)  # per-group wd set above
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        n_vis = sum(p.numel() for p in groups["vd"] + groups["vn"])
        n_rest = sum(p.numel() for p in groups["rd"] + groups["rn"])
        logger.info(
            "Decoupled LR: vision_lr=%g (%d trainable params) | base_lr=%g (%d trainable params)",
            float(vision_lr), n_vis, base_lr, n_rest,
        )
        return self.optimizer


class _PanoAdaptTrainer(_SafeTrainer):
    """Trainer with PanoAdapt spatial PE and optional DenseCL auxiliary loss.

    Spatial PE (Layer 2): Before each forward pass, computes default
    M-RoPE position_ids via ``get_rope_index``, then shifts the width
    axis for panoramic tile views so adjacent views share continuous
    spatial encodings.

    DenseCL (Layer 3): Registers a forward hook on PatchMerger to extract
    vision features, then computes InfoNCE overlap loss as an auxiliary
    training signal added to the main LM loss.
    """

    def __init__(
        self,
        *args: Any,
        panoadapt_config: PanoAdaptConfig,
        pano_view_config: Optional[PanoViewConfig] = None,
        stage1_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pa_cfg = panoadapt_config
        self._include_global = pano_view_config.include_global if pano_view_config else True
        self._hook: Optional[Any] = None
        self._densecl_loss: Optional[torch.nn.Module] = None
        self._last_pano_meta_list: Optional[List[Dict[str, Any]]] = None
        self._yaw_rope: Optional[torch.nn.Module] = None
        self._yaw_rope_handle: Optional[Any] = None
        # When set, the yaw_rope module's log_inv_freq is initialized from the
        # Stage-1 SSL-pretrained state instead of the LLaMA log-spaced default.
        # Stage-2 then continues to train it alongside the LM LoRA.
        self._stage1_checkpoint: Optional[str] = stage1_checkpoint

        from cora.baseline.panoadapt import create_vlm_adapter

        model_type = getattr(panoadapt_config, "model_type", "qwen_vl")
        self._adapter = create_vlm_adapter(
            model_type=model_type,
            overlap_ratio=panoadapt_config.overlap_ratio,
            include_global=self._include_global,
        )

        if panoadapt_config.overlap_loss:
            from cora.baseline.panoadapt import VisionFeatureHook, create_panoadapt_loss

            self._hook = VisionFeatureHook()
            self._densecl_loss = create_panoadapt_loss(panoadapt_config)
            self._register_hook(self.model)

    # -- model unwrapping ---------------------------------------------------

    @staticmethod
    def _unwrap_to_rope_model(model: torch.nn.Module) -> torch.nn.Module:
        m = model
        while hasattr(m, "base_model"):
            next_m = m.base_model
            if next_m is m:
                break
            m = next_m
        while hasattr(m, "model") and not hasattr(m, "get_rope_index"):
            next_m = m.model
            if next_m is m:
                break
            m = next_m
        return m

    @staticmethod
    def _unwrap_to_cond_gen(model: torch.nn.Module) -> torch.nn.Module:
        m = model
        while hasattr(m, "base_model"):
            next_m = m.base_model
            if next_m is m:
                break
            m = next_m
        if hasattr(m, "model"):
            next_m = m.model
            if next_m is not m:
                m = next_m
        return m

    # -- hook management ----------------------------------------------------

    def _register_hook(self, model: torch.nn.Module) -> None:
        if self._hook is None:
            return
        cg = self._unwrap_to_cond_gen(model)
        hook_target = self._adapter.get_vision_hook_target() if self._adapter is not None else None
        self._hook.register(cg, hook_target_name=hook_target)
        logger.info("PanoAdapt: VisionFeatureHook registered")
        if getattr(self._pa_cfg, "yaw_rope_enabled", False):
            self._register_yaw_rope(cg, hook_target_name=hook_target)

    def _resolve_vision_projector(
        self, model: torch.nn.Module, hook_target_name: Optional[str],
    ) -> Optional[torch.nn.Module]:
        if hook_target_name is not None:
            for name, mod in model.named_modules():
                if name == hook_target_name or name.endswith(hook_target_name):
                    return mod
        # fall back to the same search VisionFeatureHook uses
        for name, mod in model.named_modules():
            lower = name.lower()
            if "merger" in lower or "multi_modal_projector" in lower:
                return mod
        return None

    def _register_yaw_rope(self, model: torch.nn.Module, hook_target_name: Optional[str]) -> None:
        from cora.model.positional import attach_yaw_rope_hook

        target = self._resolve_vision_projector(model, hook_target_name)
        if target is None:
            raise RuntimeError(
                "yaw_rope_enabled=True but no vision projector found; "
                "expected merger / multi_modal_projector module."
            )

        yaw_rope, handle = attach_yaw_rope_hook(
            model=model,
            target_module=target,
            rope_dim=self._pa_cfg.yaw_rope_dim,
            init_temperature=self._pa_cfg.yaw_rope_init_temperature,
            include_global=self._include_global,
        )
        self._yaw_rope = yaw_rope
        self._yaw_rope_handle = handle
        if self._stage1_checkpoint:
            yaw_pt = Path(self._stage1_checkpoint) / "panoadapt_yaw_rope.pt"
            if not yaw_pt.exists():
                raise FileNotFoundError(
                    f"stage1_checkpoint={self._stage1_checkpoint} has no "
                    f"panoadapt_yaw_rope.pt; Stage-2 cannot rehydrate yaw_rope."
                )
            state = torch.load(yaw_pt, map_location="cpu")
            ref = next(yaw_rope.parameters())
            yaw_rope.load_state_dict(state)
            # Keep yaw_rope params fp32 (see attach_yaw_rope_hook note re:
            # GradScaler on fp16 models). Move only the device.
            yaw_rope.to(device=ref.device)
            logger.info("Stage-2: yaw_rope initialized from %s", yaw_pt)
        logger.info(
            "PanoAdapt: PanoramaYawRoPE registered on %s (embed_dim=%d, rope_dim=%s)",
            type(target).__name__, yaw_rope.embed_dim, str(self._pa_cfg.yaw_rope_dim),
        )

    def cleanup(self) -> None:
        if self._hook is not None:
            self._hook.remove()

    # -- spatial PE ---------------------------------------------------------

    def _apply_pano_widths(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        config: Any,
    ) -> torch.Tensor:
        image_token_id = config.image_token_id
        spatial_merge = config.vision_config.spatial_merge_size
        overlap = self._pa_cfg.overlap_ratio

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
                n_tokens = t * (h // spatial_merge) * llm_w
                if pos + n_tokens > len(image_positions):
                    break

                view_positions = image_positions[pos : pos + n_tokens]

                # Global view (view_idx=0 when include_global) keeps default positions
                if self._include_global and view_idx == 0:
                    pos += n_tokens
                    continue

                tile_idx = view_idx - (1 if self._include_global else 0)
                # Pull adjacent tiles back by overlap * llm_w each so the width-axis
                # gap shrinks from llm_w to stride * llm_w (= overlapping by `overlap`).
                pano_shift = -int(round(tile_idx * overlap * llm_w))
                position_ids[2, batch_idx, view_positions] += pano_shift

                pos += n_tokens

        return position_ids

    # -- compute_loss override ----------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._pa_cfg.spatial_pe and self._adapter is not None:
            inner = self._unwrap_to_rope_model(model)
            rope_inputs = self._adapter.compute_rope_inputs(inner, inputs)
            position_ids = rope_inputs["position_ids"]
            image_grid_info = inputs.get("image_grid_thw")
            position_ids = self._adapter.modify_position_ids(
                position_ids, inputs["input_ids"], image_grid_info, inner,
            )
            inputs = dict(inputs)
            inputs["position_ids"] = position_ids
            if "rope_deltas" in rope_inputs:
                inputs["rope_deltas"] = rope_inputs["rope_deltas"]

        # -- Clear hook buffer --
        if self._hook is not None:
            self._hook.clear()

        # -- Standard forward + LM loss --
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # -- DenseCL auxiliary loss --
        if self._hook is not None and self._hook.has_features and self._densecl_loss is not None:
            loss = result[0] if isinstance(result, tuple) else result
            densecl_loss = self._compute_densecl(inputs, model)
            if densecl_loss is not None:
                # Defense-in-depth: a degenerate batch can make the overlap
                # (VICReg) term explode (variance-hinge sqrt singularity). With
                # fp16 weights one such step irrecoverably corrupts params
                # (GradScaler protects scaled grads, not fp16 weights). Skip the
                # overlap term for this step if non-finite so the LM update still
                # proceeds and training cannot be permanently poisoned.
                if not torch.isfinite(densecl_loss):
                    logger.warning(
                        "PanoAdapt overlap loss non-finite (%s); skipping overlap "
                        "term this step (lm=%.4f).",
                        densecl_loss.item(), loss.item(),
                    )
                else:
                    total_loss = loss + self._pa_cfg.overlap_loss_weight * densecl_loss
                    logger.debug(
                        "PanoAdapt loss: lm=%.4f densecl=%.4f total=%.4f",
                        loss.item(), densecl_loss.item(), total_loss.item(),
                    )
                    result = (total_loss, result[1]) if isinstance(result, tuple) else total_loss

        return result

    def _compute_densecl(self, inputs: Dict[str, Any], model: torch.nn.Module) -> Optional[torch.Tensor]:
        assert self._hook is not None and self._densecl_loss is not None
        features = self._hook.get_features()
        if features is None:
            return None

        inner = self._unwrap_to_rope_model(model)
        image_grid_thw = inputs.get("image_grid_thw")

        # Qwen2.5-VL: the merger hook fires before visual.forward reapplies the
        # window-attention reverse permutation, so hooked features arrive in
        # window-permuted order. Restore row-major [tile, row, col] order so the
        # overlap/VICReg edge slicing below compares spatially-corresponding
        # patches (no-op for InternVL/Gemma, which are already row-major).
        if self._adapter is not None:
            features = self._adapter.reorder_vision_features(features, inner, image_grid_thw)

        tile_feats: List[torch.Tensor] = []
        grid_h_val: Optional[int] = None
        grid_w_val: Optional[int] = None

        if image_grid_thw is not None:
            spatial_merge = self._adapter.get_spatial_merge_size(inner)
            num_images = image_grid_thw.shape[0]
            start_view = 1 if (self._include_global and num_images > 1) else 0
            offset = 0
            for vi in range(start_view):
                t, h, w = image_grid_thw[vi].tolist()
                offset += int(t * (h // spatial_merge) * (w // spatial_merge))

            for vi in range(start_view, num_images):
                t, h, w = image_grid_thw[vi].tolist()
                lh, lw = h // spatial_merge, w // spatial_merge
                nt = int(t * lh * lw)
                if offset + nt > features.shape[0]:
                    break
                tile_feats.append(features[offset: offset + nt])
                if grid_h_val is None:
                    grid_h_val, grid_w_val = lh, lw
                offset += nt
        else:
            from cora.baseline.panoadapt import _split_consecutive_groups

            image_token_id = self._adapter.get_image_token_id(inner)
            input_ids = inputs["input_ids"]
            is_image = input_ids[0] == image_token_id
            if not is_image.any():
                return None

            image_positions = is_image.nonzero(as_tuple=True)[0]
            views = _split_consecutive_groups(image_positions)
            if len(views) == 0:
                return None

            num_images = len(views)
            start_view = 1 if (self._include_global and num_images > 1) else 0

            if features.ndim == 3:
                # Hook captured [num_images, tokens_per_view, D] (e.g. Gemma3
                # multi_modal_projector outputs [N, 256, text_dim]).
                # Directly index per-view features.
                for vi in range(start_view, min(features.shape[0], num_images)):
                    tile_feats.append(features[vi])  # [tokens_per_view, D]
                if tile_feats:
                    tokens_per_view = features.shape[1]
                    grid_side = int(tokens_per_view ** 0.5)
                    grid_h_val = grid_w_val = grid_side
            else:
                # 2D [total_tokens, D] — uniform slice per view (InternVL style).
                tokens_per_view = features.shape[0] // num_images
                for vi in range(start_view, num_images):
                    offset_v = vi * tokens_per_view
                    if offset_v + tokens_per_view > features.shape[0]:
                        break
                    tile_feats.append(features[offset_v: offset_v + tokens_per_view])

                grid_side = int(tokens_per_view ** 0.5)
                grid_h_val = grid_w_val = grid_side

        if len(tile_feats) < 2 or grid_h_val is None:
            return None

        stacked = torch.stack(tile_feats)
        # Per-batch phys_overlap override when the (hfov, overlap) random aug
        # is on. Batch size is 1 in our recipe, so the first sample's meta
        # describes the whole batch.
        ratio_override: Optional[float] = None
        if self._last_pano_meta_list:
            head = self._last_pano_meta_list[0]
            if head is not None and "phys_overlap" in head:
                ratio_override = float(head["phys_overlap"])
        if ratio_override is not None and not getattr(self, "_logged_first_aug", False):
            logger.info(
                "Random-aug overlap loss: first batch phys_overlap=%.3f "
                "hfov=%.1f n_tiles=%d (override default %.3f)",
                ratio_override,
                float(self._last_pano_meta_list[0].get("hfov_deg", 0.0)),
                int(self._last_pano_meta_list[0].get("n_tiles", 0)),
                float(self._pa_cfg.overlap_ratio or 0.0),
            )
            self._logged_first_aug = True
        return self._densecl_loss(
            stacked,
            num_views=len(tile_feats),
            grid_h=grid_h_val,
            grid_w=grid_w_val,
            overlap_ratio=ratio_override,
        )


class _OrthoLoRATrainer(_PanoAdaptTrainer):
    """PanoAdapt trainer with Ortho-LoRA gradient surgery.

    Instead of summing SFT and SSL losses and back-propagating once,
    this trainer back-propagates each loss separately, then applies
    per-parameter orthogonal projection when the two gradient directions
    conflict (negative cosine similarity). This is based on the Ortho-LoRA
    algorithm (arXiv:2601.09684, Algorithm 1).

    The key idea: when ``g_sft · g_ssl < 0`` for a LoRA parameter, project
    out the conflicting component::

        g_ssl ← g_ssl − (g_sft · g_ssl / ||g_sft||²) · g_sft

    Then combine: ``g_combined = g_sft + g_ssl``.

    This is done independently for each LoRA A and B matrix, preserving
    the bipartite structure of LoRA decompositions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conflict_count: int = 0
        self._total_param_count: int = 0
        self._step_count: int = 0

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Ortho-LoRA training step with per-parameter gradient surgery."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ---- 1. Forward pass (single forward, hook captures vision features) ----
        with self.compute_loss_context_manager():
            # Apply spatial PE if enabled
            if self._pa_cfg.spatial_pe and self._adapter is not None:
                inner = self._unwrap_to_rope_model(model)
                rope_inputs = self._adapter.compute_rope_inputs(inner, inputs)
                position_ids = rope_inputs["position_ids"]
                image_grid_info = inputs.get("image_grid_thw")
                position_ids = self._adapter.modify_position_ids(
                    position_ids, inputs["input_ids"], image_grid_info, inner,
                )
                inputs = dict(inputs)
                inputs["position_ids"] = position_ids
                if "rope_deltas" in rope_inputs:
                    inputs["rope_deltas"] = rope_inputs["rope_deltas"]

            # Clear hook buffer
            if self._hook is not None:
                self._hook.clear()

            # Forward pass — compute SFT loss (standard LM loss)
            result = _SafeTrainer.compute_loss(self, model, inputs)
            sft_loss = result[0] if isinstance(result, tuple) else result

            # Compute SSL loss from hooked features
            ssl_loss: Optional[torch.Tensor] = None
            if self._hook is not None and self._hook.has_features and self._densecl_loss is not None:
                ssl_loss = self._compute_densecl(inputs, model)

        # ---- 2. SFT backward (retain graph if SSL loss exists) ----
        has_ssl = ssl_loss is not None and ssl_loss.requires_grad
        self.accelerator.backward(sft_loss, retain_graph=has_ssl)

        # Collect SFT gradients for LoRA parameters
        sft_grads: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                sft_grads[name] = param.grad.clone()
        model.zero_grad()

        if has_ssl:
            # ---- 3. SSL backward ----
            assert ssl_loss is not None  # guarded by has_ssl
            weighted_ssl = self._pa_cfg.overlap_loss_weight * ssl_loss
            self.accelerator.backward(weighted_ssl)

            # Collect SSL gradients for LoRA parameters
            ssl_grads: Dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if param.grad is not None and "lora" in name.lower():
                    ssl_grads[name] = param.grad.clone()
            model.zero_grad()

            # ---- 4. Ortho-LoRA projection (per A/B matrix) ----
            conflict_count = 0
            total_count = 0
            for name in sft_grads:
                g_sft = sft_grads[name]
                if name in ssl_grads:
                    g_ssl = ssl_grads[name]
                    total_count += 1

                    g_sft_flat = g_sft.flatten()
                    g_ssl_flat = g_ssl.flatten()
                    dot = (g_sft_flat * g_ssl_flat).sum()

                    if dot < 0:
                        # Conflict: project SSL gradient to be orthogonal to SFT
                        conflict_count += 1
                        norm_sq = g_sft_flat.norm().square() + 1e-8
                        g_ssl_flat = g_ssl_flat - (dot / norm_sq) * g_sft_flat
                        g_ssl = g_ssl_flat.view_as(g_ssl)

                    combined = g_sft + g_ssl
                else:
                    combined = g_sft

                # Set combined gradient on the parameter
                for n2, p2 in model.named_parameters():
                    if n2 == name:
                        p2.grad = combined
                        break

            # Also set SSL-only gradients (params that have SSL grad but not SFT)
            for name in ssl_grads:
                if name not in sft_grads:
                    for n2, p2 in model.named_parameters():
                        if n2 == name:
                            p2.grad = ssl_grads[name]
                            break

            self._conflict_count += conflict_count
            self._total_param_count += total_count
            self._step_count += 1

            # Log conflict rate periodically
            if self._step_count % 50 == 0 and self._total_param_count > 0:
                rate = self._conflict_count / max(self._total_param_count, 1)
                logger.info(
                    "OrthoLoRA step %d: conflict_rate=%.3f (%d/%d) ",
                    self._step_count, rate,
                    self._conflict_count, self._total_param_count,
                )
                self._conflict_count = 0
                self._total_param_count = 0

            total_loss = sft_loss + weighted_ssl
        else:
            # No SSL loss — just use SFT gradients
            for name in sft_grads:
                for n2, p2 in model.named_parameters():
                    if n2 == name:
                        p2.grad = sft_grads[name]
                        break
            total_loss = sft_loss

        return total_loss.detach() / self.args.gradient_accumulation_steps

class BaselineTrainer:
    """LoRA finetuning driver for commercial VLMs.

    Uses HF Trainer with PEFT LoRA, **not** PyTorch Lightning.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir) / config.model.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API -------------------------------------------------------

    def train(self) -> str:
        """Run LoRA finetuning. Returns path to saved adapter."""
        cfg = self.config
        set_seed(cfg.training.seed)

        # Load model
        model, processor, tokenizer = BaselineModelRegistry.load_model(cfg.model)

        # Stage-2 entrypoint: merge a Stage-1 vision SSL LoRA into the base
        # weights BEFORE attaching the Stage-2 LM LoRA. After merge_and_unload
        # the base model presents as a plain (non-PEFT) backbone with the
        # Stage-1 panorama-aware vision_tower baked in, so the next LoRA
        # attachment treats it as a regular base model.
        if cfg.stage1_checkpoint:
            from peft import PeftModel

            ck = Path(cfg.stage1_checkpoint)
            if not (ck / "adapter_model.safetensors").exists():
                raise FileNotFoundError(
                    f"stage1_checkpoint={ck} has no adapter_model.safetensors"
                )
            logger.info("Stage-2: loading + merging Stage-1 vision LoRA from %s", ck)
            model = PeftModel.from_pretrained(model, str(ck))
            model = model.merge_and_unload()
            logger.info("Stage-2: Stage-1 vision LoRA merged into base weights")

        # Optionally split Qwen2.5-VL's fused vision attn.qkv into separate
        # q/k/v Linears so LoRA adapts them independently (mirrors InternVL/
        # Gemma). Must run before get_peft_model. No-op for non-Qwen models.
        if cfg.lora.split_vision_qkv:
            n_split = _split_qwen_vision_qkv(model)
            if n_split == 0:
                logger.warning("split_vision_qkv=True but no Qwen vision attn.qkv found (non-Qwen model?)")

        # Determine LoRA target modules. ``lora.lm_only=True`` overrides the
        # list to a single fullmatch regex anchored on ``language_model.``,
        # which PEFT applies via ``re.fullmatch`` — necessary because the
        # default suffix list (``q_proj/k_proj/v_proj/o_proj``) would also
        # match vision_tower attention modules and double-adapt them.
        targets: Any
        if cfg.lora.lm_only:
            targets = r".*language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj)"
        else:
            targets = (
                cfg.lora.target_modules
                or cfg.model.lora_target_modules
                or BaselineModelRegistry.get_default_lora_targets(cfg.model.model_type)
            )

        # Determine task type based on model
        is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}
        task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM

        # Apply LoRA
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=targets,
            task_type=task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if cfg.training.gradient_checkpointing:
            model.enable_input_require_grads()

        # Build datasets
        erp_resize_tuple: Optional[tuple] = None
        if cfg.model.erp_resize_width:
            w_px = int(cfg.model.erp_resize_width)
            h_px = int(cfg.model.erp_resize_height) if cfg.model.erp_resize_height else w_px // 2
            erp_resize_tuple = (w_px, h_px)
            logger.info("ERP pre-resize active: %dx%d (single-image path only)", w_px, h_px)
        ds_kwargs: Dict[str, Any] = dict(
            processor=processor,
            tokenizer=tokenizer,
            model_type=cfg.model.model_type,
            image_column=cfg.data.image_column,
            instruction_column=cfg.data.instruction_column,
            response_column=cfg.data.response_column,
            pano_view_config=cfg.effective_pano_view,
            erp_resize=erp_resize_tuple,
        )
        train_ds = VLMDataset(
            csv_path=cfg.data_train_csv,
            max_samples=cfg.data.max_train_samples,
            augment=True,
            **ds_kwargs,
        )
        pv = cfg.effective_pano_view
        if pv is not None:
            logger.info(
                "Multi-view enabled: strategy=%s hfov=%.0f° overlap=%.1f pitch=[%.0f,%.0f] include_global=%s → multi-image mode",
                pv.strategy, pv.hfov_deg, pv.overlap,
                pv.pitch_min, pv.pitch_max, pv.include_global,
            )
        eval_ds: Optional[VLMDataset] = None
        if cfg.data_val_csv:
            eval_ds = VLMDataset(
                csv_path=cfg.data_val_csv,
                max_samples=cfg.data.max_eval_samples,
                **ds_kwargs,
            )

        # Collate function
        if is_seq2seq:
            collate_fn = _collate_seq2seq(tokenizer)
        else:
            collate_fn = _collate_causal(
                processor,
                tokenizer,
                max_length=cfg.training.max_length,
                image_token_id=_resolve_image_token_id(model, processor),
            )

        # wandb
        report_to: List[str] = []
        if cfg.wandb_enabled:
            report_to.append("wandb")
            if cfg.wandb_project:
                os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

        run_name = f"{cfg.experiment_name}__{cfg.model.name}"

        # Auto batch-size
        batch_size = cfg.training.batch_size
        if batch_size == -1:
            batch_size = autobatch(
                model=model,
                dataset=train_ds,
                collate_fn=collate_fn,
                fraction=0.85,
                default_batch_size=1,
            )
            logger.info("AutoBatch resolved batch_size = %d", batch_size)

        # Training args
        prec = _resolve_precision(cfg.training.mixed_precision)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=cfg.training.num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_ratio=cfg.training.warmup_ratio,
            max_grad_norm=cfg.training.max_grad_norm,
            fp16=prec["fp16"],
            bf16=prec["bf16"],
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            # Reentrant checkpointing silently drops gradients on layers whose
            # only inputs are non-grad tensors (frozen vision-encoder embeddings
            # feed the encoder when only LoRA params are trainable). Switch to
            # the non-reentrant implementation so vision-tower LoRA actually
            # receives gradients during backward.
            gradient_checkpointing_kwargs={"use_reentrant": False}
                if cfg.training.gradient_checkpointing else None,
            logging_steps=cfg.training.logging_steps,
            eval_strategy=cfg.training.eval_strategy if eval_ds else "no",
            save_strategy=cfg.training.save_strategy,
            save_total_limit=cfg.training.save_total_limit,
            report_to=report_to,
            run_name=run_name,
            seed=cfg.training.seed,
            remove_unused_columns=False,
            dataloader_num_workers=cfg.training.dataloader_num_workers,
            dataloader_pin_memory=cfg.training.dataloader_pin_memory,
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
        )

        pa_cfg = cfg.panoadapt
        if pa_cfg is not None and (pa_cfg.spatial_pe or pa_cfg.overlap_loss):
            trainer_cls = _OrthoLoRATrainer if pa_cfg.ortho_lora else _PanoAdaptTrainer
            trainer = trainer_cls(
                panoadapt_config=pa_cfg,
                pano_view_config=cfg.effective_pano_view,
                stage1_checkpoint=cfg.stage1_checkpoint,
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )
            logger.info(
                "PanoAdapt enabled: spatial_pe=%s overlap_loss=%s (weight=%.3f) ortho_lora=%s",
                pa_cfg.spatial_pe, pa_cfg.overlap_loss, pa_cfg.overlap_loss_weight, pa_cfg.ortho_lora,
            )
        else:
            trainer = _SafeTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )

        # Decoupled vision LR (consumed by _SafeTrainer.create_optimizer); None
        # ⇒ stock single-group optimizer.
        trainer._vision_lr = cfg.training.vision_lr

        trainer.train()

        # Save LoRA adapter
        save_path = str(self.output_dir / "lora_adapter")
        model.save_pretrained(save_path)
        trainer.save_model(str(self.output_dir / "final"))

        # Save PanoramaYawRoPE state separately. PEFT's save_pretrained only
        # writes the LoRA adapter + modules_to_save; the yaw_rope module was
        # add_module'd post-PEFT, so without this its trained ``log_inv_freq``
        # would be silently discarded.
        if isinstance(trainer, _PanoAdaptTrainer) and trainer._yaw_rope is not None:
            yaw_pt = self.output_dir / "panoadapt_yaw_rope.pt"
            torch.save(trainer._yaw_rope.state_dict(), yaw_pt)
            # Mirror under lora_adapter/ too so any consumer that points at
            # the adapter dir finds it without extra path math.
            torch.save(
                trainer._yaw_rope.state_dict(),
                self.output_dir / "lora_adapter" / "panoadapt_yaw_rope.pt",
            )
            logger.info("Saved PanoramaYawRoPE state_dict to %s", yaw_pt)

        logger.info("Training complete. Adapter saved to %s", save_path)

        # Cleanup
        if isinstance(trainer, _PanoAdaptTrainer):
            trainer.cleanup()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return save_path

    def evaluate(
        self,
        test_csv: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a trained model on *test_csv*. Returns metrics dict.

        Loads the LoRA adapter from ``self.output_dir / "lora_adapter"``
        and runs generation over the test set, computing text metrics.
        """
        cfg = self.config
        eval_dir = Path(output_dir) if output_dir else self.output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        model, processor, tokenizer = BaselineModelRegistry.load_model(cfg.model)

        # Load LoRA adapter
        adapter_path = self.output_dir / "lora_adapter"
        if adapter_path.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            logger.info("Loaded LoRA adapter from %s", adapter_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # PanoramaYawRoPE rehydration. The yaw_rope module was add_module'd
        # to the model at train time; PEFT doesn't serialize it, so we
        # rebuild + load state_dict + re-register the forward hook here.
        # Without this, generation runs the LM on un-rotated projector
        # output even though training pushed the LoRA to expect rotation
        # → train/eval mismatch.
        yaw_rope_for_eval = None
        pa_cfg = cfg.panoadapt
        if pa_cfg is not None and getattr(pa_cfg, "yaw_rope_enabled", False):
            yaw_pt = self.output_dir / "panoadapt_yaw_rope.pt"
            if not yaw_pt.exists():
                yaw_pt_alt = adapter_path / "panoadapt_yaw_rope.pt"
                if yaw_pt_alt.exists():
                    yaw_pt = yaw_pt_alt
            if not yaw_pt.exists():
                raise FileNotFoundError(
                    f"yaw_rope_enabled=True but {yaw_pt} (and adapter mirror) missing — "
                    "train job did not save the yaw RoPE state."
                )
            from cora.model.positional import attach_yaw_rope_hook

            # Resolve the same vision projector the training hook used.
            target = None
            for name, mod in model.named_modules():
                lname = name.lower()
                if "merger" in lname or "multi_modal_projector" in lname:
                    target = mod
                    break
            if target is None:
                raise RuntimeError("eval yaw_rope: vision projector not found")

            include_global = True
            if cfg.effective_pano_view is not None:
                include_global = bool(cfg.effective_pano_view.include_global)

            state_dict = torch.load(yaw_pt, map_location=device)
            yaw_rope_for_eval, _handle = attach_yaw_rope_hook(
                model=model,
                target_module=target,
                rope_dim=pa_cfg.yaw_rope_dim,
                init_temperature=pa_cfg.yaw_rope_init_temperature,
                include_global=include_global,
                state_dict=state_dict,
            )
            yaw_rope_for_eval.eval()
            logger.info("Eval: PanoramaYawRoPE rehydrated from %s", yaw_pt)

        # Read test data
        df = pd.read_csv(test_csv)
        image_paths: List[str] = []
        queries: List[str] = []
        predictions: List[str] = []
        references: List[str] = []

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        tokenizer.padding_side = "left"

        is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}

        total = len(df)
        skipped = 0

        with torch.inference_mode():
            pbar = tqdm(range(total), desc="Evaluating", unit="sample", dynamic_ncols=True)
            for i in pbar:
                row = df.iloc[i]
                img_path = row.get(cfg.data.image_column)
                if img_path is None or not Path(str(img_path)).is_file():
                    skipped += 1
                    pbar.set_postfix(skipped=skipped)
                    continue

                erp_image = Image.open(str(img_path)).convert("RGB")
                prompt = str(row.get(cfg.data.instruction_column, "Describe the image."))
                reference = str(row.get(cfg.data.response_column, ""))

                gen_pack = build_generation_inputs_with_meta(
                    cfg, processor, erp_image, prompt,
                )
                inputs = gen_pack["inputs"]
                pano_meta_for_yaw = gen_pack["meta"]

                target_dtype = next(model.parameters()).dtype
                inputs = {
                    k: (
                        v.to(device=device, dtype=target_dtype)
                        if isinstance(v, torch.Tensor) and v.is_floating_point()
                        else v.to(device) if isinstance(v, torch.Tensor) else v
                    )
                    for k, v in inputs.items()
                }

                prompt_len = inputs.get("input_ids", torch.tensor([])).shape[-1]

                if (
                    yaw_rope_for_eval is not None
                    and pano_meta_for_yaw.get("pe_mode") == "multiview"
                ):
                    yaw_rope_for_eval.set_meta([pano_meta_for_yaw])
                try:
                    outputs = model.generate(**inputs, **gen_kwargs)
                finally:
                    if yaw_rope_for_eval is not None:
                        yaw_rope_for_eval.clear_meta()
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                out_len = outputs[0].shape[0]

                # Decode full output first, then strip prompt text
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                if is_seq2seq or out_len <= prompt_len:
                    # Seq2seq models or models that return only new tokens
                    pred = full_text
                else:
                    # Strip prompt tokens, then decode
                    generated = outputs[0][prompt_len:]
                    pred = tokenizer.decode(generated, skip_special_tokens=True).strip()

                # Fallback: if sliced decode is empty but full decode has content.
                # - seq2seq/new-tokens-only models: full_text IS the response → use directly.
                # - causal models: full_text includes the prompt; try to extract only the
                #   model/assistant turn by splitting on the role separator.
                if not pred and full_text:
                    if is_seq2seq or out_len <= prompt_len:
                        # full_text is just the new tokens (seq2seq or new-token-only causal)
                        pred = full_text
                    else:
                        # Causal model: full_text = prompt_text + response_text (special tokens stripped).
                        # Try common chat-template model-turn separators to extract just the response.
                        _turn_markers = [
                            "\nmodel\n",       # Gemma3
                            "\nassistant\n",   # generic
                            "\nASSISTANT\n",  # LLaVA-style
                            "\n[/INST]\n",    # Llama-2
                            "\n[/INST]",      # Llama-2 variant
                        ]
                        for _marker in _turn_markers:
                            if _marker in full_text:
                                pred = full_text.split(_marker, 1)[-1].strip()
                                break
                        # If no marker matched, keep pred empty rather than
                        # returning the prompt as the prediction.

                # Log first few samples for debugging
                if i < 3:
                    logger.info(
                        "  [DEBUG] sample=%d prompt_len=%d out_len=%d pred_len=%d pred=%r",
                        i, prompt_len, out_len, len(pred), pred[:80],
                    )

                image_paths.append(str(img_path))
                queries.append(prompt)
                predictions.append(pred)
                references.append(reference)

                # Update progress bar with latest prediction info
                pred_short = pred[:40] + "..." if len(pred) > 40 else pred
                pbar.set_postfix(
                    done=len(predictions),
                    skipped=skipped,
                    pred=pred_short,
                )

        # Save predictions FIRST (before metric computation which can crash/deadlock)
        records = [
            {"image_path": ip, "query": q, "prediction": p, "reference": r}
            for ip, q, p, r in zip(image_paths, queries, predictions, references)
        ]
        pred_path = eval_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        pred_csv_path = eval_dir / "predictions.csv"
        pd.DataFrame(records).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
        logger.info("Predictions saved to %s", pred_path)

        # Compute metrics
        metrics = _compute_basic_metrics(predictions, references)

        try:
            from cora.evaluation.metrics import CORAEvaluator
            evaluator = CORAEvaluator()
            coco_metrics = evaluator.evaluate(predictions, references)
            if coco_metrics:
                metrics.update(coco_metrics)
                CORAEvaluator.print_summary(coco_metrics)
        except Exception as exc:
            logger.warning("COCO metrics failed: %s", exc)


        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation complete. Predictions: %s  Metrics: %s", pred_csv_path, metrics_path)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def _load_inference_model(self, adapter_dir: Optional[str] = None):
        """Load base model + trained LoRA adapter (+ yaw RoPE) for inference.

        Returns ``(model, processor, tokenizer, yaw_rope, device)``.  Mirrors the
        model-loading half of :meth:`evaluate` so single-image inference uses the
        exact same weights and view geometry as test-set evaluation.
        """
        cfg = self.config
        model, processor, tokenizer = BaselineModelRegistry.load_model(cfg.model)

        adapter_path = Path(adapter_dir) if adapter_dir else self.output_dir / "lora_adapter"
        if adapter_path.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            logger.info("Loaded LoRA adapter from %s", adapter_path)
        else:
            logger.warning("No LoRA adapter at %s — running base model only", adapter_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        yaw_rope = None
        pa_cfg = cfg.panoadapt
        if pa_cfg is not None and getattr(pa_cfg, "yaw_rope_enabled", False):
            run_dir = adapter_path.parent
            yaw_pt = run_dir / "panoadapt_yaw_rope.pt"
            if not yaw_pt.exists():
                yaw_pt_alt = adapter_path / "panoadapt_yaw_rope.pt"
                if yaw_pt_alt.exists():
                    yaw_pt = yaw_pt_alt
            if not yaw_pt.exists():
                raise FileNotFoundError(
                    f"yaw_rope_enabled=True but {yaw_pt} (and adapter mirror) missing — "
                    "train job did not save the yaw RoPE state."
                )
            from cora.model.positional import attach_yaw_rope_hook

            target = None
            for name, mod in model.named_modules():
                lname = name.lower()
                if "merger" in lname or "multi_modal_projector" in lname:
                    target = mod
                    break
            if target is None:
                raise RuntimeError("inference yaw_rope: vision projector not found")

            include_global = True
            if cfg.effective_pano_view is not None:
                include_global = bool(cfg.effective_pano_view.include_global)

            state_dict = torch.load(yaw_pt, map_location=device)
            yaw_rope, _handle = attach_yaw_rope_hook(
                model=model,
                target_module=target,
                rope_dim=pa_cfg.yaw_rope_dim,
                init_temperature=pa_cfg.yaw_rope_init_temperature,
                include_global=include_global,
                state_dict=state_dict,
            )
            yaw_rope.eval()
            logger.info("Inference: PanoramaYawRoPE rehydrated from %s", yaw_pt)

        return model, processor, tokenizer, yaw_rope, device

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Any,
        prompt: Optional[str] = None,
        adapter_dir: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Caption a single panorama with the trained LoRA adapter.

        ``image`` is a path or a PIL image.  The model + adapter are loaded once
        and cached on the instance, so repeated calls only pay for generation.
        Uses the same view construction, yaw-RoPE rehydration and greedy decoding
        as :meth:`evaluate`, so a single-image result matches the test-set run.
        """
        cfg = self.config
        prompt = prompt or "Describe the image."

        state = getattr(self, "_infer_state", None)
        if state is None:
            state = self._load_inference_model(adapter_dir=adapter_dir)
            self._infer_state = state
        model, processor, tokenizer, yaw_rope, device = state

        erp_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if generation_config:
            gen_kwargs.update(generation_config)
        tokenizer.padding_side = "left"
        is_seq2seq = cfg.model.model_type.lower() in {"blip2", "blip-2"}

        gen_pack = build_generation_inputs_with_meta(cfg, processor, erp_image, prompt)
        inputs = gen_pack["inputs"]
        pano_meta_for_yaw = gen_pack["meta"]

        target_dtype = next(model.parameters()).dtype
        inputs = {
            k: (
                v.to(device=device, dtype=target_dtype)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.to(device) if isinstance(v, torch.Tensor) else v
            )
            for k, v in inputs.items()
        }
        prompt_len = inputs.get("input_ids", torch.tensor([])).shape[-1]

        if yaw_rope is not None and pano_meta_for_yaw.get("pe_mode") == "multiview":
            yaw_rope.set_meta([pano_meta_for_yaw])
        try:
            outputs = model.generate(**inputs, **gen_kwargs)
        finally:
            if yaw_rope is not None:
                yaw_rope.clear_meta()
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        return _decode_generation(tokenizer, outputs, prompt_len, is_seq2seq)


# ---------------------------------------------------------------------------
# Generation decoding (shared by evaluate() and single-image inference)
# ---------------------------------------------------------------------------


def _decode_generation(tokenizer, outputs, prompt_len: int, is_seq2seq: bool) -> str:
    """Decode a ``generate()`` sequence into the model's answer, stripping the prompt.

    Mirrors the decode logic inside :meth:`BaselineTrainer.evaluate` so that
    single-image CLI output matches the corresponding test-set prediction.
    ``outputs`` is the raw tensor returned by ``model.generate`` (one sequence
    per row); only the first row is decoded.
    """
    out_len = outputs[0].shape[0]
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if is_seq2seq or out_len <= prompt_len:
        pred = full_text
    else:
        generated = outputs[0][prompt_len:]
        pred = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if not pred and full_text:
        if is_seq2seq or out_len <= prompt_len:
            pred = full_text
        else:
            _turn_markers = [
                "\nmodel\n",       # Gemma3
                "\nassistant\n",   # generic
                "\nASSISTANT\n",  # LLaVA-style
                "\n[/INST]\n",    # Llama-2
                "\n[/INST]",      # Llama-2 variant
            ]
            for _marker in _turn_markers:
                if _marker in full_text:
                    pred = full_text.split(_marker, 1)[-1].strip()
                    break

    return pred


# ---------------------------------------------------------------------------
# Basic text metrics (no heavy optional deps required)
# ---------------------------------------------------------------------------


def _compute_basic_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """Compute lightweight text metrics without requiring nltk / rouge-score."""
    import numpy as np

    paired = [
        (p.strip(), r.strip())
        for p, r in zip(predictions, references)
        if r is not None and str(r).strip()
    ]
    if not paired:
        return {"samples": 0}

    preds = [p for p, _ in paired]
    refs = [r for _, r in paired]

    metrics: Dict[str, Any] = {
        "samples": len(paired),
        "exact_match": sum(1 for p, r in paired if p == r) / len(paired),
        "avg_pred_tokens": float(np.mean([len(p.split()) for p in preds])),
        "avg_ref_tokens": float(np.mean([len(r.split()) for r in refs])),
    }

    # Optional: BLEU-4
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        smooth = SmoothingFunction().method1
        ref_tok = [[r.split()] for r in refs]
        pred_tok = [p.split() for p in preds]
        metrics["bleu4"] = float(
            corpus_bleu(ref_tok, pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        )
    except Exception:
        pass

    # Optional: ROUGE-L
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds) if r and p]
        if scores:
            metrics["rougeL"] = float(np.mean(scores))
    except Exception:
        pass

    return metrics
