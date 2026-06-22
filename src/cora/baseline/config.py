"""Baseline LoRA finetuning configuration schema.

Pydantic-based configuration for commercial VLM LoRA finetuning,
ported from legacy/root_scripts/vlm_finetune_and_eval.py dataclasses.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class BaselineModelConfig(BaseModel):
    """Configuration for a single commercial VLM."""

    name: str = "qwen2.5-vl-7b"
    hf_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor_id: Optional[str] = None  # defaults to hf_model_id
    model_type: str = "qwen_vl"  # qwen_vl, llava, llava_onevision, blip2, gemma3
    dtype: str = "float16"
    lora_target_modules: Optional[List[str]] = None
    image_size: int = 224
    dynamic_resolution: bool = False
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    # Pre-resize ERP to a fixed target before the processor (used by ERP-RoPE
    # width-matched single-image experiments). Must be multiples of the
    # backbone's patch×merge size (28 for Qwen2.5-VL). When set, callers
    # should also pin min_pixels=max_pixels=W*H to skip Qwen smart_resize.
    erp_resize_width: Optional[int] = None
    erp_resize_height: Optional[int] = None
    # HuggingFace attn_implementation passed to from_pretrained. Default None
    # means the model's default (eager) — kept so existing baselines are bit-
    # identical. Stage 1 sets "sdpa" because the vision encoder's
    # O(N²) attention matrix at large bs*max_v blows past 24 GB at 448 px.
    attn_implementation: Optional[str] = None


class BaselineLoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # overrides model default
    # When True, target_modules is replaced by a fullmatch regex that anchors
    # on ``language_model.`` so LoRA never bleeds into vision_tower. Used by
    # Stage 2 (LM-only training on top of a merged Stage-1 vision LoRA).
    lm_only: bool = False
    # When True, Qwen2.5-VL's fused vision attention ``attn.qkv`` (one Linear
    # dim->3*dim) is split into separate q_proj/k_proj/v_proj Linears BEFORE LoRA
    # so PEFT can adapt q/k/v independently (mirrors InternVL/Gemma split vision
    # attention). Numerically identical to the fused matrix at init. No-op for
    # non-Qwen models. Pair with target_modules including q_proj/k_proj/v_proj.
    split_vision_qkv: bool = False


class BaselineTrainingConfig(BaseModel):
    """Training hyperparameters for HF Trainer."""

    num_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    # When set, trainable vision-tower params (name contains ``.visual.``) form
    # their own AdamW group at this LR while LM LoRA / yaw_rope keep
    # ``learning_rate``. None ⇒ single global LR (HF default). Lets a heavily
    # pretrained ViT (Qwen2.5-VL) adapt gently without over-writing features.
    vision_lr: Optional[float] = None
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = False
    mixed_precision: Optional[str] = None  # fp16, bf16, None
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    eval_strategy: str = "no"
    max_length: Optional[int] = 512  # Max input sequence length (truncation). None=no limit
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = False


class BaselineDataConfig(BaseModel):
    """Column mapping for CSV datasets."""

    image_column: str = "url"
    instruction_column: str = "instruction"
    response_column: str = "response"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class PanoViewConfig(BaseModel):
    """Multi-view panoramic input configuration.

    Strategies:
      - ``"anyres_e2p"``: 1 global resized + N yaw tiles (default: 1+8=9 views)
      - ``"cubemap"``: 4 side faces at 90° FOV (+ optional global = 4 or 5 views)
      - ``"pinhole"``: N yaw tiles only, no global context (default: 8 views)
    """

    strategy: str = "anyres_e2p"
    include_global: bool = True
    hfov_deg: float = 45.0
    overlap: float = 0.0
    closed_loop_yaw: bool = True
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    base_size: int = 336
    tile_render_size: int = 672
    vit_size: Optional[int] = None
    # Random augmentation (Stage 1). When set, each TRAIN sample draws
    # (hfov_deg, overlap) uniformly from the range; eval still uses the static
    # hfov_deg / overlap above. Tuple form [lo, hi]. n_tiles consequently varies
    # per sample → batch_size must stay 1 (collate cannot pad ragged n_tiles).
    hfov_deg_range: Optional[List[float]] = None
    overlap_range: Optional[List[float]] = None


# Backward-compatible alias
AnyresE2PConfig = PanoViewConfig


class PanoAdaptConfig(BaseModel):
    """PanoAdapt: spatial PE (Layer 2) + overlap loss (Layer 3) for panoramic VLMs."""

    model_type: str = "qwen_vl"  # qwen_vl, qwen2_vl, internvl, gemma3
    rope_type: str = "3d"  # 3d (Qwen M-RoPE), 1d (InternVL/Gemma3 PanoRoPE-1D)
    spatial_pe: bool = False
    overlap_loss: bool = False
    overlap_loss_type: str = "densecl"  # densecl, vicreg_batchwise, vicreg_pairwise
    overlap_loss_weight: float = 0.0
    overlap_loss_temperature: float = 0.07
    # None ⇒ auto-derive from the realized yaw-tile geometry (single source of
    # truth; see BaselineConfig.resolve_and_check_overlap). An explicit value is
    # honored but cross-checked against the geometry and warned/raised on
    # mismatch. Default is intentionally NOT a constant so it can never silently
    # contradict PanoViewConfig.overlap again.
    overlap_ratio: Optional[float] = None
    vicreg_sim_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0
    ortho_lora: bool = False  # Enable Ortho-LoRA gradient surgery (project conflicting LoRA gradients)
    # Stage 3 — learnable yaw RoPE applied to vision tokens right after the
    # vision projector (multi_modal_projector / merger). Rotates rope_dim
    # channels by the token's true yaw (from pano_meta), trains the RoPE
    # frequency table. Independent of spatial_pe (LM-side position_ids shift).
    yaw_rope_enabled: bool = False
    yaw_rope_dim: Optional[int] = None  # None ⇒ embed_dim // 2
    yaw_rope_init_temperature: float = 10000.0


class BaselineConfig(BaseModel):
    """Top-level configuration for baseline LoRA finetuning."""

    experiment_name: str = "baseline_finetune"
    output_dir: str = "runs/baseline"
    model: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    lora: BaselineLoRAConfig = Field(default_factory=BaselineLoRAConfig)
    training: BaselineTrainingConfig = Field(default_factory=BaselineTrainingConfig)
    data: BaselineDataConfig = Field(default_factory=BaselineDataConfig)
    pano_view: Optional[PanoViewConfig] = None
    anyres_e2p: Optional[PanoViewConfig] = None
    panoadapt: Optional[PanoAdaptConfig] = None
    max_new_tokens: int = 128
    data_train_csv: str = "data/quic360/train.csv"
    data_val_csv: Optional[str] = None
    data_test_csv: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_enabled: bool = False
    # Path to a Stage-1 vision SSL pretrain checkpoint dir containing
    # ``adapter_model.safetensors`` + ``panoadapt_yaw_rope.pt``. When set,
    # BaselineTrainer:
    #   1. loads the PEFT vision LoRA from this dir and merges it into the
    #      base model's weights (``merge_and_unload``), so vision_tower is
    #      now Stage-1 pretrained but presents as a plain backbone for
    #      Stage-2 LoRA attachment;
    #   2. initializes the PanoramaYawRoPE module from the saved
    #      ``panoadapt_yaw_rope.pt`` (otherwise the module starts at the
    #      log-spaced RoPE init each run).
    # Leaving this None preserves the original (fresh-init) trackB/ablD path.
    stage1_checkpoint: Optional[str] = None

    @property
    def effective_pano_view(self) -> Optional[PanoViewConfig]:
        """Return pano_view config, falling back to anyres_e2p for backward compat."""
        return self.pano_view or self.anyres_e2p

    # Tolerance (in overlap-fraction units) below which an explicitly-set
    # overlap_ratio is considered consistent with the realized geometry.
    _OVERLAP_TOL: float = 0.05

    @model_validator(mode="after")
    def resolve_and_check_overlap(self) -> "BaselineConfig":
        """Single source of truth for the overlap loss's ``overlap_ratio``.

        The overlap loss slices ``k = int(grid_w * overlap_ratio)`` feature
        columns from adjacent tiles and forces them equal. Those columns only
        correspond to the *same* physical scene when that fraction equals the
        fraction the renderer actually overlaps adjacent tiles — which the
        closed-loop tile-count quantization makes differ from the configured
        ``PanoViewConfig.overlap``. Historically the two knobs had contradictory
        independent defaults (overlap=0.0 vs overlap_ratio=0.5) and nothing tied
        them, so omitting both silently trained a mis-specified objective.

        Behavior:
          * ``overlap_ratio is None``  → derive it from the realized geometry
            (R1/R3). The config becomes self-consistent by construction.
          * explicit ``overlap_ratio`` → honored for reproducibility, but
            cross-checked: a mismatch beyond ``_OVERLAP_TOL`` warns loudly, or
            raises if ``CORA_STRICT_OVERLAP=1`` (R2).
          * always logs the resolved geometry + MATCH/MISMATCH verdict (R4).
        """
        pa = self.panoadapt
        if pa is None:
            return self
        # Resolve whenever any pano machinery is active: the PE width-shift also
        # reads overlap_ratio, so it must be a concrete float even when only
        # spatial_pe is on. The MISMATCH warn/raise, however, only matters when
        # the overlap *loss* is active (that's what mis-pairs regions).
        if not (pa.overlap_loss or pa.spatial_pe):
            if pa.overlap_ratio is None:
                pa.overlap_ratio = 0.0
            return self

        pv = self.effective_pano_view
        strategy = (pv.strategy.lower() if pv else "anyres_e2p")

        # Only yaw-tiled strategies have a well-defined adjacent-tile overlap.
        if pv is None or strategy in {"anyres_e2p", "pinhole"}:
            from cora.processors.anyres_e2p import resolve_yaw_geometry

            hfov = pv.hfov_deg if pv else 45.0
            render_overlap = pv.overlap if pv else 0.0
            closed_loop = pv.closed_loop_yaw if pv else True
            n_tiles, phys_overlap = resolve_yaw_geometry(hfov, render_overlap, closed_loop)
        elif strategy == "cubemap":
            # 4×90° faces share only an edge → no horizontal overlap region.
            n_tiles, phys_overlap = 4, 0.0
        else:
            n_tiles, phys_overlap = 0, 0.0

        configured = pa.overlap_ratio
        include_global = pv.include_global if pv else True

        if configured is None:
            pa.overlap_ratio = round(phys_overlap, 6)
            verdict = "DERIVED"
        else:
            diff = abs(float(configured) - phys_overlap)
            if diff <= self._OVERLAP_TOL:
                verdict = "MATCH"
            elif not pa.overlap_loss:
                # spatial_pe-only: overlap_ratio just shifts PE columns; a
                # mismatch is not a scientific mis-pairing, only note it.
                verdict = "MISMATCH(pe-only)"
            else:
                verdict = "MISMATCH"
                msg = (
                    "PanoAdapt overlap MISMATCH: overlap_ratio=%.4f but the "
                    "renderer (strategy=%s hfov=%.1f overlap=%.4f closed_loop=%s) "
                    "physically overlaps adjacent tiles by %.4f over %d tiles. "
                    "The overlap loss is comparing NON-corresponding regions. "
                    "Set overlap_ratio to ~%.4f (or omit it to auto-derive)."
                    % (
                        float(configured), strategy,
                        (pv.hfov_deg if pv else 45.0),
                        (pv.overlap if pv else 0.0),
                        (pv.closed_loop_yaw if pv else True),
                        phys_overlap, n_tiles, phys_overlap,
                    )
                )
                if os.environ.get("CORA_STRICT_OVERLAP", "").strip() in {"1", "true", "True"}:
                    raise ValueError(msg)
                logger.warning("%s (set CORA_STRICT_OVERLAP=1 to make this fatal)", msg)

        logger.info(
            "PanoAdapt overlap resolved [%s]: strategy=%s hfov=%.1f "
            "render_overlap=%.4f closed_loop=%s → n_tiles=%d phys_overlap=%.4f "
            "| loss overlap_ratio=%.4f include_global=%s",
            verdict, strategy, (pv.hfov_deg if pv else 45.0),
            (pv.overlap if pv else 0.0), (pv.closed_loop_yaw if pv else True),
            n_tiles, phys_overlap, pa.overlap_ratio, include_global,
        )
        if include_global:
            logger.info(
                "PanoAdapt note: include_global=True — view 0 is a global "
                "image; the cyclic roll in the overlap loss/PE pairs it with a "
                "tile at the seam. Verify this is intended for overlap SSL."
            )
        return self
