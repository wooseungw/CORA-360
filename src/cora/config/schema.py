from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

RESAMPLER_CONFIGS = {
    "mlp": {
        "hidden_dim": 1536,
        "depth": 3,
        "use_ln": True,
        "pool_type": "avg",
    },
    "bimamba": {
        "hidden_dim": 1024,
        "num_layers": 4,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2.0,
        "norm_first": True,
        "dropout": 0.05,
    },
    "qformer": {
        "num_query_tokens": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
    },
    "perceiver": {
        "num_latents": 64,
        "heads": 8,
        "depth": 6,
    },
    "spatial_pool": {
        "pool_size": 4,
        "depth": 2,
    },
    "masked_drop": {
        "mask_ratio": 0.5,
        "depth": 2,
    },
    "c_abstractor": {
        "hidden_dim": 1024,
        "depth": 3,
        "num_queries": 0,
        "kernel_size": 7,
        "expand": 4,
        "se_reduction": 4,
        "mlp_depth": 2,
        "drop_path": 0.0,
    },
}


class ImageProcessingConfig(BaseModel):
    crop_strategy: Literal[
        "sliding_window", "e2p", "cubemap",
        "anyres", "anyres_max", "anyres_e2p", "resize",
    ] = "anyres_e2p"
    image_size: Optional[List[int]] = None
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    stitching_mode: str = "concat"
    stitch_target_to_view_width: bool = True
    stitch_interp: str = "linear"
    fov_deg: float = Field(default=90.0, gt=0.0)
    use_vision_processor: bool = True
    anyres_max_patches: int = Field(default=9, gt=0)
    normalize: bool = True

    model_config = {"extra": "allow"}


class LoRAConfig(BaseModel):
    use_lora: bool = False
    rank: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    save_lora_only: bool = False

    model_config = {"extra": "allow"}


class GenerationConfig(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    vision_name: str = "google/siglip-base-patch16-224"
    vision_backbone_type: str = "hf"
    vision_backbone_kwargs: Optional[Dict[str, Any]] = None
    vision_input_key: str = "pixel_values"
    vision_output_key: str = "last_hidden_state"
    vision_forward_method: Optional[str] = None
    vision_hidden_size: Optional[int] = None
    use_vicreg_norm: bool = False

    language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    resampler_type: Literal[
        "mlp", "bimamba", "qformer", "perceiver",
        "spatial_pool", "masked_drop", "c_abstractor",
    ] = "bimamba"
    latent_dimension: int = 768
    resampler_config: Dict[str, Any] = Field(default_factory=dict)

    use_projection_positional_encoding: bool = True
    pe_view_encoding_type: str = "sinusoidal"
    pe_spatial_encoding_type: str = "sinusoidal"
    pe_enable_continuity: bool = True

    use_vicreg_projector: bool = True
    vicreg_projector_dim: Optional[int] = None
    vicreg_projector_depth: int = 2
    vicreg_projector_ln: bool = True

    vicreg_projector_dropout: float = Field(
        default=0.1, ge=0.0, le=0.5,
        description=(
            "Dropout probability in VICReg projector hidden layers. "
            "Used by contrastive loss to create two stochastic views. "
            "Set 0.0 for classic VICReg (no dropout needed)."
        ),
    )

    use_text_projection: bool = False

    @model_validator(mode="before")
    @classmethod
    def apply_resampler_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        rtype = values.get("resampler_type", "bimamba")
        user_cfg = values.get("resampler_config", {})
        defaults = RESAMPLER_CONFIGS.get(rtype, {}).copy()
        values["resampler_config"] = {**defaults, **user_cfg}
        return values

    model_config = {"extra": "allow"}


class StageDataConfig(BaseModel):
    csv_train: Union[str, List[str]] = ""
    csv_val: Union[str, List[str]] = ""

    model_config = {"extra": "allow"}


class StageConfig(BaseModel):
    epochs: int = Field(default=3, gt=0)
    lr: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=1, ge=-1)  # -1 = autobatch

    @field_validator("batch_size")
    @classmethod
    def _reject_zero_batch(cls, v: int) -> int:
        if v == 0:
            raise ValueError("batch_size=0 is invalid; use -1 for autobatch or a positive integer")
        return v
    accumulate_grad_batches: int = Field(default=1, gt=0)
    vicreg_loss_weight: float = Field(default=0.0, ge=0.0)
    vicreg_mode: str = "pairwise"
    vicreg_similarity_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_overlap_ratio: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Fraction of view width used for VICReg invariance loss. "
            "If None, falls back to image_processing.overlap_ratio. "
            "Set lower than physical overlap (e.g. 0.25 vs 0.5) to leave "
            "free columns per view and prevent identity-chain collapse."
        ),
    )
    vision_loss_type: str = Field(
        default="contrastive",
        description=(
            "'contrastive' = PanoContrastiveLoss (symmetric InfoNCE, two dropout views). "
            "'vicreg' = original VICRegLoss (pairwise variance/covariance regularisation). "
            "'densecl' = DenseCL (single-view overlap InfoNCE, simplest)."
        ),
    )
    densecl_temperature: float = Field(
        default=0.07, gt=0.0,
        description="Temperature for DenseCL InfoNCE. Lower → sharper similarity.",
    )
    contrastive_tau_overlap: float = Field(
        default=0.07, gt=0.0,
        description="Temperature for overlap InfoNCE. Lower → sharper.",
    )
    contrastive_tau_tile: float = Field(
        default=0.2, gt=0.0,
        description=(
            "Temperature for within-tile InfoNCE. Keep higher than tau_overlap "
            "to tolerate repetitive textures (fewer false negatives)."
        ),
    )
    contrastive_tile_weight: float = Field(
        default=0.1, ge=0.0,
        description=(
            "Weight λ_tile for within-tile discrimination loss. "
            "Set 0.0 to use overlap alignment only."
        ),
    )

    global_local_loss_weight: float = Field(default=0.0, ge=0.0)
    global_local_loss_type: str = "cosine"  # "cosine" or "mse"
    vision_trainable_blocks: int = 0
    vision_lora_enabled: bool = False
    vision_lora_target_modules: Optional[List[str]] = None
    max_text_length: Union[int, str] = 128
    image_processing: Optional[ImageProcessingConfig] = None
    data: Optional[StageDataConfig] = None

    model_config = {"extra": "allow"}

    @classmethod
    def default_for(cls, stage: str) -> StageConfig:
        if stage == "vision":
            return cls(
                epochs=5, lr=5e-4, batch_size=8,
                vicreg_loss_weight=1.0, vision_trainable_blocks=2,
                max_text_length=32, vision_loss_type="contrastive",
            )
        elif stage == "resampler":
            return cls(
                epochs=1, lr=1e-4, batch_size=1,
                accumulate_grad_batches=4, max_text_length=128,
                vicreg_loss_weight=0.1, vision_loss_type="contrastive",
            )
        elif stage == "finetune":
            return cls(
                epochs=1, lr=5e-5, batch_size=1,
                accumulate_grad_batches=4, max_text_length=128,
            )
        return cls()


class TrainingConfig(BaseModel):
    stages: List[str] = Field(default_factory=lambda: ["vision", "resampler", "finetune"])
    max_epochs: int = 10
    devices: Union[int, List[int]] = 1
    strategy: str = "auto"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    num_workers: int = 8
    eval_batch_size: int = 2
    system_msg: str = "You are a helpful assistant. Describe the panorama image."
    wandb_project: Optional[str] = None
    cache_cleanup_interval: int = 1000
    seed: int = 42
    stage_configs: Dict[str, Any] = Field(default_factory=dict)
    deepspeed: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    auto_eval: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

    model_config = {"extra": "allow"}

    def get_stage_config(self, stage: str) -> StageConfig:
        defaults = StageConfig.default_for(stage)
        overrides = self.stage_configs.get(stage, {})
        if isinstance(overrides, StageConfig):
            return overrides
        if isinstance(overrides, dict):
            merged = {**defaults.model_dump(exclude_none=True), **overrides}
            return StageConfig(**merged)
        return defaults


class OutputConfig(BaseModel):
    runs_dir: str = "runs"
    outputs_dir: str = "outputs"

    def resolve_experiment_dir(self, experiment_name: str) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        base = Path(self.runs_dir) / experiment_name
        base.mkdir(parents=True, exist_ok=True)

        idx = 1
        while True:
            candidate = base / f"{today}_{idx:03d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            idx += 1

    def resolve_output_dir(self, experiment_name: str) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        out = Path(self.outputs_dir) / experiment_name / today
        out.mkdir(parents=True, exist_ok=True)
        return out


class BaselineConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: Dict[str, Any] = Field(default_factory=lambda: {
        "epochs": 3, "lr": 2e-5, "batch_size": 1, "accumulate_grad_batches": 4,
    })
    data: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "runs/baseline"

    model_config = {"extra": "allow"}


class CORAConfig(BaseModel):
    experiment: Dict[str, str] = Field(default_factory=lambda: {"name": "auto", "version": "1.0"})
    environment: Dict[str, Any] = Field(default_factory=lambda: {"wandb_project": "cora-training"})
    paths: Dict[str, str] = Field(default_factory=dict)

    models: ModelConfig = Field(default_factory=ModelConfig)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    data: Dict[str, Any] = Field(default_factory=dict)
    system_messages: Optional[Dict[str, str]] = None

    model_config = {"extra": "allow"}

    def get_stage_config(self, stage: str) -> StageConfig:
        return self.training.get_stage_config(stage)
