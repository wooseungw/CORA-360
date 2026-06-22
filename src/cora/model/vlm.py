"""Main Panorama VLM orchestrator.

:class:`PanoramaVLM` combines all CORA model components — vision encoder,
resampler, projectors, language model, and fusion — into a single module that
supports 3-stage progressive training:

1. **Vision stage** – VICReg self-supervised learning on resampled features.
2. **Resampler stage** – Align vision tokens with the frozen language model.
3. **Finetune stage** – End-to-end fine-tuning (typically with LoRA).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cora.config.schema import CORAConfig
from cora.model.language_fusion import LanguageFusion
from cora.model.language_model import LanguageModel
from cora.model.projectors import PanoramaProjector, VICRegProjector
from cora.model.resampler import ResamplerModule
from cora.model.vision_encoder import VisionBackbone

__all__ = ["PanoramaVLM"]

logger = logging.getLogger(__name__)


class PanoramaVLM(nn.Module):
    """Main VLM orchestrator combining vision, resampling, projection, and language.

    Supports three training stages:

    * **vision** – runs vision encoder + resampler + VICReg projector; returns
      features for VICReg loss computation.
    * **resampler** – runs the full vision-to-language pipeline with a frozen
      language model for alignment training.
    * **finetune** – end-to-end training (usually with LoRA on the LLM).

    Args:
        config: A fully populated :class:`~cora.config.schema.CORAConfig` instance.
    """

    def __init__(self, config: CORAConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Vision Encoder
        logger.info("Initializing Vision Encoder...")
        self.vision_encoder = VisionBackbone(
            vision_name=config.models.vision_name,
            use_vicreg_norm=config.models.use_vicreg_norm,
            backbone_type=config.models.vision_backbone_type,
        )
        self.vision_hidden_size: int = self.vision_encoder.hidden_size

        # 2. Resampler
        logger.info("Initializing Resampler...")
        self.resampler_module = ResamplerModule(config.models, self.vision_hidden_size)
        self.resampler = self.resampler_module  # convenience alias
        self.latent_dimension: int = self.resampler_module.output_dim

        # 3. VICReg Projector (stage 1)
        if config.models.use_vicreg_projector:
            self.vicreg_projector: nn.Module = VICRegProjector(
                in_dim=self.latent_dimension,
                out_dim=config.models.vicreg_projector_dim or self.latent_dimension,
                depth=config.models.vicreg_projector_depth,
                use_ln=config.models.vicreg_projector_ln,
                dropout=getattr(config.models, "vicreg_projector_dropout", 0.0),
            )
        else:
            self.vicreg_projector = nn.Identity()

        # 4. Language Model
        logger.info("Initializing Language Model...")
        self.language_model_wrapper = LanguageModel(config.models.language_model_name)
        self.language_model = self.language_model_wrapper.model
        self.tokenizer = self.language_model_wrapper.tokenizer
        self.llm_hidden_size: int = self.language_model.config.hidden_size

        # 5. Panorama Projector (vision -> LLM)
        logger.info("Initializing Panorama Projector...")
        self.projector = PanoramaProjector(
            config=config.models,
            latent_dimension=self.latent_dimension,
            language_hidden_size=self.llm_hidden_size,
            image_processing=config.image_processing,
        )
        self.vision_to_language_projection = self.projector  # backward-compat alias

        # 6. Text Projection (optional)
        if config.models.use_text_projection:
            self.text_projection: nn.Module = nn.Linear(
                self.llm_hidden_size, self.llm_hidden_size,
            )
        else:
            self.text_projection = nn.Identity()

        # 7. Language Fusion
        vision_token_str = "<|vision|>"
        vision_token_id = self.tokenizer.convert_tokens_to_ids(vision_token_str)
        if vision_token_id == self.tokenizer.unk_token_id:
            logger.warning(
                "Vision token '%s' not found in vocabulary; using unk_token_id.",
                vision_token_str,
            )

        finetune_cfg = config.training.stage_configs.get("finetune")
        max_text_len = 2048
        if finetune_cfg is not None:
            if isinstance(finetune_cfg, dict):
                max_text_len = finetune_cfg.get("max_text_length", 2048)
            else:
                max_text_len = getattr(finetune_cfg, "max_text_length", 2048)

        self.fusion = LanguageFusion(
            language_model=self.language_model,
            tokenizer=self.tokenizer,
            vision_token_id=vision_token_id,
            ignore_index=-100,
            max_text_length=max_text_len,
        )

        # 8. LoRA (optional)
        if config.lora.use_lora:
            logger.info("Applying LoRA to language model...")
            self.language_model_wrapper.setup_lora(
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
            )

        self.dtype_cache: Dict[str, torch.dtype] = {}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_vision_tower(self) -> VisionBackbone:
        """Return the vision encoder module."""
        return self.vision_encoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_resampler(
        self,
        pixel_values: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Run vision encoder -> resampler pipeline.

        Returns:
            ``[B*V, S, D_latent]`` resampled features.
        """
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs["vision_features"]
        return self.resampler_module(vision_features)

    def _project_from_resampled(
        self,
        resampled: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Project pre-computed resampled features to LLM token space.

        Handles global/tile separation: the global view (index 0) is projected
        through the linear layer only (no PE, no stitching), while E2P tiles
        go through the full projector pipeline.

        Args:
            resampled: ``[B*V, S, D_latent]`` from :meth:`_process_resampler`.

        Returns:
            ``[B, T, D_lm]`` vision tokens ready for language fusion.
        """
        has_global = num_views > 1
        if has_global:
            S, D = resampled.shape[1], resampled.shape[2]
            all_views = resampled.view(batch_size, num_views, S, D)
            global_resampled = all_views[:, 0]          # [B, S, D_latent]
            tile_resampled = all_views[:, 1:].reshape(
                batch_size * (num_views - 1), S, D,
            )
            num_tiles = num_views - 1

            # Tiles: full projector pipeline (PE + linear + stitching)
            tile_tokens = self.projector(
                resampled_features=tile_resampled,
                batch_size=batch_size,
                num_views=num_tiles,
                dtype_cache=self.dtype_cache,
                language_model=self.language_model,
            )

            # Global: linear projection only (no PE, no stitching)
            global_input = global_resampled.to(self.projector.linear.weight.dtype)
            global_tokens = self.projector.linear(global_input)
            if global_tokens.dtype != tile_tokens.dtype:
                global_tokens = global_tokens.to(tile_tokens.dtype)

            return torch.cat([global_tokens, tile_tokens], dim=1)

        return self.projector(
            resampled_features=resampled,
            batch_size=batch_size,
            num_views=num_views,
            dtype_cache=self.dtype_cache,
            language_model=self.language_model,
        )

    def _project_vision_tokens(
        self,
        pixel_values: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Run full vision → resampler → projector pipeline.

        Convenience wrapper that calls :meth:`_process_resampler` then
        :meth:`_project_from_resampled`.
        """
        resampled = self._process_resampler(pixel_values, batch_size, num_views)
        return self._project_from_resampled(resampled, batch_size, num_views)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "finetune",
        **kwargs: Any,
    ) -> Any:
        """Forward pass supporting all three training stages.

        Args:
            pixel_values: ``[B, V, C, H, W]`` or ``[B, C, H, W]``.
            input_ids: ``[B, L]`` text token IDs (stages 2 & 3).
            attention_mask: ``[B, L]`` attention mask.
            labels: ``[B, L]`` training labels.
            stage: One of ``"vision"``, ``"resampler"``, ``"finetune"``.

        Returns:
            Stage-dependent output dict or model outputs.
        """
        B_pixel = pixel_values.shape[0] if pixel_values is not None else 1
        num_views = (
            pixel_values.shape[1]
            if pixel_values is not None and pixel_values.ndim == 5
            else 1
        )

        if stage == "vision":
            resampled = self._process_resampler(pixel_values, B_pixel, num_views)
            vicreg_feats = self.vicreg_projector(resampled)

            has_global = num_views > 1
            if has_global:
                all_views = vicreg_feats.view(B_pixel, num_views, -1, vicreg_feats.size(-1))
                global_feats = all_views[:, 0]  # [B, S, D]
                tile_feats = all_views[:, 1:].reshape(B_pixel * (num_views - 1), -1, vicreg_feats.size(-1))
                num_tiles = num_views - 1
                # Pre-projector (resampler output) for diagnostics
                resamp_views = resampled.view(B_pixel, num_views, -1, resampled.size(-1))
                resamp_tiles = resamp_views[:, 1:].reshape(
                    B_pixel * (num_views - 1), -1, resampled.size(-1),
                )
            else:
                global_feats = None
                tile_feats = vicreg_feats
                num_tiles = num_views
                resamp_tiles = resampled

            return {
                "vicreg_features": tile_feats,
                "resampler_features": resamp_tiles,
                "global_features": global_feats,
                "batch_size": B_pixel,
                "num_views": num_tiles,
            }

        if stage == "resampler":
            # Stage 2: Joint VICReg regularisation + LM alignment.
            # Compute resampled features ONCE, then branch to both paths.
            resampled = self._process_resampler(pixel_values, B_pixel, num_views)

            # Branch A – VICReg features (frozen projector, gradient → resampler)
            vicreg_feats = self.vicreg_projector(resampled)
            has_global = num_views > 1
            if has_global:
                vv = vicreg_feats.view(B_pixel, num_views, -1, vicreg_feats.size(-1))
                global_feats = vv[:, 0]
                tile_feats = vv[:, 1:].reshape(
                    B_pixel * (num_views - 1), -1, vicreg_feats.size(-1),
                )
                num_tiles = num_views - 1
                # Pre-projector resampler output (tiles only) for contrastive
                resamp_views = resampled.view(B_pixel, num_views, -1, resampled.size(-1))
                resamp_tiles = resamp_views[:, 1:].reshape(
                    B_pixel * (num_views - 1), -1, resampled.size(-1),
                )
            else:
                global_feats = None
                tile_feats = vicreg_feats
                num_tiles = num_views
                resamp_tiles = resampled

            # Branch B – LM loss (via PanoramaProjector → fusion → LLM)
            vision_tokens = self._project_from_resampled(
                resampled, B_pixel, num_views,
            )
            fused_inputs = self.fusion.fuse(
                vision_tokens=vision_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_output = self.language_model(
                inputs_embeds=fused_inputs["inputs_embeds"],
                attention_mask=fused_inputs["attention_mask"],
                labels=fused_inputs.get("labels"),
                return_dict=True,
            )

            return {
                "loss": lm_output.loss,
                "vicreg_features": tile_feats,
                "resampler_features": resamp_tiles,
                "global_features": global_feats,
                "batch_size": B_pixel,
                "num_views": num_tiles,
            }

        if stage == "finetune":
            # Stage 3: LM loss only (no VICReg).
            vision_tokens = self._project_vision_tokens(
                pixel_values, B_pixel, num_views,
            )

            fused_inputs = self.fusion.fuse(
                vision_tokens=vision_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            return self.language_model(
                inputs_embeds=fused_inputs["inputs_embeds"],
                attention_mask=fused_inputs["attention_mask"],
                labels=fused_inputs.get("labels"),
                return_dict=True,
            )

        raise ValueError(
            f"Unknown training stage: {stage!r}. "
            "Expected one of 'vision', 'resampler', 'finetune'."
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate text conditioned on panoramic image input.

        Args:
            pixel_values: ``[B, V, C, H, W]`` or ``[B, C, H, W]``.
            input_ids: Optional user-prompt token IDs.
            attention_mask: Matching attention mask.
            generation_config: Dict of generation hyper-parameters
                (``max_new_tokens``, ``temperature``, etc.).

        Returns:
            Generated token IDs from the language model.
        """
        B_pixel = pixel_values.shape[0] if pixel_values is not None else 1
        num_views = (
            pixel_values.shape[1]
            if pixel_values is not None and pixel_values.ndim == 5
            else 1
        )

        vision_tokens = self._project_vision_tokens(
            pixel_values, B_pixel, num_views,
        )

        if input_ids is None:
            input_ids, attention_mask = self.fusion.create_default_prompt(
                B_pixel, pixel_values.device,
            )

        fused_inputs = self.fusion.fuse(
            vision_tokens=vision_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        gen_kwargs = dict(generation_config or {})
        gen_kwargs.update(kwargs)
        gen_kwargs.setdefault("max_new_tokens", 128)

        return self.language_model.generate(
            inputs_embeds=fused_inputs["inputs_embeds"],
            attention_mask=fused_inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
