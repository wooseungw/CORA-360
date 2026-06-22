"""Inference generator for CORA: loads checkpoint and generates text from images."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer

from cora.config.manager import ConfigManager
from cora.processors.images import PanoramaImageProcessor
from cora.processors.processor import PanoramaProcessor

logger = logging.getLogger(__name__)

# Qwen3 thinking-mode prefix: forces the model to skip internal reasoning.
_NO_THINK_PREFIX = "<think>\n\n</think>\n\n"
_THINK_STRIP_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class PanoramaGenerator:
    """Load a CORA checkpoint and generate text for panoramic images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Locate config
        config = ConfigManager.auto_detect_config(self.checkpoint_path)
        if config is None:
            raise FileNotFoundError(f"Config not found near checkpoint: {self.checkpoint_path}")
        self.config = config

        # Load model
        logger.info("Loading model from %s ...", checkpoint_path)
        from cora.training.module import PanoramaTrainingModule

        self.module = PanoramaTrainingModule.load_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            config=self.config,
            stage="finetune",
            map_location="cpu",
        )
        self.model = self.module.model
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Disable gradient checkpointing (useful for training, wasteful for inference)
        lm = getattr(self.model, "language_model", None)
        lm_inner = getattr(lm, "model", None) if lm else None
        if lm_inner and hasattr(lm_inner, "gradient_checkpointing_disable"):
            lm_inner.gradient_checkpointing_disable()

        # Processor
        self._setup_processor()

        # Detect Qwen3 to apply no-think workaround during generation
        model_name = getattr(self.config.models, "language_model_name", "").lower()
        self._is_qwen3 = "qwen3" in model_name

    def _setup_processor(self) -> None:
        img_cfg = self.config.image_processing
        img_proc = PanoramaImageProcessor(
            image_size=tuple(img_cfg.image_size) if img_cfg.image_size else None,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio,
            normalize=img_cfg.normalize,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.models.language_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Register <|vision|> special token so it produces the same single
        # token ID that LanguageFusion.fuse() expects.  Without this, the
        # placeholder is split into sub-tokens and vision embeddings are
        # inserted at a fallback position instead of the intended location.
        vision_token = "<|vision|>"
        existing = [str(t) for t in tokenizer.additional_special_tokens]
        if vision_token not in existing:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": existing + [vision_token]}
            )

        self.processor = PanoramaProcessor(img_proc, tokenizer)

    @staticmethod
    def _postprocess(text: str) -> str:
        """Strip ``<think>...</think>`` blocks and clean whitespace."""
        cleaned = _THINK_STRIP_RE.sub("", text)
        # Also strip everything before the actual answer (e.g. repeated system/user)
        # HF decode with skip_special_tokens may leave the full conversation
        # so we take only the last assistant turn's content.
        return cleaned.strip()

    @staticmethod
    def _build_gen_config(
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build generation kwargs with deterministic (greedy) defaults.

        Qwen3 ships with ``do_sample=True, temperature=0.6`` in its default
        ``generation_config``.  For reproducible evaluation we override to
        greedy decoding (``do_sample=False``) unless the caller explicitly
        passes different values.  This matches the baseline evaluation setup.
        """
        cfg: Dict[str, Any] = {
            "max_new_tokens": 128,
            "do_sample": False,
        }
        if overrides:
            cfg.update(overrides)
        return cfg

    @torch.inference_mode()
    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe the panorama image.",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text for a single image + prompt pair.

        For Qwen3-based models the no-think prefix (``<think>\\n\\n</think>``)
        is automatically appended to the prompt so the model skips its internal
        chain-of-thought and produces a direct answer.  Any residual
        ``<think>...</think>`` blocks in the output are stripped during
        post-processing.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        formatted = self.processor.text_formatter.format_conversation(
            prompt, add_generation_prompt=True
        )

        # Append no-think prefix for Qwen3 models so the model emits a direct
        # answer instead of spending tokens on internal reasoning.
        if self._is_qwen3:
            formatted += _NO_THINK_PREFIX

        tokenized = self.processor.tokenizer(formatted, return_tensors="pt")

        inputs: Dict[str, Any] = {k: v.to(self.device) for k, v in tokenized.items()}
        pixel_values = self.processor.image_processor(image)
        inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=self.dtype)

        gen_cfg = self._build_gen_config(generation_config)

        outputs = self.model.generate(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            **gen_cfg,
        )

        # NOTE: model.generate() uses inputs_embeds internally, so outputs
        # contain ONLY newly generated tokens (no prompt prefix).
        text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._postprocess(text)

    @torch.inference_mode()
    def generate_with_pixel_values(
        self,
        pixel_values: torch.Tensor,
        prompt: str = "Describe the panorama image.",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text with pre-computed pixel values (skip image processing).

        Use this for batch evaluation where the same image has multiple
        queries — compute ``pixel_values`` once via
        ``generator.processor.image_processor(image)`` and reuse.
        """
        formatted = self.processor.text_formatter.format_conversation(
            prompt, add_generation_prompt=True
        )
        if self._is_qwen3:
            formatted += _NO_THINK_PREFIX

        tokenized = self.processor.tokenizer(formatted, return_tensors="pt")
        inputs: Dict[str, Any] = {k: v.to(self.device) for k, v in tokenized.items()}
        inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=self.dtype)

        gen_cfg = self._build_gen_config(generation_config)

        outputs = self.model.generate(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            **gen_cfg,
        )

        # NOTE: model.generate() uses inputs_embeds internally, so outputs
        # contain ONLY newly generated tokens (no prompt prefix).
        text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._postprocess(text)

    @torch.inference_mode()
    def batch_generate(
        self,
        images: List[Union[str, Image.Image]],
        prompts: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate text for multiple image+prompt pairs (sequential)."""
        return [self.generate(img, p, generation_config) for img, p in zip(images, prompts)]
