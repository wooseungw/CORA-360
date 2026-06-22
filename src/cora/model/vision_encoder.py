"""Vision backbone wrapper supporting HuggingFace, timm, torchhub, and custom models.

Provides a unified interface for extracting patch-level features from pretrained
vision encoders (SigLIP, CLIP, DINOv2, etc.) with support for multi-view
panoramic inputs.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

__all__ = ["VisionBackbone"]

logger = logging.getLogger(__name__)


class VisionBackbone(nn.Module):
    """Wraps a pretrained vision encoder with a standard ``[B*V, S, D]`` output API.

    Supports multiple backend types:

    * ``"hf"`` – HuggingFace ``AutoModel`` (default, recommended).
    * ``"timm"`` – timm ``create_model``.
    * ``"torchhub"`` – ``torch.hub.load`` (format: ``"repo:model_name"``).
    * ``"module"`` – arbitrary ``module.path.ClassName``.

    The encoder is **frozen** by default; use :meth:`unfreeze_last_n_blocks` to
    selectively unfreeze layers for vision-stage training.

    Args:
        vision_name: Model identifier (HF model ID, timm model name, etc.).
        use_vicreg_norm: Apply a trainable LayerNorm on top of encoder output.
        backbone_type: One of ``"hf"``, ``"timm"``, ``"torchhub"``, ``"module"``.
        backbone_kwargs: Extra keyword arguments forwarded to the model loader.
        input_key: Key used when calling the encoder (default ``"pixel_values"``).
        output_key: Attribute/key to extract from encoder output.
        forward_method: If set, call this method instead of ``__call__``.
        hidden_size: Override the auto-detected hidden size.
    """

    def __init__(
        self,
        vision_name: str,
        use_vicreg_norm: bool = False,
        backbone_type: str = "hf",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        input_key: str = "pixel_values",
        output_key: str = "last_hidden_state",
        forward_method: Optional[str] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.input_key = input_key
        self.output_key = output_key
        self.forward_method = forward_method
        self.vision_name = vision_name

        logger.info("Loading vision backbone: %s (type=%s)", vision_name, backbone_type)

        encoder = self._load_encoder(vision_name, backbone_type, backbone_kwargs)
        self.encoder = encoder
        self.hidden_size = int(hidden_size) if hidden_size is not None else self._infer_hidden_size(encoder)
        self.use_vicreg_norm = use_vicreg_norm
        self.norm: nn.Module = nn.LayerNorm(self.hidden_size) if use_vicreg_norm else nn.Identity()

        # Default: freeze the encoder, keep norm trainable
        self.requires_grad_(False)
        self.norm.requires_grad_(True)

    # ------------------------------------------------------------------
    # Encoder loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_encoder(
        vision_name: str,
        backbone_type: str,
        backbone_kwargs: Dict[str, Any],
    ) -> nn.Module:
        if backbone_type == "hf":
            encoder = AutoModel.from_pretrained(
                vision_name, trust_remote_code=True, **backbone_kwargs
            )
            # Many HF models wrap the vision encoder inside a `.vision_model`
            if hasattr(encoder, "vision_model"):
                encoder = encoder.vision_model
            return encoder

        if backbone_type == "timm":
            import timm  # type: ignore[import-untyped]
            return timm.create_model(vision_name, pretrained=True, **backbone_kwargs)

        if backbone_type == "torchhub":
            repo, model_name = vision_name.split(":", 1)
            return torch.hub.load(repo, model_name, **backbone_kwargs)

        if backbone_type == "module":
            import importlib
            module_path, attr_name = vision_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            encoder_cls = getattr(module, attr_name)
            return encoder_cls(**backbone_kwargs)

        raise ValueError(f"Unsupported backbone_type: {backbone_type!r}")

    # ------------------------------------------------------------------
    # Layer unfreezing
    # ------------------------------------------------------------------

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last *n* transformer blocks of the vision encoder.

        Searches for common layer-list attributes (``encoder.layers``,
        ``layers``, ``blocks``) and unfreezes the trailing *n* blocks.
        """
        if n <= 0:
            return

        logger.info("Unfreezing last %d blocks of vision encoder", n)

        layers = None
        # CLIP / SigLIP
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            layers = self.encoder.encoder.layers
        # Direct ViT
        elif hasattr(self.encoder, "layers"):
            layers = self.encoder.layers
        # timm ViT
        elif hasattr(self.encoder, "blocks"):
            layers = self.encoder.blocks

        if layers is not None:
            num_layers = len(layers)
            for layer in layers[max(0, num_layers - n):]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            logger.warning(
                "Could not auto-detect layer structure for %s. "
                "Manual unfreezing may be required.",
                self.vision_name,
            )

    def setup_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[list[str]] = None,
    ) -> bool:
        """Apply LoRA adapters to the wrapped vision encoder via PEFT.

        PEFT's official LoRA guide recommends wrapping the base model with
        ``get_peft_model()`` and explicitly choosing supported target modules
        such as attention/linear projections.
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. Vision LoRA setup skipped.")
            return False

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

        try:
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
            return True
        except Exception as exc:
            logger.error("Failed to setup vision LoRA: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Hidden-size inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hidden_size(vision_model: nn.Module) -> int:
        """Attempt to infer the hidden dimension from model config or attributes."""
        config = getattr(vision_model, "config", None)
        for key in ("hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"):
            if config is not None and hasattr(config, key):
                return int(getattr(config, key))
            if hasattr(vision_model, key):
                return int(getattr(vision_model, key))
        for key in ("num_features", "feature_dim"):
            if hasattr(vision_model, key):
                return int(getattr(vision_model, key))
        raise AttributeError(
            f"Could not infer hidden size for {type(vision_model).__name__}. "
            "Please specify `hidden_size` explicitly."
        )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def normalize_inputs(
        self,
        pixel_values: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[int, int, torch.Tensor]:
        """Reshape pixel inputs to ``[B, V, C, H, W]`` and optionally cast dtype.

        Returns:
            Tuple of ``(batch_size, num_views, pixel_values_5d)``.
        """
        if pixel_values.ndim == 5:
            batch_size, num_views = pixel_values.shape[:2]
        elif pixel_values.ndim == 4:
            batch_size = pixel_values.shape[0]
            num_views = 1
            pixel_values = pixel_values.unsqueeze(1)
        else:
            raise ValueError(
                f"Expected 4-D or 5-D pixel_values, got shape {pixel_values.shape}"
            )

        if target_dtype is not None and pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(target_dtype)

        return batch_size, num_views, pixel_values

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, normalized_pixels: torch.Tensor) -> torch.Tensor:
        """Run the encoder on ``[B, V, C, H, W]`` inputs and return ``[B*V, S, D]``.

        Handles SigLIP ``interpolate_pos_encoding`` flag and multiple output
        formats (tensors, tuples, dicts, dataclasses).
        """
        # Flatten views into batch dimension
        flattened = normalized_pixels.view(-1, *normalized_pixels.shape[2:])

        if self.forward_method == "forward_features" and hasattr(self.encoder, "forward_features"):
            vision_output = self.encoder.forward_features(flattened)
        else:
            call_kwargs: Dict[str, Any] = {self.input_key: flattened}

            # SigLIP needs interpolate_pos_encoding=True for non-native resolutions
            try:
                model_type = getattr(getattr(self.encoder, "config", None), "model_type", "")
                if model_type and "siglip" in model_type:
                    call_kwargs["interpolate_pos_encoding"] = True
            except Exception:  # noqa: BLE001
                pass

            try:
                vision_output = self.encoder(**call_kwargs, return_dict=True)
            except TypeError:
                vision_output = self.encoder(**call_kwargs)

        vision_hidden_states = self._extract_hidden_states(vision_output)

        # Ensure [B*V, S, D] layout
        if vision_hidden_states.dim() == 4:
            vision_hidden_states = self._reshape_4d_to_3d(vision_hidden_states)

        return self.norm(vision_hidden_states)

    def _extract_hidden_states(self, vision_output: Any) -> torch.Tensor:
        """Extract the relevant hidden-state tensor from encoder output."""
        if isinstance(vision_output, torch.Tensor):
            return vision_output
        if isinstance(vision_output, (tuple, list)) and vision_output:
            return vision_output[0]
        if hasattr(vision_output, self.output_key):
            return getattr(vision_output, self.output_key)
        if isinstance(vision_output, dict) and self.output_key in vision_output:
            return vision_output[self.output_key]
        raise ValueError(
            f"Cannot extract hidden states via key {self.output_key!r} "
            f"from output of type {type(vision_output).__name__}"
        )

    def _reshape_4d_to_3d(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, C, H, W]`` or ``[B, H, W, C]`` to ``[B, H*W, D]``."""
        if t.shape[1] == self.hidden_size:
            # Channel-first: [B, D, H, W] -> [B, H*W, D]
            return t.flatten(2).transpose(1, 2)
        if t.shape[-1] == self.hidden_size:
            # Channel-last: [B, H, W, D] -> [B, H*W, D]
            return t.view(t.size(0), -1, self.hidden_size)
        # Fallback: assume channel-first
        return t.flatten(2).transpose(1, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Full forward pass: normalise → extract → return feature dict.

        Args:
            pixel_values: ``[B, V, C, H, W]`` or ``[B, C, H, W]``.
            target_dtype: Optional dtype to cast inputs to.

        Returns:
            Dict with keys ``"vision_features"`` (``[B*V, S, D]``),
            ``"batch_size"``, ``"num_views"``, ``"device"``.
        """
        batch_size, num_views, normalized_pixels = self.normalize_inputs(pixel_values, target_dtype)
        vision_features = self.extract_features(normalized_pixels)
        return {
            "vision_features": vision_features,
            "batch_size": batch_size,
            "num_views": num_views,
            "device": normalized_pixels.device,
        }

    # ------------------------------------------------------------------
    # CLS token heuristic
    # ------------------------------------------------------------------

    def has_cls_token(self, vision_output: torch.Tensor) -> bool:
        """Heuristic check for whether the encoder prepends a CLS token."""
        for attr in ("cls_token", "class_token", "class_embedding"):
            if hasattr(self.encoder, attr):
                return True
        seq_len = vision_output.size(1)
        if seq_len > 1:
            root = math.isqrt(seq_len)
            if root * root == seq_len:
                return False
            if (root := math.isqrt(seq_len - 1)) * root == (seq_len - 1):
                return True
        return False
