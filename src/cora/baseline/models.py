"""Baseline model registry for commercial VLMs.

Registry of supported VLM model types with their HuggingFace classes,
default LoRA target modules, and loading logic. Ported from
legacy/root_scripts/vlm_finetune_and_eval.py adapter pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import BaselineModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional model class imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from transformers import LlavaOnevisionForConditionalGeneration
except ImportError:
    LlavaOnevisionForConditionalGeneration = None  # type: ignore[assignment,misc]

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# dtype helper
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string dtype specification to a ``torch.dtype``.

    Falls back to float16 when bfloat16 is requested but unsupported.
    """
    key = dtype_str.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = _DTYPE_MAP[key]
    if dtype is torch.bfloat16:
        bf16_ok = (
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if not bf16_ok:
            logger.warning("bfloat16 not supported on this GPU; falling back to float16.")
            return torch.float16
    return dtype


# ---------------------------------------------------------------------------
# ERP-RoPE width-matched single-image: lock pixel budget to the pre-resize size
# so Qwen smart_resize is a no-op (preserves grid_w/grid_h exactly).
# ---------------------------------------------------------------------------

def _apply_erp_resize_pixel_lock(config: BaselineModelConfig) -> None:
    if not config.erp_resize_width:
        return
    w_px = int(config.erp_resize_width)
    h_px = int(config.erp_resize_height) if config.erp_resize_height else w_px // 2
    locked = w_px * h_px
    if config.min_pixels and config.min_pixels != locked:
        logger.warning(
            "erp_resize=%dx%d implies %d pixels but min_pixels=%d is set; "
            "leaving user override in place.", w_px, h_px, locked, config.min_pixels,
        )
        return
    config.min_pixels = locked
    config.max_pixels = locked
    config.dynamic_resolution = True
    logger.info(
        "ERP resize lock: %dx%d → min/max_pixels=%d (dynamic_resolution=True)",
        w_px, h_px, locked,
    )


# ---------------------------------------------------------------------------
# Registry data: model_type → (model_class_loader, processor_loader, default_lora_targets)
# ---------------------------------------------------------------------------

def _model_entry_qwen25_vl() -> Dict[str, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    return {
        "model_cls": Qwen2_5_VLForConditionalGeneration,
        "processor_loader": lambda pid: AutoProcessor.from_pretrained(pid, trust_remote_code=True),
        "processor_post_init": _qwen_vl_processor_post_init,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": True,
    }


def _model_entry_qwen2_vl() -> Dict[str, Any]:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    return {
        "model_cls": Qwen2VLForConditionalGeneration,
        "processor_loader": lambda pid: AutoProcessor.from_pretrained(pid, trust_remote_code=True),
        "processor_post_init": _qwen_vl_processor_post_init,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": True,
    }


def _qwen_vl_processor_post_init(
    processor: Any,
    image_size: int = 224,
    dynamic_resolution: bool = False,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Any:
    """Configure Qwen VL image processor resolution settings.

    When ``dynamic_resolution=False`` (default), locks min=max=image_size²
    for deterministic batching.  When ``True``, uses the provided pixel
    range or falls back to Qwen2.5-VL training defaults (112²–448²).
    """
    try:
        if hasattr(processor, "image_processor"):
            if dynamic_resolution:
                processor.image_processor.min_pixels = min_pixels or 28 * 28 * 16
                processor.image_processor.max_pixels = max_pixels or 28 * 28 * 256
                logger.info(
                    "Qwen VL dynamic resolution: min_pixels=%d, max_pixels=%d",
                    processor.image_processor.min_pixels,
                    processor.image_processor.max_pixels,
                )
            else:
                pixels = image_size * image_size
                processor.image_processor.min_pixels = pixels
                processor.image_processor.max_pixels = pixels
    except Exception:
        pass
    return processor


def _model_entry_llava() -> Dict[str, Any]:
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    return {
        "model_cls": LlavaForConditionalGeneration,
        "processor_loader": lambda pid: LlavaProcessor.from_pretrained(pid),
        "processor_post_init": None,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": False,
    }


def _model_entry_llava_onevision() -> Dict[str, Any]:
    from transformers import AutoProcessor

    if LlavaOnevisionForConditionalGeneration is None:
        raise ImportError(
            "LlavaOnevisionForConditionalGeneration not available. "
            "Upgrade transformers: pip install --upgrade transformers"
        )
    return {
        "model_cls": LlavaOnevisionForConditionalGeneration,
        "processor_loader": lambda pid: AutoProcessor.from_pretrained(pid),
        "processor_post_init": None,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": False,
    }


def _model_entry_blip2() -> Dict[str, Any]:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    return {
        "model_cls": Blip2ForConditionalGeneration,
        "processor_loader": lambda pid: Blip2Processor.from_pretrained(pid),
        "processor_post_init": None,
        "default_lora_targets": [
            "q", "v",  # cross-attention in Q-Former
            "query", "key", "value",  # language model
        ],
        "trust_remote_code": False,
    }


def _model_entry_gemma3() -> Dict[str, Any]:
    from transformers import AutoProcessor

    if Gemma3ForConditionalGeneration is None:
        raise ImportError(
            "Gemma3ForConditionalGeneration not available. "
            "Upgrade transformers: pip install --upgrade transformers"
        )
    return {
        "model_cls": Gemma3ForConditionalGeneration,
        "processor_loader": lambda pid: AutoProcessor.from_pretrained(pid),
        "processor_post_init": None,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": False,
    }


def _internvl_processor_post_init(
    processor: Any,
    image_size: int = 448,
    dynamic_resolution: bool = False,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Any:
    """Configure InternVL image processor patch settings."""
    try:
        if hasattr(processor, "image_processor"):
            if dynamic_resolution:
                processor.image_processor.min_patches = 1
                processor.image_processor.max_patches = min_pixels or 6
                logger.info(
                    "InternVL dynamic patches: min=%d, max=%d",
                    processor.image_processor.min_patches,
                    processor.image_processor.max_patches,
                )
            else:
                processor.image_processor.max_patches = 1
                processor.image_processor.min_patches = 1
    except Exception:
        pass
    return processor


def _patch_internvl_tokenizer(tokenizer: Any) -> Any:
    """Patch Qwen2-based tokenizer with InternVL-specific token attributes.

    transformers >= 4.56 ``InternVLProcessor`` expects named token
    attributes (``start_image_token``, ``end_image_token``, etc.) on
    the tokenizer, but InternVL3.5 ships a plain Qwen2TokenizerFast
    that only registers them as added tokens.
    """
    _TOKEN_ATTRS = {
        "start_image_token": "<img>",
        "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>",
        "video_token": "<|video_pad|>",
    }
    for attr, token_str in _TOKEN_ATTRS.items():
        if not hasattr(tokenizer, attr):
            setattr(tokenizer, attr, token_str)
        id_attr = f"{attr}_id"
        if not hasattr(tokenizer, id_attr):
            tid = tokenizer.convert_tokens_to_ids(token_str)
            setattr(tokenizer, id_attr, tid)
    return tokenizer


def _load_internvl_processor(pretrained_id: str) -> Any:
    """Load InternVL processor with tokenizer patched for transformers >= 4.56."""
    from transformers import AutoImageProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_id, trust_remote_code=True)
    tokenizer = _patch_internvl_tokenizer(tokenizer)

    image_processor = AutoImageProcessor.from_pretrained(pretrained_id, trust_remote_code=True)

    try:
        from transformers import InternVLProcessor

        # InternVLProcessor in transformers >= 4.56 requires video_processor;
        # pass a minimal placeholder when the model has no dedicated one.
        try:
            from transformers import AutoVideoProcessor

            video_processor = AutoVideoProcessor.from_pretrained(pretrained_id, trust_remote_code=True)
        except Exception:
            video_processor = image_processor  # reuse image processor as fallback

        processor = InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
        # Copy chat_template from tokenizer so processor.apply_chat_template works.
        # Replace <image> placeholder with <IMG_CONTEXT> which is what InternVLProcessor expects.
        if not getattr(processor, "chat_template", None) and getattr(tokenizer, "chat_template", None):
            tpl = tokenizer.chat_template
            if "<image>" in tpl and "<IMG_CONTEXT>" not in tpl:
                tpl = tpl.replace("<image>", "<IMG_CONTEXT>")
            processor.chat_template = tpl
    except Exception as exc:
        logger.warning("InternVLProcessor init failed (%s); falling back to manual assembly.", exc)
        # Build a minimal processor-like wrapper manually
        processor = _InternVLProcessorShim(image_processor, tokenizer)

    return processor


class _InternVLProcessorShim:
    """Minimal shim that mimics the HF Processor interface for InternVL training.

    Used as a fallback when the native ``InternVLProcessor`` cannot be
    instantiated (e.g. missing ``video_processor`` or token-attribute
    mismatches across transformers versions).
    """

    def __init__(self, image_processor: Any, tokenizer: Any) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, *, text: Any = None, images: Any = None, **kwargs: Any) -> Any:
        result: Dict[str, Any] = {}
        if images is not None:
            img_out = self.image_processor(images=images, return_tensors=kwargs.get("return_tensors"))
            result.update(img_out)
        if text is not None:
            txt_out = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
                truncation=kwargs.get("truncation", False),
                max_length=kwargs.get("max_length"),
            )
            result.update(txt_out)
        return result

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def _model_entry_internvl() -> Dict[str, Any]:
    """InternVL3 HF-native entry (transformers >= 4.56).

    Uses ``InternVLForConditionalGeneration`` which inherits from
    ``LlavaForConditionalGeneration`` and includes ``GenerationMixin``,
    so ``prepare_inputs_for_generation`` / ``generate`` work out-of-the-box
    with PEFT ``PeftModelForCausalLM``.
    """
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    return {
        "model_cls": InternVLForConditionalGeneration,
        "processor_loader": lambda pid: AutoProcessor.from_pretrained(pid),
        "processor_post_init": _internvl_processor_post_init,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": False,
    }




def _model_entry_internvl_legacy() -> Dict[str, Any]:
    """InternVL2.5 and older — requires trust_remote_code.

    These models are NOT natively integrated in transformers.
    Uses AutoModelForCausalLM with trust_remote_code=True.
    """
    from transformers import AutoModelForCausalLM
    return {
        "model_cls": AutoModelForCausalLM,
        "processor_loader": _load_internvl_processor,
        "processor_post_init": _internvl_processor_post_init,
        "default_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": True,
    }

# Maps model_type aliases → canonical factory function
_REGISTRY: Dict[str, Any] = {
    "qwen_vl": _model_entry_qwen25_vl,
    "qwen25_vl": _model_entry_qwen25_vl,
    "qwenvl": _model_entry_qwen25_vl,
    "qwen2_vl": _model_entry_qwen2_vl,
    # ERP-RoPE (Dense360 §4.1) variant — same Qwen2.5-VL model loader, but the
    # PanoAdapt PE side swaps in width-axis γ·f(w) M-RoPE override at runtime
    # (see panoadapt_erp_rope.py). Listed in the model registry so config
    # validation accepts model_type='qwen_vl_erp_rope'.
    "qwen_vl_erp_rope": _model_entry_qwen25_vl,
    "qwen25_vl_erp_rope": _model_entry_qwen25_vl,
    "llava": _model_entry_llava,
    "llava_next": _model_entry_llava,
    "llava_onevision": _model_entry_llava_onevision,
    "onevision": _model_entry_llava_onevision,
    "blip2": _model_entry_blip2,
    "blip-2": _model_entry_blip2,
    "gemma3": _model_entry_gemma3,
    "gemma-3": _model_entry_gemma3,
    "internvl": _model_entry_internvl,
    "internvl_chat": _model_entry_internvl,
    "internvl25": _model_entry_internvl_legacy,
    "internvl_legacy": _model_entry_internvl_legacy,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BaselineModelRegistry:
    """Registry for loading commercial VLMs for LoRA finetuning."""

    @staticmethod
    def list_models() -> List[str]:
        """Return canonical model type names (deduplicated)."""
        seen: Dict[int, str] = {}
        for alias, factory in _REGISTRY.items():
            fid = id(factory)
            if fid not in seen:
                seen[fid] = alias
        return sorted(seen.values())

    @staticmethod
    def get_default_lora_targets(model_type: str) -> List[str]:
        """Return default LoRA target module names for *model_type*."""
        key = model_type.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {BaselineModelRegistry.list_models()}"
            )
        entry = _REGISTRY[key]()
        return list(entry["default_lora_targets"])

    @staticmethod
    def load_model(
        config: BaselineModelConfig,
        device: str = "auto",
    ) -> Tuple[torch.nn.Module, Any, Any]:
        """Load a pretrained VLM.

        Parameters
        ----------
        config:
            Model configuration specifying HF model ID and type.
        device:
            Target device – ``"auto"`` lets HF place on available GPU.

        Returns
        -------
        tuple of (model, processor, tokenizer)
        """
        key = config.model_type.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                f"Supported: {BaselineModelRegistry.list_models()}"
            )

        entry = _REGISTRY[key]()
        dtype = _resolve_dtype(config.dtype)
        processor_id = config.processor_id or config.hf_model_id

        _apply_erp_resize_pixel_lock(config)

        # Load processor
        processor = entry["processor_loader"](processor_id)
        post_init = entry.get("processor_post_init")
        if post_init is not None:
            processor = post_init(
                processor,
                image_size=config.image_size,
                dynamic_resolution=config.dynamic_resolution,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            )

        # Load model
        model_cls = entry["model_cls"]
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
        }
        if entry.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True
        if config.attn_implementation:
            load_kwargs["attn_implementation"] = config.attn_implementation

        model = model_cls.from_pretrained(config.hf_model_id, **load_kwargs)

        # Model-specific post-init (e.g. InternVL forward patching)
        model_post_init = entry.get("model_post_init")
        if model_post_init is not None:
            model = model_post_init(model)

        # Disable KV-cache for training
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        # Extract tokenizer from processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(processor_id)

        # Ensure pad token is set
        if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            "Loaded model=%s (type=%s, dtype=%s)",
            config.hf_model_id,
            config.model_type,
            dtype,
        )
        return model, processor, tokenizer

    @staticmethod
    def load_processor(
        config: BaselineModelConfig,
    ) -> Tuple[Any, Any]:
        """Load the processor/tokenizer path used by baseline experiments."""
        key = config.model_type.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                f"Supported: {BaselineModelRegistry.list_models()}"
            )

        entry = _REGISTRY[key]()
        processor_id = config.processor_id or config.hf_model_id

        _apply_erp_resize_pixel_lock(config)

        processor = entry["processor_loader"](processor_id)
        post_init = entry.get("processor_post_init")
        if post_init is not None:
            processor = post_init(
                processor,
                image_size=config.image_size,
                dynamic_resolution=config.dynamic_resolution,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            )

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(processor_id)

        if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            "Loaded processor=%s (type=%s)",
            processor_id,
            config.model_type,
        )
        return processor, tokenizer
