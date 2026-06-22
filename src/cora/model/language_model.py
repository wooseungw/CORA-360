"""Language model wrapper with LoRA support for Panorama VLM.

Wraps a HuggingFace causal language model (Qwen, Llama, Gemma, etc.) with
automatic tokenizer setup, vision-token registration, and optional LoRA
fine-tuning via PEFT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

__all__ = ["LanguageModel"]

logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    """HuggingFace causal LM wrapper with tokenizer management and LoRA support.

    On construction the wrapper:

    1. Loads the pretrained model with SDPA attention.
    2. Ensures the tokenizer has ``pad_token`` and a ``<|vision|>`` special token.
    3. Resizes embeddings if new tokens were added.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``).
        use_cache: Whether to enable KV-cache in the underlying model config.
    """

    def __init__(self, model_name: str, use_cache: bool = False) -> None:
        super().__init__()
        self.model_name = model_name

        # Build load kwargs
        load_kwargs: Dict[str, Any] = {
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }

        # Disable <think> for reasoning models (DeepSeek-R1, QwQ, etc.)
        lm_name_lower = model_name.lower()
        if any(kw in lm_name_lower for kw in ("deepseek-r1", "qwq", "r1-distill")):
            load_kwargs["enable_think"] = False
            logger.info("Disabled 'think' mode for reasoning model: %s", model_name)

        logger.info("Loading language model: %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if hasattr(self.model, "config"):
            self.model.config.use_cache = use_cache

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._setup_tokenizer()

    # ------------------------------------------------------------------
    # Tokenizer setup
    # ------------------------------------------------------------------

    def _setup_tokenizer(self) -> None:
        """Ensure tokenizer has pad, eos, and ``<|vision|>`` special tokens."""
        tokens_added = False

        # 1. Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token = eos_token: '%s'", self.tokenizer.eos_token)
            else:
                self.tokenizer.add_special_tokens(
                    {"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}
                )
                tokens_added = True
                logger.info("Added eos_token and pad_token: '<|endoftext|>'")

        # 2. Add vision placeholder token
        vision_token = "<|vision|>"
        existing_specials = [str(t) for t in self.tokenizer.additional_special_tokens]
        if vision_token not in existing_specials:
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [vision_token]}
            )
            if added > 0:
                tokens_added = True
                logger.info("Added special token: %s", vision_token)

        if tokens_added:
            self.resize_token_embeddings()

        self.tokenizer.padding_side = "right"

    def resize_token_embeddings(self) -> None:
        """Resize model embeddings to match the tokenizer vocabulary size."""
        old_size = self.model.get_input_embeddings().weight.size(0)
        self.model.resize_token_embeddings(len(self.tokenizer))
        new_size = self.model.get_input_embeddings().weight.size(0)
        logger.info("Resized token embeddings: %d -> %d", old_size, new_size)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def setup_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ) -> bool:
        """Apply LoRA adapters to the language model via PEFT.

        Args:
            rank: LoRA rank (*r*).
            alpha: LoRA alpha scaling factor.
            dropout: LoRA dropout probability.
            target_modules: List of module name patterns to adapt. Defaults to
                common linear projection names.

        Returns:
            ``True`` if LoRA was successfully applied, ``False`` otherwise.
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. LoRA setup skipped.")
            return False

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        try:
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            return True
        except Exception as e:
            logger.error("Failed to setup LoRA: %s", e)
            return False

    def load_lora_weights(self, path: str) -> bool:
        """Load LoRA adapter weights from disk.

        Args:
            path: Path to the saved PEFT adapter directory.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. Cannot load LoRA weights.")
            return False

        logger.info("Loading LoRA weights from %s", path)
        try:
            if hasattr(self.model, "load_adapter"):
                self.model.load_adapter(path, adapter_name="default")
                self.model.set_adapter("default")
            else:
                self.model = PeftModel.from_pretrained(self.model, path)
            return True
        except Exception as e:
            logger.error("Failed to load LoRA weights: %s", e)
            return False

    # ------------------------------------------------------------------
    # Forward / generate pass-through
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying causal LM."""
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying causal LM's generate method."""
        return self.model.generate(*args, **kwargs)

    @property
    def config(self) -> Any:
        """Access the underlying model configuration."""
        return self.model.config
