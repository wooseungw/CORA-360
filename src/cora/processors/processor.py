"""
Unified multimodal processor for CORA.

Combines image processing (PanoramaImageProcessor) and text formatting
(UniversalTextFormatter) with tokenization into a single callable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from PIL import Image
from transformers import BatchEncoding

from cora.processors.images import PanoramaImageProcessor
from cora.processors.text import UniversalTextFormatter


class PanoramaProcessor:
    """Multimodal processor: image views + text tokenization."""

    def __init__(
        self,
        image_processor: PanoramaImageProcessor,
        tokenizer: Any,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.text_formatter = UniversalTextFormatter(tokenizer, system_msg=system_prompt)

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: Optional[str] = "pt",
    ) -> BatchEncoding:
        """Process text and optional images into model-ready tensors."""

        pixel_values = None
        if images is not None:
            pixel_values = self.image_processor(images)

        if isinstance(text, str):
            text = [text]

        tokenized = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        data = dict(tokenized)
        if pixel_values is not None:
            data["pixel_values"] = pixel_values

        return BatchEncoding(data, tensor_type=return_tensors)

    def batch_decode(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(*args, **kwargs)
