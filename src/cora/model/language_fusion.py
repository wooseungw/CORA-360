"""Vision-language token fusion for CORA.

Provides :class:`LanguageFusion`, a utility that stitches projected vision
embeddings into the text-token stream at ``<|vision|>`` placeholder positions.
Handles batch-size adjustment, padding, and label masking for training.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..processors.text import UniversalTextFormatter

__all__ = ["LanguageFusion"]


class LanguageFusion:
    """Utility for fusing vision embeddings with language token embeddings.

    Replaces ``<|vision|>`` placeholder tokens in the text stream with projected
    vision embeddings, handling variable-length sequences via right-padding.

    Args:
        language_model: The underlying causal language model (used for its
            input embedding layer).
        tokenizer: Associated tokenizer instance.
        vision_token_id: Token ID of the ``<|vision|>`` placeholder.
        ignore_index: Label value for masked positions (default ``-100``).
        max_text_length: Maximum text sequence length before truncation.
        system_prompt: Optional system prompt for the text formatter.
        default_generation_prompt: Fallback user prompt when none is provided
            during generation.
        formatter: Pre-configured :class:`UniversalTextFormatter`; one is
            created automatically if not supplied.
        fallback_insert_position: Where to insert vision tokens when no
            ``<|vision|>`` placeholder is found (``"prefix"`` or ``"suffix"``).
    """

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: object,
        vision_token_id: int,
        ignore_index: int = -100,
        max_text_length: int = 2048,
        system_prompt: Optional[str] = None,
        default_generation_prompt: Optional[str] = None,
        formatter: Optional[UniversalTextFormatter] = None,
        fallback_insert_position: str = "prefix",
    ) -> None:
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.vision_token_id = vision_token_id
        self.ignore_index = ignore_index
        self.max_text_length = max_text_length
        self.default_generation_prompt = (
            default_generation_prompt or "Describe the panoramic scene."
        )

        if fallback_insert_position not in {"prefix", "suffix"}:
            fallback_insert_position = "prefix"
        self.fallback_insert_position = fallback_insert_position

        self.text_formatter = formatter or UniversalTextFormatter(
            tokenizer,
            system_msg=system_prompt,
        )

    # ------------------------------------------------------------------
    # Configuration updates
    # ------------------------------------------------------------------

    def update_vision_token_id(self, vision_token_id: int) -> None:
        """Update the vision placeholder token ID."""
        self.vision_token_id = vision_token_id

    def update_max_text_length(self, max_text_length: int) -> None:
        """Update the maximum text length for truncation."""
        self.max_text_length = max_text_length

    # ------------------------------------------------------------------
    # Public helpers used by PanoramaVLM
    # ------------------------------------------------------------------

    def prepare_generation_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare or default-create generation input tensors.

        If *input_ids* is ``None``, creates a default prompt using the text
        formatter.  Otherwise normalises shapes and adjusts to the target
        batch size.
        """
        if input_ids is None:
            input_ids, attention_mask = self.create_default_prompt(batch_size, device)
        else:
            input_ids = self._ensure_2d(input_ids).to(device)
            if attention_mask is None:
                attention_mask = (input_ids != self._pad_token_id()).long()
            attention_mask = self._ensure_2d(attention_mask).to(device)
            input_ids, attention_mask = self.adjust_batch_size(
                input_ids, attention_mask, batch_size,
            )

        input_ids = self._trim_tensor(input_ids)
        attention_mask = self._trim_tensor(attention_mask)
        return input_ids, attention_mask

    def create_default_prompt(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a tokenised default prompt for generation.

        Uses :meth:`UniversalTextFormatter.tokenize_for_generation` if
        available, otherwise falls back to :meth:`format_conversation` +
        manual tokenisation.
        """
        if hasattr(self.text_formatter, "tokenize_for_generation"):
            tokenized = self.text_formatter.tokenize_for_generation(
                user_msg=self.default_generation_prompt,
                max_length=self.max_text_length,
            )
        else:
            # Fallback: format + tokenize manually
            prompt_text = self.text_formatter.format_conversation(
                user_msg=self.default_generation_prompt,
                add_generation_prompt=True,
            )
            tokenized = self.tokenizer(
                prompt_text,
                max_length=self.max_text_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        input_ids = self._ensure_2d(input_ids)
        attention_mask = self._ensure_2d(attention_mask)
        input_ids, attention_mask = self.adjust_batch_size(
            input_ids, attention_mask, batch_size,
        )
        return input_ids, attention_mask

    def adjust_batch_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicate or trim tensors to match *target_batch_size*."""
        current = input_ids.size(0)
        if current == target_batch_size:
            return input_ids, attention_mask
        if current == 1 and target_batch_size > 1:
            return (
                input_ids.repeat(target_batch_size, 1),
                attention_mask.repeat(target_batch_size, 1),
            )
        if current > target_batch_size:
            return input_ids[:target_batch_size], attention_mask[:target_batch_size]

        repeat_factor = math.ceil(target_batch_size / current)
        input_ids = input_ids.repeat(repeat_factor, 1)[:target_batch_size]
        attention_mask = attention_mask.repeat(repeat_factor, 1)[:target_batch_size]
        return input_ids, attention_mask

    def prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Embed text tokens and prepare a dict suitable for fusion.

        Returns:
            Dict with ``"input_ids"``, ``"attention_mask"``, ``"inputs_embeds"``,
            and optionally ``"labels"``.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for text preparation")

        input_ids = self._ensure_2d(input_ids)
        if attention_mask is None:
            attention_mask = (input_ids != self._pad_token_id()).long()
        attention_mask = self._ensure_2d(attention_mask)
        if labels is not None:
            labels = self._ensure_2d(labels)

        input_ids = self._trim_tensor(input_ids)
        attention_mask = self._trim_tensor(attention_mask)
        if labels is not None:
            labels = self._trim_tensor(labels)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
        if labels is not None:
            result["labels"] = labels
        return result

    def fuse(
        self,
        vision_tokens: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse vision embeddings into the text-token stream.

        For each sample in the batch, ``<|vision|>`` placeholders in
        *input_ids* are replaced with the corresponding row of
        *vision_tokens*.  If no placeholder is found, vision tokens are
        inserted at the configured fallback position.

        Args:
            vision_tokens: ``[B, T_vis, D]`` projected vision embeddings.
            input_ids: ``[B, L]`` text token IDs (used if *text_inputs* is
                ``None``).
            attention_mask: ``[B, L]`` attention mask.
            labels: ``[B, L]`` training labels.
            text_inputs: Pre-computed text inputs dict (from
                :meth:`prepare_text_inputs`).

        Returns:
            Dict with ``"inputs_embeds"``, ``"attention_mask"``, and
            optionally ``"labels"`` — all padded to the same length.
        """
        if text_inputs is None:
            text_inputs = self.prepare_text_inputs(input_ids, attention_mask, labels)

        embeds = text_inputs["inputs_embeds"]
        attn = text_inputs["attention_mask"]
        ids = text_inputs["input_ids"]
        lbls = text_inputs.get("labels")

        batch_size = embeds.size(0)
        fused_embeds: List[torch.Tensor] = []
        fused_attn: List[torch.Tensor] = []
        fused_labels: Optional[List[torch.Tensor]] = [] if lbls is not None else None
        num_vision_tokens = vision_tokens.size(1)

        for idx in range(batch_size):
            text_embed = embeds[idx]
            text_attn = attn[idx]
            text_ids = ids[idx]
            text_labels = lbls[idx] if lbls is not None else None

            vision_mask = text_ids == self.vision_token_id
            has_vision_token = torch.any(vision_mask)

            if has_vision_token:
                first_position = int(torch.nonzero(vision_mask, as_tuple=False)[0].item())
                filtered_embed = text_embed[~vision_mask]
                filtered_attn = text_attn[~vision_mask]
                filtered_labels = text_labels[~vision_mask] if text_labels is not None else None
                tokens_before = int((~vision_mask[:first_position]).sum().item())
                insert_pos = tokens_before
            else:
                filtered_embed = text_embed
                filtered_attn = text_attn
                filtered_labels = text_labels
                insert_pos = (
                    0 if self.fallback_insert_position == "prefix"
                    else filtered_embed.size(0)
                )

            # Concatenate: [prefix | vision | suffix]
            prefix_embed = filtered_embed[:insert_pos]
            suffix_embed = filtered_embed[insert_pos:]
            combined_embed = torch.cat(
                [prefix_embed, vision_tokens[idx], suffix_embed], dim=0,
            )

            prefix_attn = filtered_attn[:insert_pos]
            suffix_attn = filtered_attn[insert_pos:]
            vision_attn = torch.ones(
                num_vision_tokens, dtype=filtered_attn.dtype, device=vision_tokens.device,
            )
            combined_attn = torch.cat([prefix_attn, vision_attn, suffix_attn], dim=0)

            fused_embeds.append(combined_embed)
            fused_attn.append(combined_attn)

            if fused_labels is not None and filtered_labels is not None:
                prefix_labels = filtered_labels[:insert_pos]
                suffix_labels = filtered_labels[insert_pos:]
                vision_labels = torch.full(
                    (num_vision_tokens,),
                    self.ignore_index,
                    dtype=filtered_labels.dtype,
                    device=vision_tokens.device,
                )
                fused_labels.append(
                    torch.cat([prefix_labels, vision_labels, suffix_labels], dim=0)
                )

        padded_embeds, padded_attn, padded_labels = self._pad_fused_sequences(
            fused_embeds, fused_attn, fused_labels, vision_tokens.size(-1),
        )

        output: Dict[str, torch.Tensor] = {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_attn,
        }
        if padded_labels is not None:
            output["labels"] = padded_labels
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_fused_sequences(
        self,
        fused_embeds: List[torch.Tensor],
        fused_attn: List[torch.Tensor],
        fused_labels: Optional[List[torch.Tensor]],
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Right-pad variable-length fused sequences to a common length."""
        target_len = max(seq.size(0) for seq in fused_embeds)

        padded_embeds: List[torch.Tensor] = []
        padded_attn: List[torch.Tensor] = []
        padded_labels_list: Optional[List[torch.Tensor]] = (
            [] if fused_labels is not None else None
        )

        for idx, embed_seq in enumerate(fused_embeds):
            pad_amount = target_len - embed_seq.size(0)
            if pad_amount > 0:
                pad_embed = torch.zeros(
                    (pad_amount, embed_dim),
                    dtype=embed_seq.dtype,
                    device=embed_seq.device,
                )
                pad_attn = torch.zeros(
                    pad_amount,
                    dtype=fused_attn[idx].dtype,
                    device=fused_attn[idx].device,
                )
                embed_seq = torch.cat([embed_seq, pad_embed], dim=0)
                fused_attn[idx] = torch.cat([fused_attn[idx], pad_attn], dim=0)
                if padded_labels_list is not None and fused_labels is not None:
                    pad_labels = torch.full(
                        (pad_amount,),
                        self.ignore_index,
                        dtype=fused_labels[idx].dtype,
                        device=fused_labels[idx].device,
                    )
                    fused_labels[idx] = torch.cat([fused_labels[idx], pad_labels], dim=0)
            padded_embeds.append(embed_seq)
            padded_attn.append(fused_attn[idx])
            if padded_labels_list is not None and fused_labels is not None:
                padded_labels_list.append(fused_labels[idx])

        stacked_embeds = torch.stack(padded_embeds, dim=0)
        stacked_attn = torch.stack(padded_attn, dim=0)
        stacked_labels = (
            torch.stack(padded_labels_list, dim=0)
            if padded_labels_list is not None
            else None
        )
        return stacked_embeds, stacked_attn, stacked_labels

    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is at least 2-D ``[B, L]``."""
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _trim_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Truncate sequence dimension to :attr:`max_text_length`."""
        if self.max_text_length is None:
            return tensor
        if tensor.size(1) <= self.max_text_length:
            return tensor
        return tensor[:, : self.max_text_length]

    def _pad_token_id(self) -> int:
        """Resolve the pad token ID from the tokenizer."""
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return pad_id
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return eos_id
        return 0
