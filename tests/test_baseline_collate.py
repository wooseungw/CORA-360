"""Unit tests for baseline collate functions and _SafeTrainer."""

import torch
from unittest.mock import MagicMock

import pytest


def _make_mock_processor(seq_len=20, prompt_len=10):
    """Mock processor returning fixed-size tensors."""

    def _call(text=None, images=None, return_tensors=None, padding=None, **kw):
        batch_size = len(text) if text else 1
        if not padding:
            # single-sample call (for prompt_len measurement)
            return {
                "input_ids": torch.ones(1, prompt_len, dtype=torch.long),
                "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }
        # batch call
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long) * 5,
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

    mock = MagicMock(side_effect=_call)
    return mock


def _make_mock_tokenizer(pad_token_id=0):
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    return tok


def _make_features(n=2):
    return [
        {"full_text": f"full text {i}", "prompt_text": f"prompt {i}", "image": MagicMock()}
        for i in range(n)
    ]


# ---- Tests ----------------------------------------------------------------


class TestCollateCausal:
    def test_no_truncation(self):
        """input_ids should NOT be truncated when max_length=None."""
        from cora.baseline.finetune import _collate_causal

        collate = _collate_causal(
            processor=_make_mock_processor(seq_len=30),
            tokenizer=_make_mock_tokenizer(),
            max_length=None,
        )
        batch = collate(_make_features(2))
        assert batch["input_ids"].shape[1] == 30

    def test_truncation_applied(self):
        """input_ids shape[1] should be <= max_length."""
        from cora.baseline.finetune import _collate_causal

        collate = _collate_causal(
            processor=_make_mock_processor(seq_len=30),
            tokenizer=_make_mock_tokenizer(),
            max_length=10,
        )
        batch = collate(_make_features(2))
        assert batch["input_ids"].shape[1] <= 10

    def test_labels_prompt_masked(self):
        """Prompt portion of labels should be -100."""
        from cora.baseline.finetune import _collate_causal

        prompt_len = 8
        collate = _collate_causal(
            processor=_make_mock_processor(seq_len=20, prompt_len=prompt_len),
            tokenizer=_make_mock_tokenizer(pad_token_id=-999),  # won't match any token
            max_length=None,
        )
        batch = collate(_make_features(1))
        labels = batch["labels"][0]
        # first prompt_len tokens should be -100
        assert (labels[:prompt_len] == -100).all()
        # remaining tokens should NOT be -100
        assert (labels[prompt_len:] != -100).any()

    def test_truncation_before_masking(self):
        """With small max_length > prompt_len, labels should have some non-masked tokens."""
        from cora.baseline.finetune import _collate_causal

        prompt_len = 5
        max_length = 15
        collate = _collate_causal(
            processor=_make_mock_processor(seq_len=30, prompt_len=prompt_len),
            tokenizer=_make_mock_tokenizer(pad_token_id=-999),
            max_length=max_length,
        )
        batch = collate(_make_features(1))
        labels = batch["labels"][0]
        assert labels.shape[0] == max_length
        # Should have some non-masked tokens (response tokens)
        non_masked = (labels != -100).sum().item()
        assert non_masked > 0, "All labels are -100; truncation may have happened after masking"

    def test_pixel_values_not_truncated(self):
        """pixel_values should be passed through unchanged."""
        from cora.baseline.finetune import _collate_causal

        collate = _collate_causal(
            processor=_make_mock_processor(seq_len=30),
            tokenizer=_make_mock_tokenizer(),
            max_length=10,
        )
        batch = collate(_make_features(1))
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[-1] == 224  # image dim unchanged


class TestSafeTrainer:
    def test_is_trainer_subclass(self):
        """_SafeTrainer must be a subclass of transformers.Trainer."""
        from cora.baseline.finetune import _SafeTrainer
        from transformers import Trainer

        assert issubclass(_SafeTrainer, Trainer)
