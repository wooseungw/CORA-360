"""Evaluation module: BLEU-4, CIDEr, METEOR, SPICE, ROUGE-L metrics + LLM-as-Judge."""

from cora.evaluation.judge_prompts import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    JUDGE_PROMPT,
    PAIRWISE_JUDGE_PROMPT,
    compute_weighted_score,
)
from cora.evaluation.metrics import CORAEvaluator

__all__ = [
    "CORAEvaluator",
    "JUDGE_PROMPT",
    "PAIRWISE_JUDGE_PROMPT",
    "DIMENSION_WEIGHTS",
    "DIMENSIONS",
    "compute_weighted_score",
]
