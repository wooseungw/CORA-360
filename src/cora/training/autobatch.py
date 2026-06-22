"""YOLO-style GPU memory profiling for automatic batch size selection.

Profiles batch sizes [1, 2, 4, 8, 16] using a deep-copied probe model,
measures peak VRAM per batch, and extrapolates the optimal batch size
that fits within ``fraction`` of total GPU memory.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def autobatch(
    module: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable,
    fraction: float = 0.70,
    default_batch_size: int = 1,
    training_step_fn: Optional[Callable] = None,
) -> int:
    """Find the largest batch size that fits in GPU memory.

    Args:
        module: The full training module (e.g. ``PanoramaTrainingModule``).
        dataset: Training dataset to sample batches from.
        collate_fn: Collate function matching the DataLoader.
        fraction: Target fraction of total VRAM to use (0.0–1.0).
        default_batch_size: Fallback if profiling fails.
        training_step_fn: Optional callable ``(module, batch) -> loss``.
            If None, falls back to ``module(**batch)`` and extracts loss.

    Returns:
        Optimal batch size (≥ 1).
    """
    if not torch.cuda.is_available():
        logger.warning("AutoBatch: CUDA unavailable, batch_size=%d", default_batch_size)
        return default_batch_size

    device = torch.device("cuda")
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / gb

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    probe = deepcopy(module).to(device).train()

    allocated_after_model = torch.cuda.memory_allocated(device) / gb
    free = total - allocated_after_model

    logger.info(
        "AutoBatch: %s %.1fG total, %.1fG model, %.1fG free (%.0f%% target)",
        props.name, total, allocated_after_model, free, fraction * 100,
    )

    batch_sizes = [1, 2, 4, 8, 16]
    mem_usage: list[tuple[int, float]] = []

    for bs in batch_sizes:
        if bs > len(dataset):
            break
        try:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            batch = collate_fn([dataset[i % len(dataset)] for i in range(bs)])
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda"):
                if training_step_fn is not None:
                    loss = training_step_fn(probe, batch)
                else:
                    result = probe(**batch)
                    loss = (
                        result.loss
                        if hasattr(result, "loss")
                        else result.get("loss", result[0])
                    )
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()

            peak = torch.cuda.max_memory_allocated(device) / gb
            mem_usage.append((bs, peak))
            probe.zero_grad(set_to_none=True)
            del batch, loss
            torch.cuda.empty_cache()
            logger.info("  batch %2d -> %.2fG peak", bs, peak)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                probe.zero_grad(set_to_none=True)
                logger.info("  batch %2d -> OOM", bs)
                break
            raise

    del probe
    torch.cuda.empty_cache()

    if len(mem_usage) < 2:
        logger.warning(
            "AutoBatch: insufficient data points, batch_size=%d", default_batch_size
        )
        return default_batch_size

    xs, ys = zip(*mem_usage)
    p = np.polyfit(xs, ys, deg=1)
    optimal = int((free * fraction - p[1]) / p[0])

    oom_limit = (
        batch_sizes[len(mem_usage)]
        if len(mem_usage) < len(batch_sizes)
        else batch_sizes[-1] * 2
    )
    optimal = min(optimal, oom_limit - 1)
    optimal = max(optimal, 1)

    logger.info(
        "AutoBatch: optimal batch_size = %d (%.1fG / %.1fG, %.0f%%)",
        optimal, np.polyval(p, optimal), total, fraction * 100,
    )
    return optimal
