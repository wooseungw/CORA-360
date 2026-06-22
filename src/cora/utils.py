"""Utility functions for CORA.

Provides helper functions used across the model package, including
spatial grid inference and cached dtype resolution for mixed-precision training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["infer_hw", "resolve_module_dtype"]


def infer_hw(num_patches: int) -> Tuple[int, int]:
    """Infer (height, width) from a total patch count assuming near-square grid.

    Tries the exact integer square root first, then expands minimally so that
    ``h * w >= num_patches``.

    Args:
        num_patches: Total number of spatial patches/tokens.

    Returns:
        Tuple ``(height, width)`` satisfying ``h * w >= num_patches``.

    Raises:
        ValueError: If *num_patches* is not a positive integer.
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}")
    h = w = int(np.sqrt(num_patches))
    while h * w < num_patches:
        if h <= w:
            h += 1
        else:
            w += 1
    return h, w


def resolve_module_dtype(
    cache: Dict[str, torch.dtype],
    cache_key: str,
    module: Optional[nn.Module],
    default: Optional[torch.dtype] = None,
) -> Optional[torch.dtype]:
    """Cache-aware dtype resolution for a module's parameters or buffers.

    Inspects the first parameter (or buffer if no parameters exist) of *module*
    to determine its dtype, caching the result under *cache_key* for subsequent
    lookups.  This avoids repeated iteration during the forward pass in
    mixed-precision training scenarios.

    Args:
        cache: Mutable ``{key: dtype}`` dict shared across the model.
        cache_key: Lookup key for this module.
        module: The ``nn.Module`` to inspect (may be ``None``).
        default: Fallback dtype when the module has no parameters/buffers.

    Returns:
        Resolved ``torch.dtype`` or ``None`` if no dtype could be determined.
    """
    if cache_key in cache:
        return cache[cache_key]
    if module is None:
        return default

    dtype: Optional[torch.dtype] = None

    # Try parameters first
    try:
        param = next(module.parameters())
        dtype = param.dtype
    except (StopIteration, AttributeError):
        pass

    # Fall back to buffers
    if dtype is None:
        try:
            buf = next(module.buffers())
            dtype = buf.dtype
        except (StopIteration, AttributeError):
            pass

    if dtype is None:
        dtype = default

    if dtype is not None:
        cache[cache_key] = dtype
    return dtype
