"""Regression tests for PanoRoPE position-id shifts.

These guard against the prior bug where the shift was applied with the wrong
sign / factor (`+ tile_idx * stride * T` instead of `- tile_idx * overlap * T`),
which pushed adjacent views *further apart* in PID space rather than
overlapping them.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

# Load panoadapt directly to avoid pulling in the broader baseline package
# (which has heavyweight optional deps like pandas).
_SPEC = importlib.util.spec_from_file_location(
    "_panoadapt_for_test", "src/cora/baseline/panoadapt.py",
)
_PA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PA)


class _Stub1D(_PA._PanoRoPE1DAdapter):
    _IMAGE_TOKEN_ID = 99
    _SPATIAL_MERGE_SIZE = 1

    def get_vision_hook_target(self) -> str:
        return "x"

    def get_image_token_id(self, model) -> int:  # noqa: ARG002
        return 99


def _seq_two_views(view_tokens: int = 4) -> torch.Tensor:
    """Build a sequence with two image-view groups separated by a non-image token.

    Layout: ``[text, text, <view0 image tokens>, sep, <view1 image tokens>, text]``
    """
    return torch.tensor(
        [[1, 2] + [99] * view_tokens + [1] + [99] * view_tokens + [3]]
    )


def test_1d_shift_is_negative_for_overlap_05():
    """View 1 should pull *back* by overlap*T, not push forward."""
    seq = _seq_two_views(4)
    adapter = _Stub1D(overlap_ratio=0.5, include_global=False)
    pids = torch.arange(seq.shape[1]).unsqueeze(0).clone()
    out = adapter.modify_position_ids(pids, seq, None, None)

    # view 0 PIDs unchanged: positions 2..5 → [2, 3, 4, 5]
    # view 1 PIDs originally [7..10]; shift -2 → [5, 6, 7, 8]
    expected = [0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 11]
    assert out[0].tolist() == expected


def test_1d_shift_is_zero_for_overlap_zero():
    """overlap_ratio=0 must leave PIDs identical to the default arange."""
    seq = _seq_two_views(4)
    adapter = _Stub1D(overlap_ratio=0.0, include_global=False)
    pids = torch.arange(seq.shape[1]).unsqueeze(0).clone()
    out = adapter.modify_position_ids(pids, seq, None, None)
    assert out[0].tolist() == list(range(seq.shape[1]))


def test_1d_shift_quarter_overlap():
    seq = _seq_two_views(4)
    adapter = _Stub1D(overlap_ratio=0.25, include_global=False)
    pids = torch.arange(seq.shape[1]).unsqueeze(0).clone()
    out = adapter.modify_position_ids(pids, seq, None, None)
    # shift = -round(1 * 0.25 * 4) = -1
    expected = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 11]
    assert out[0].tolist() == expected


def test_qwen_width_axis_pulled_back_for_overlap_05():
    """Qwen 3-D M-RoPE width axis must overlap, not separate, adjacent tiles."""
    adapter = _PA.QwenVLAdapter(overlap_ratio=0.5, include_global=False)

    class _Cfg:
        image_token_id = 99

        class vision_config:  # noqa: D106
            spatial_merge_size = 1

    class _Model:
        config = _Cfg()

    seq = torch.tensor([[99] * 8])  # 8 image tokens, two views of 4
    pids = torch.zeros(3, 1, 8, dtype=torch.long)
    pids[2, 0] = torch.arange(8)  # width axis: view 0 [0..3], view 1 [4..7]
    grid = torch.tensor([[1, 1, 4], [1, 1, 4]])

    out = adapter.modify_position_ids(pids, seq, grid, _Model())
    # overlap=0.5 → view 1 width PIDs shifted by -2 → [2, 3, 4, 5]
    assert out[2, 0].tolist() == [0, 1, 2, 3, 2, 3, 4, 5]


def test_qwen_width_axis_unchanged_for_overlap_zero():
    adapter = _PA.QwenVLAdapter(overlap_ratio=0.0, include_global=False)

    class _Cfg:
        image_token_id = 99

        class vision_config:  # noqa: D106
            spatial_merge_size = 1

    class _Model:
        config = _Cfg()

    seq = torch.tensor([[99] * 8])
    pids = torch.zeros(3, 1, 8, dtype=torch.long)
    pids[2, 0] = torch.arange(8)
    grid = torch.tensor([[1, 1, 4], [1, 1, 4]])
    out = adapter.modify_position_ids(pids, seq, grid, _Model())
    assert out[2, 0].tolist() == list(range(8))


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_1d_three_view_monotonic_back_shift(overlap: float):
    """With V views, view i's first PID must monotonically decrease in the
    overlap shift, never increase, for any overlap_ratio in [0, 1)."""
    V, T = 3, 4
    seq = torch.tensor([[1] + sum(([99] * T + [1] for _ in range(V)), [])[:-1] + [3]])
    adapter = _Stub1D(overlap_ratio=overlap, include_global=False)
    pids = torch.arange(seq.shape[1]).unsqueeze(0).clone()
    out = adapter.modify_position_ids(pids, seq, None, None)

    # Locate first PID of each view group and ensure shift = -round(i*overlap*T).
    image_mask = seq[0] == 99
    image_idx = image_mask.nonzero(as_tuple=True)[0]
    diffs = image_idx[1:] - image_idx[:-1]
    splits = (diffs > 1).nonzero(as_tuple=True)[0] + 1
    groups = torch.tensor_split(image_idx, splits.tolist())

    for tile_idx, view_pos in enumerate(groups):
        first_pid = out[0, view_pos[0]].item()
        original = view_pos[0].item()
        expected_shift = -round(tile_idx * overlap * T)
        assert first_pid - original == expected_shift, (
            f"tile={tile_idx} overlap={overlap}: shift={first_pid - original}, "
            f"expected {expected_shift}"
        )
