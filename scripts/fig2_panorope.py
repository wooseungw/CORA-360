#!/usr/bin/env python3
"""
Figure 2 — PanoRoPE: Panorama-Aware Position Re-indexing
Paper Method section figure.

KEY VISUAL STORY
  (a) Standard VLM: each view independently assigns local column IDs 0…W-1
      *and* local row IDs 0…H-1.  Overlapping spatial regions get DIFFERENT
      2-D position IDs → colour discontinuity at view boundaries.
  (b) PanoRoPE: global continuous positions derived from panoramic geometry.
      Overlapping spatial regions get the SAME global position → colours match exactly.

Position-ID model (2-D):
  Row (pitch) component — same for both (a) and (b):
      pitch[r]  = r / (H − 1)    ∈ [0, 1]

  Column (yaw) component:
      (a) Standard (enable_continuity=False):
              g_std[v, c] = v + c / W         (view-local index)
              STD_MAX     = (V−1) + (W−1)/W   (= 2.875 for V=3,W=8)
              norm_std    = g_std / STD_MAX    ∈ [0, 1]
      (b) PanoRoPE:
              g[v, c]    = v · s  +  c / W    (fractional global col)
              L_total    = V − (V−1) · r       (= 2.0 for V=3,r=0.5)
              norm_pano  = g / L_total         ∈ [0, 1)

  Combined scalar for colourmapping (weighted blend):
      val = COL_W · col_norm  +  ROW_W · pitch_norm

Implementation reference: src/cora/model/positional.py
  PanoramaPositionalEncoding._yaw_encoding()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# ── Paper-quality rendering settings ──────────────────────────────────────
matplotlib.rcParams.update({
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "font.family":       "serif",
    "mathtext.fontset":  "dejavuserif",
    "figure.dpi":        150,
})

# ── Geometry (mirrors src/cora/model/positional.py) ───────────────────────
V       = 3             # views shown in the figure
H       = 4             # patch rows per view
W       = 8             # patch columns per view
OVERLAP = 0.5           # overlap_ratio  (r)
STRIDE  = 1.0 - OVERLAP             # s = 0.5
K       = int(W * OVERLAP)          # overlap cols per view side = 4
L_TOTAL = V - (V - 1) * OVERLAP     # effective unique span = 2.0

# Blending weights for 2-D → scalar visualisation
COL_W = 0.70   # column (yaw) dominates story
ROW_W = 0.30   # row (pitch) adds visible gradient

# ── Build 3-D arrays: [V, H, W] ───────────────────────────────────────────
# Pitch (row) — identical for both methods
pitch_norm: np.ndarray = np.zeros((V, H, W))
for _r in range(H):
    pitch_norm[:, _r, :] = _r / (H - 1)   # 0.0, 0.33, 0.67, 1.0

# (a) Standard: LOCAL column index, repeated per view
#     each view independently assigns col IDs 0...W-1
std_col_int: np.ndarray = np.zeros((V, H, W))
std_col_norm: np.ndarray = np.zeros((V, H, W))
for _c in range(W):
    std_col_int[:, :, _c]  = _c              # integer label 0..7
    std_col_norm[:, :, _c] = _c / (W - 1)   # normalised [0,1]

# (b) PanoRoPE: g[v,c] = v*s + c/W,  normalised by L_total
pano_col: np.ndarray = np.zeros((V, H, W))
for _v in range(V):
    for _c in range(W):
        pano_col[_v, :, _c] = (_v * STRIDE + _c / W) / L_TOTAL
# pano_col already in [0, 1)

# ── Combine into single scalar ∈ [0, 1] ──────────────────────────────────
std_ids  = COL_W * std_col_norm + ROW_W * pitch_norm   # (a)
pano_ids = COL_W * pano_col     + ROW_W * pitch_norm   # (b)

# ── Verify key invariant: right overlap of view v  ==  left overlap of v+1
for _v in range(V - 1):
    right = pano_ids[_v,  :, W - K:]   # last K cols of view v
    left  = pano_ids[_v + 1, :, :K]    # first K cols of view v+1
    assert np.allclose(right, left), f"PanoRoPE invariant violated at v={_v}"

# ── Colour scheme ──────────────────────────────────────────────────────────
CMAP       = matplotlib.colormaps["plasma"]
NORM       = Normalize(vmin=0.0, vmax=1.0)   # shared [0,1]

C_OVL      = "#C62828"    # overlap region  (dark red)
C_MATCH    = "#00695C"    # same-pos annotation (dark teal)
C_MISMATCH = "#BF360C"    # mismatch annotation (deep orange)
C_WRAP     = "#4527A0"    # 360° arrow  (deep purple)
C_BORDER   = "#1A237E"    # view border (indigo)

# ── Layout (data-unit coordinates) ────────────────────────────────────────
VIEW_GAP = 1.6
TOTAL    = V * W + (V - 1) * VIEW_GAP    # total x span = 27.2

# ═════════════════════════════════════════════════════════════════════════
# Row drawing
# ═════════════════════════════════════════════════════════════════════════

def _draw_row(
    ax:      plt.Axes,
    ids:     np.ndarray,   # [V, H, W]  values in [0,1]
    lbl_arr: np.ndarray,   # [V, H, W]  raw values to print as labels
    lbl_fmt: str,          # "int" or "float2"
    title:   str,
    is_pano: bool,
) -> None:
    """Render one comparison row (standard VLM or PanoRoPE)."""

    ax.set_xlim(-0.9, TOTAL + 0.9)
    ax.set_ylim(-1.4,  H + 2.5)   # same for both rows → equal cell size
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10,
                 color="#1A1A2E")


    # ── Draw each view ─────────────────────────────────────────────────────
    for v in range(V):
        x0 = v * (W + VIEW_GAP)


        # -- Cells ---------------------------------------------------------
        for r in range(H):
            for c in range(W):
                val  = float(ids[v, r, c])
                rgba = CMAP(NORM(val))
                lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc   = "white" if lum < 0.50 else "#111111"

                # Patch cell
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0 + c + 0.04, H - 1 - r + 0.04), 0.92, 0.92,
                    boxstyle="round,pad=0.04",
                    facecolor=rgba, edgecolor="none", zorder=2,
                ))


        # -- Overlap region highlights -------------------------------------
        _ovl_kw = dict(
            facecolor=C_OVL, alpha=0.14,
            edgecolor=C_OVL, linewidth=1.8,
            linestyle=(0, (6, 3)),
            zorder=3,
        )
        if v < V - 1:                               # right overlap
            ax.add_patch(mpatches.Rectangle(
                (x0 + W - K, 0), K, H, **_ovl_kw))
        if v > 0:                                   # left overlap
            ax.add_patch(mpatches.Rectangle(
                (x0,         0), K, H, **_ovl_kw))

        # -- View border ---------------------------------------------------
        ax.add_patch(mpatches.Rectangle(
            (x0, 0), W, H,
            facecolor="none", edgecolor=C_BORDER,
            linewidth=2.0, zorder=5,
        ))

        # -- View label ----------------------------------------------------
        ax.text(
            x0 + W / 2, -0.35,
            rf"View $v_{{{v}}}$",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=C_BORDER,
        )

    # ── Correspondence annotations between adjacent views ─────────────────
    c_ann = C_MATCH    if is_pano else C_MISMATCH
    y_ar  = H + 0.28
    y_lb  = H + 0.70

    for v in range(V - 1):
        cx_r = v       * (W + VIEW_GAP) + W - K / 2  # centre right-overlap
        cx_l = (v + 1) * (W + VIEW_GAP)     + K / 2  # centre left-overlap
        mid  = (cx_r + cx_l) / 2

        _br = dict(color=c_ann, lw=1.4, zorder=6)
        ax.plot([cx_r - 0.1, cx_r - 0.1], [H, y_ar], **_br)
        ax.plot([cx_l + 0.1, cx_l + 0.1], [H, y_ar], **_br)
        ax.annotate(
            "", xy=(cx_l + 0.1, y_ar), xytext=(cx_r - 0.1, y_ar),
            arrowprops=dict(
                arrowstyle="<->", color=c_ann, lw=1.8,
                connectionstyle="arc3,rad=0.0",
            ), zorder=6,
        )

        lbl = "same position" if is_pano else "position mismatch"
        ax.text(mid, y_lb, lbl,
                ha="center", va="bottom",
                fontsize=8.5, color=c_ann, fontweight="bold")

    # ── 360° wrap-around arrow (PanoRoPE only) ────────────────────────────
    if is_pano:
        x_tail = (V - 1) * (W + VIEW_GAP) + W - K / 2
        x_head = K / 2
        y_arc  = H + 1.35
        ax.annotate(
            "", xy=(x_head, y_arc), xytext=(x_tail, y_arc),
            arrowprops=dict(
                arrowstyle="<->", color=C_WRAP, lw=1.8,
                connectionstyle="arc3,rad=-0.22",
            ), zorder=6,
        )
        ax.text(
            TOTAL / 2, y_arc + 0.52,
            "360° wrap-around continuity",
            ha="center", va="bottom",
            fontsize=9, color=C_WRAP, fontweight="bold", style="italic",
        )

    # ── Overlap-region column markers ─────────────────────────────────────
    for v in range(V):
        x0 = v * (W + VIEW_GAP)
        for x_mark in [x0, x0 + W]:
            ax.plot([x_mark, x_mark], [H, H + 0.15],
                    color=C_OVL, lw=1.2, zorder=5)


# ═════════════════════════════════════════════════════════════════════════
# Assemble figure
# ═════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(
    2, 1,
    figsize=(13.8, 7.6),
    gridspec_kw={"hspace": 0.48},
)

_draw_row(
    axes[0], std_ids, std_col_int, "int",
    title="(a)  Standard VLM \u2014 Independent per-view local position IDs",
    is_pano=False,
)
_draw_row(
    axes[1], pano_ids, pano_col, "float2",
    title="(b)  PanoRoPE \u2014 Overlap-aware continuous 2D position encoding",
    is_pano=True,
)


# ── Save ──────────────────────────────────────────────────────────────────
_out = Path(__file__).parent
for _fmt in ("pdf", "png"):
    _p = _out / f"fig2_panorope.{_fmt}"
    fig.savefig(_p, bbox_inches="tight",
                dpi=300 if _fmt == "pdf" else 200)
    print(f"Saved: {_p}")
