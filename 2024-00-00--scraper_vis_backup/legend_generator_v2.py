#!/usr/bin/env python3
"""
Legend-only figures:
1) Geometry encoding legend (stenosis % via markers, lesion length via linetype)
   - Optional vessel entries (LAD/LCX) gated by a flag
   - Markers are hollow (no fill)
2) Distribution histogram legend for RA-m levels (BuPu palette)

Outputs:
- marker_linetype(_vessel)_legend.(png|svg)
- images/R_levels_legend.(png|svg)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl

# =========================================================
# FLAGS / USER SETTINGS
# =========================================================
INCLUDE_VESSEL_LEGEND = False   # <-- toggle LAD/LCX legend entries ON/OFF

# Output paths
OUT_GEOM_PNG = "marker_linetype_legend.png"
OUT_GEOM_SVG = "marker_linetype_legend.svg"

RLEG_DIR = "images"
OUT_R_PNG = os.path.join(RLEG_DIR, "R_levels_legend.png")
OUT_R_SVG = os.path.join(RLEG_DIR, "R_levels_legend.svg")

# =========================================================
# 1) GEOMETRY LEGEND (stenosis marker + length linetype)
# =========================================================
# Hollow marker styling
MS = 10
MEW = 1.5  # marker edge width

# ── Marker legend entries for stenosis ──
circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                       markersize=MS, markerfacecolor='none', markeredgewidth=MEW,
                       label='0% stenosis')

square = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                       markersize=MS, markerfacecolor='none', markeredgewidth=MEW,
                       label='45% stenosis')

triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                         markersize=MS, markerfacecolor='none', markeredgewidth=MEW,
                         label='60% stenosis')

# ── Line legend entries for lesion length ──
dotted = mlines.Line2D([], [], color='black', linestyle=':',  linewidth=2, label='1.2 cm')
dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='2.5 cm')

handles_geom = [circle, square, triangle, dotted, dashed]

# ── Optional vessel legend entries (LAD/LCX) ──
if INCLUDE_VESSEL_LEGEND:
    lad = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='LAD')
    lcx = mlines.Line2D([], [], color='gray',  linestyle='-', linewidth=2, label='LCX', alpha=0.6)
    handles_geom += [lad, lcx]

# ── Create figure with only the legend ──
# (Slightly wider if vessel legend is included)
fig_w = 3.75 if INCLUDE_VESSEL_LEGEND else 3.25
fig_h = 0.85 if INCLUDE_VESSEL_LEGEND else 0.75

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=400)
ax.axis('off')

legend = ax.legend(
    handles=handles_geom,
    loc='center',
    frameon=False,
    ncol=5,
    handlelength=2.0,
    handletextpad=0.1,
    columnspacing=0.6,
    fontsize=9
)

fig.savefig(OUT_GEOM_PNG, dpi=400, transparent=True, bbox_inches='tight')
fig.savefig(OUT_GEOM_SVG, transparent=True, bbox_inches='tight')
plt.show()

# =========================================================
# 2) DISTRIBUTION HISTOGRAM LEGEND (RA-m levels)
# =========================================================
# ================= USER-TUNABLE SETTINGS =================
FIG_WIDTH     = 2.0    # inches
FIG_HEIGHT    = 2.2    # inches
FONT_SIZE     = 12
HANDLE_LENGTH = 2.5
HANDLE_TEXTPAD = 0.2
LABEL_SPACING  = 0.8   # vertical spacing between entries

# =========================================================
LEVELS = [
    (0.24, r"Base $R_{A\text{-}m}$"),
    (0.43, r"$R_{A\text{-}m}x2.7$"),
    (0.62, r"$R_{A\text{-}m}x4.0$"),
    (0.81, r"$R_{A\text{-}m}x6.3$")
]
hue_order = [lab for _, lab in LEVELS]

# Color palette, same as main plots
vals = np.r_[0.25, np.linspace(0.6, 1.0, len(hue_order) - 1)]
color_map = mpl.cm.BuPu(vals)
palette = {lab: col for lab, col in zip(hue_order, color_map)}

# We want Base R to appear at the bottom, so reverse for legend order
legend_order = hue_order[::-1]

# Build handles
handles_r = [
    mlines.Line2D([], [], color=palette[label], linestyle='--', linewidth=18,
                  label=label, alpha=0.75)
    for label in legend_order
]

# Make the figure & no axes
os.makedirs(RLEG_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=400)
ax.axis('off')

legend = ax.legend(
    handles=handles_r,
    loc='center',
    frameon=False,
    ncol=1,
    fontsize=FONT_SIZE,
    handlelength=HANDLE_LENGTH,
    handletextpad=HANDLE_TEXTPAD,
    borderpad=0.0,
    labelspacing=LABEL_SPACING
)

fig.savefig(OUT_R_PNG, dpi=400, transparent=True, bbox_inches='tight')
fig.savefig(OUT_R_SVG, transparent=True, bbox_inches='tight')
plt.show()