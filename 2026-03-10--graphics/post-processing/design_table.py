#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Polygon

# =========================================================
# USER INPUTS
# =========================================================

# Put these in the order you want them shown:
# bottom -> top for stenosis, left -> right for microvascular disease
stenosis_labels = ['0%', '45%', '60%']
micro_labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4']

# Optional small case IDs for the design matrix (set to None for blank cells)
case_ids = np.array([
    ['1',  '2',  '3',  '4'],
    ['5',  '6',  '7',  '8'],
    ['9', '10', '11', '12']
], dtype=object)

# Example metric values for the actual heatmap
# Shape must be (3 rows for stenosis, 4 cols for microvascular levels)
# Replace with your real values (e.g., FFR, CFR, distal flow, etc.)
metric_values = np.array([
    [0.98, 0.96, 0.93, 0.90],  # 0% stenosis
    [0.90, 0.85, 0.79, 0.73],  # 45% stenosis
    [0.80, 0.72, 0.64, 0.57],  # 60% stenosis
])

metric_name = 'FFR'

# Colormaps
cmap_bupu = mpl.cm.BuPu
cmap_reds = mpl.cm.Reds


# =========================================================
# HELPERS
# =========================================================
def draw_design_matrix(ax, stenosis_labels, micro_labels, case_ids=None,
                       bupu_range=(0.35, 0.80), reds_range=(0.35, 0.80)):
    """
    Draws a poster-style design matrix:
    - top header row = BuPu gradient left -> right
    - left header column = Reds gradient bottom -> top
    - interior cells split diagonally:
        upper triangle = Reds (stenosis burden)
        lower triangle = BuPu (microvascular burden)
    """
    n_rows = len(stenosis_labels)
    n_cols = len(micro_labels)

    bupu_vals = np.linspace(bupu_range[0], bupu_range[1], n_cols)
    reds_vals = np.linspace(reds_range[0], reds_range[1], n_rows)

    # Corner cell
    ax.add_patch(Rectangle((0, n_rows), 1, 1, facecolor='white',
                           edgecolor='black', lw=1.4))
    ax.text(0.5, n_rows + 0.63, 'Study', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(0.5, n_rows + 0.34, 'design', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Top header row: microvascular axis
    for j, label in enumerate(micro_labels):
        color = cmap_bupu(bupu_vals[j])
        ax.add_patch(Rectangle((j + 1, n_rows), 1, 1,
                               facecolor=color, edgecolor='black', lw=1.4))
        ax.text(j + 1.5, n_rows + 0.5, label,
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Left header column: stenosis axis
    for i, label in enumerate(stenosis_labels):
        color = cmap_reds(reds_vals[i])
        ax.add_patch(Rectangle((0, i), 1, 1,
                               facecolor=color, edgecolor='black', lw=1.4))
        ax.text(0.5, i + 0.5, label,
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Interior split cells
    for i in range(n_rows):
        for j in range(n_cols):
            x, y = j + 1, i

            red_color = cmap_reds(reds_vals[i])
            bupu_color = cmap_bupu(bupu_vals[j])

            # Diagonal split from bottom-left to top-right
            # Upper triangle = Reds
            tri_red = Polygon([(x, y), (x, y + 1), (x + 1, y + 1)],
                              closed=True, facecolor=red_color, edgecolor='none')

            # Lower triangle = BuPu
            tri_bupu = Polygon([(x, y), (x + 1, y), (x + 1, y + 1)],
                               closed=True, facecolor=bupu_color, edgecolor='none')

            ax.add_patch(tri_red)
            ax.add_patch(tri_bupu)

            # Border
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False,
                                   edgecolor='black', lw=1.2))

            # Optional cell label
            if case_ids is not None:
                ax.text(x + 0.5, y + 0.5, str(case_ids[i, j]),
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color='black')

    # Axis annotations
    ax.text((n_cols + 1) / 2 + 0.5, -0.35,
            'Microvascular resistance / disease  →',
            ha='center', va='center', fontsize=11)

    ax.text(-0.42, n_rows / 2,
            'Stenosis burden  →',
            ha='center', va='center', rotation=90, fontsize=11)

    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('A. Experimental / simulation design space', fontsize=13, pad=12)


def draw_metric_heatmap(ax, values, stenosis_labels, micro_labels, metric_name='Metric',
                        cmap='BuPu', fmt='{:.2f}'):
    """
    Standard heatmap for one output metric.
    """
    im = ax.imshow(values, origin='lower', cmap=cmap, aspect='equal')

    ax.set_xticks(np.arange(len(micro_labels)))
    ax.set_xticklabels(micro_labels, rotation=0)
    ax.set_yticks(np.arange(len(stenosis_labels)))
    ax.set_yticklabels(stenosis_labels)

    ax.set_xlabel('Microvascular resistance / disease')
    ax.set_ylabel('Stenosis burden')
    ax.set_title(f'B. {metric_name} heatmap', fontsize=13, pad=12)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Annotate cell values
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    threshold = vmin + 0.55 * (vmax - vmin)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt_color = 'white' if val > threshold else 'black'
            ax.text(j, i, fmt.format(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)


# =========================================================
# PLOT
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)

draw_design_matrix(
    axes[0],
    stenosis_labels=stenosis_labels,
    micro_labels=micro_labels,
    case_ids=case_ids
)

draw_metric_heatmap(
    axes[1],
    values=metric_values,
    stenosis_labels=stenosis_labels,
    micro_labels=micro_labels,
    metric_name=metric_name,
    cmap='BuPu',
    fmt='{:.2f}'
)

plt.show()
