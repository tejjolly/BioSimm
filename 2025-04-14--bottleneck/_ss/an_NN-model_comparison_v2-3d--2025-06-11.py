#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models_heatmaps_no_std.py
---------------------------------
• Uses means from two existing heat‑maps (Model 1 vs Model 2).
• No ± std‑dev annotations.
• Produces four heat‑maps:
      – Model 1          – Model 2
      – Δ  (Model1‑Model2)
      – % improvement vs Model 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────────────────────────────────────────────────────
# 1)  INPUT:  means only  (edit here if the numbers change)
# ───────────────────────────────────────────────────────────
metrics = {
    "Model 1": {                # values from FIRST heat‑map
        "Precision":   {0: 0.93, 1: 0.93, 2: 0.88, 3: 0.90},
        "Sensitivity": {0: 0.94, 1: 0.87, 2: 0.90, 3: 0.91},
        "Specificity": {0: 0.95, 1: 0.97, 2: 0.98, 3: 0.97},
    },
    "Model 2": {                # values from SECOND heat‑map
        "Precision":   {0: 0.92, 1: 0.94, 2: 0.89, 3: 0.93},
        "Sensitivity": {0: 0.94, 1: 0.86, 2: 0.87, 3: 0.96},
        "Specificity": {0: 0.95, 1: 0.98, 2: 0.98, 3: 0.97},
    },
}

# ───────────────────────────────────────────────────────────
# 2)  BUILD MEAN DATAFRAMES
# ───────────────────────────────────────────────────────────
mean_tabs = {}
classes   = [f"Class {i}" for i in sorted(next(iter(metrics.values()))["Precision"])]

for model_name, mdict in metrics.items():
    mean_rows = {
        metric_name: [class_dict[c] for c in sorted(class_dict)]
        for metric_name, class_dict in mdict.items()
    }
    mean_tabs[model_name] = pd.DataFrame(mean_rows, index=classes).T

# ───────────────────────────────────────────────────────────
# 3)  Δ  and %‑improvement  (Model 1 – Model 2)
# ───────────────────────────────────────────────────────────
modelA, modelB = list(metrics)  # preserves insertion order
delta_df = mean_tabs[modelB] - mean_tabs[modelA]
pct_df   = 100 * delta_df / mean_tabs[modelA]

# ───────────────────────────────────────────────────────────
# 4)  HEAT‑MAPS
# ───────────────────────────────────────────────────────────
sns.set(style="white")
vmin_global = min(tab.min().min() for tab in mean_tabs.values())
vmax_global = 1.0

# ── helper ────────────────────────────────────────────────────────────
def plot_single(ax, df, title,
                *, cmap="Reds_r", mode="raw",
                fmt=lambda x: f"{x:.2f}"):
    """
    mode="raw"        → sequential map, vmin=0.5, vmax=1,  cbar=True
    mode="delta"      → diverging map  centred at 0,         cbar=False
    """
    if mode == "raw":
        vmin, vmax, center, cbar_flag = 0.50, 1.00, None, True
    elif mode == "delta":
        lim = np.abs(df.values).max()
        vmin, vmax, center, cbar_flag = -lim, +lim, 0, False
    else:
        raise ValueError("mode must be 'raw' or 'delta'")

    sns.heatmap(
        df,
        annot=df.applymap(fmt), fmt="",
        cmap=cmap, vmin=vmin, vmax=vmax, center=center,
        linewidths=.5, linecolor="grey", cbar=cbar_flag, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Metric")

# ── draw ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2, figsize=(12, 8),
    gridspec_kw={"height_ratios": [1, 1.15]}
)
(axA, axB), (axD, axP) = axes        # unpack four Axes objects

plot_single(axA, mean_tabs[modelA], modelA, mode="raw")
plot_single(axB, mean_tabs[modelB], modelB, mode="raw")

plot_single(axD, delta_df, f"3-D Model – 2-D Model  (Δ)",
            cmap="coolwarm", mode="delta",
            fmt=lambda x: f"{x:+.2f}")
plot_single(axP, pct_df,  "% Improvement vs Model 1",
            cmap="coolwarm", mode="delta",
            fmt=lambda x: f"{x:+.1f}%")

plt.tight_layout()
plt.show()

