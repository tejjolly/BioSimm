#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note: This Neural net is a 2-d bottleneck classifier
"""
"""
compare_models_with_heatmaps_v2.py
----------------------------------
• Two models → tables + heat‑maps (means share common scale)
• Heat‑maps for Model A & B now *annotate mean ± std* in every cell
• Additional heat‑maps for Δ and %‑improvement (diverging, centre 0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────────────────────────────────────────────────────
# 1) INPUT BLOCK  ←‑‑‑‑‑‑‑‑‑ edit only the numbers!
# ───────────────────────────────────────────────────────────
metrics = {
    "Neural Net": {                           # Model A
        "Precision":   {0:(0.923,0.047), 1:(0.892,0.101), 2:(0.898,0.085), 3:(0.920,0.069)},
        "Sensitivity": {0:(0.930,0.051), 1:(0.873,0.079), 2:(0.860,0.127), 3:(0.912,0.058)},
        "Specificity": {0:(0.946,0.034), 1:(0.956,0.045), 2:(0.986,0.011), 3:(0.974,0.023)}
    },
    "Linear Classifier": {                    # Model B
        "Precision":   {0:(0.930,0.025), 1:(0.828,0.021), 2:(0.731,0.172), 3:(0.868,0.082)},
        "Sensitivity": {0:(0.830,0.043), 1:(0.874,0.081), 2:(0.852,0.222), 3:(0.842,0.055)},
        "Specificity": {0:(0.958,0.014), 1:(0.938,0.011), 2:(0.944,0.046), 3:(0.958,0.031)}
    }
}
# ───────────────────────────────────────────────────────────
# 2) BUILD DATAFRAMES
# ───────────────────────────────────────────────────────────
pretty_tabs, mean_tabs, std_tabs = {}, {}, {}
classes = [f"Class {i}" for i in sorted(metrics['Neural Net']['Precision'])]

for model_name, mdict in metrics.items():
    pretty_rows, mean_rows, std_rows = {}, {}, {}
    for metric_name, class_dict in mdict.items():
        pretty_rows[metric_name] = [f"{m:.3f} ± {s:.3f}" for _, (m, s) in sorted(class_dict.items())]
        mean_rows  [metric_name] = [m for _, (m, _) in sorted(class_dict.items())]
        std_rows   [metric_name] = [s for _, (_, s) in sorted(class_dict.items())]
    pretty_tabs[model_name] = pd.DataFrame(pretty_rows, index=classes).T
    mean_tabs  [model_name] = pd.DataFrame(mean_rows,   index=classes).T
    std_tabs   [model_name] = pd.DataFrame(std_rows,    index=classes).T

# ───────────────────────────────────────────────────────────
# 3) Δ and %‑improvement numeric tables
# ───────────────────────────────────────────────────────────
modelA, modelB = list(metrics)                   # keep order
delta_vals, pct_vals = {}, {}

for metric in metrics[modelA]:
    d, p = [], []
    for cls in sorted(metrics[modelA][metric]):
        mA = metrics[modelA][metric][cls][0]
        mB = metrics[modelB][metric][cls][0]
        d.append(mA - mB)
        p.append(100*(mA - mB)/mB if mB else np.nan)
    delta_vals[metric] = d
    pct_vals  [metric] = p

delta_df = pd.DataFrame(delta_vals, index=classes).T
pct_df   = pd.DataFrame(pct_vals,   index=classes).T

# ───────────────────────────────────────────────────────────
# 4)  PRINT TEXT TABLES
# ───────────────────────────────────────────────────────────
pd.set_option("display.width", 140)
print("\n=== Model A:", modelA, "===\n", pretty_tabs[modelA], "\n")
print("=== Model B:", modelB, "===\n", pretty_tabs[modelB], "\n")

print("=== Mean difference (Model A – Model B) ===\n",
      delta_df.map(lambda x: f"{x:+.3f}"), "\n")
print("=== % improvement relative to Model B ===\n",
      pct_df.map(lambda x: f"{x:+.1f} %"), "\n")

# ───────────────────────────────────────────────────────────
# 5)  HEAT‑MAPS
# ───────────────────────────────────────────────────────────
sns.set(style="white")
vmin_global = min(mean_tabs[modelA].min().min(),
                  mean_tabs[modelB].min().min())
vmax_global = 1.0          # metrics max out at 1

def make_annot(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of 'mean\n±std' strings (same index/cols)."""
    out = mean_df.copy().astype(str)
    for r in mean_df.index:
        for c in mean_df.columns:
            m = mean_df.loc[r, c]
            s = std_df .loc[r, c]
            out.loc[r, c] = f"{m:.2f}\n±{s:.2f}"
    return out

fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                         gridspec_kw={'height_ratios': [1, 1.15]})
(axA, axB), (axD, axP) = axes

# — Model A —
annot_A = make_annot(mean_tabs[modelA], std_tabs[modelA])
sns.heatmap(mean_tabs[modelA], annot=annot_A,
            fmt="", cmap="Reds_r",
            vmin=vmin_global, vmax=vmax_global,
            linewidths=0.5, linecolor="grey", cbar=False, ax=axA)
axA.set_title(f"{modelA}  (mean ± σ)")
axA.set_xlabel("Class"); axA.set_ylabel("Metric")

# — Model B —
annot_B = make_annot(mean_tabs[modelB], std_tabs[modelB])
sns.heatmap(mean_tabs[modelB], annot=annot_B,
            fmt="", cmap="Reds_r",
            vmin=vmin_global, vmax=vmax_global,
            linewidths=0.5, linecolor="grey", cbar=False, ax=axB)
axB.set_title(f"{modelB}  (mean ± σ)")
axB.set_xlabel("Class"); axB.set_ylabel("")

# — Absolute Δ (diverging) —
sns.heatmap(delta_df, annot=delta_df.map(lambda x: f"{x:+.2f}"),
            fmt="", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="grey", cbar=False, ax=axD)
axD.set_title("Model A – Model B  (Δ)")
axD.set_xlabel("Class"); axD.set_ylabel("Metric")

# — %‑improvement (diverging) —
sns.heatmap(pct_df, annot=pct_df.map(lambda x: f"{x:+.1f}%"),
            fmt="", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="grey", cbar=False, ax=axP)
axP.set_title("% Improvement vs Model B")
axP.set_xlabel("Class"); axP.set_ylabel("Metric")

plt.tight_layout()
plt.show()
