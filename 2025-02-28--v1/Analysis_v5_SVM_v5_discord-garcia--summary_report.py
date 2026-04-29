#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4‑class SVM on the original ‘discord’ labels (0, 1, 2, 3)
• Three decision‑region scatter plots (one per feature pair)
• One heat‑map per pair (Precision, Sensitivity, Specificity × 4 classes)
Colour palette: four evenly spaced colours from “RdYlGn_r”.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score
)
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────────────────────────────────────
# 0) helper
# ──────────────────────────────────────────────────────────────────────
def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.asarray(spec)

# ──────────────────────────────────────────────────────────────────────
# 1) LOAD + PREP
# ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/data.csv")
df = df[df["Condition"] == "Hyperemic"].copy()

df["discord"] = pd.to_numeric(df["discord"], errors="coerce")
df = df.dropna(subset=["discord"])

feature_candidates = ["P_Loss_Coeff", "BMR/HMR", "HMR"]
df = df.dropna(subset=feature_candidates)

# train‑only split (set >0 if you want a hold‑out slice)
test_size = 0.0
all_idx   = df.index.to_numpy()
if test_size and test_size > 0:
    idx_train, idx_test = train_test_split(
        all_idx, test_size=test_size,
        stratify=df["discord"].values, random_state=10
    )
else:
    idx_train, idx_test = all_idx, np.array([], dtype=int)

print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

# ──────────────────────────────────────────────────────────────────────
# 2) COLOURS
# ──────────────────────────────────────────────────────────────────────
cmap_4      = get_cmap("RdYlGn_r", 4)               # discrete 4‑colour map
class_colors = [cmap_4(i) for i in range(cmap_4.N)] # RGBA tuples → list
legend_labels = ['CFR > 2, FFR > 0.8', 'CFR > 2.0, FFR < 0.8',
                 'CFR < 2.0, FFR < 0.8', 'CFR < 2.0, FFR < 0.8']

all_pairs = list(combinations(feature_candidates, 2))   # 3 feature pairs

# ──────────────────────────────────────────────────────────────────────
# 3) CORE FUNCTION
# ──────────────────────────────────────────────────────────────────────
def plot_pair(df, train_idx, test_idx, xcol, ycol, ax):
    df_train = df.loc[train_idx, [xcol, ycol, "discord"]]
    df_test  = df.loc[test_idx,  [xcol, ycol, "discord"]]
    df_plot  = pd.concat([df_train, df_test])

    X_tr = df_train[[xcol, ycol]].values
    y_tr = df_train["discord"].astype(int).values

    scaler = StandardScaler().fit(X_tr)
    svm    = SVC(kernel="linear", C=1, decision_function_shape="ovr") \
                 .fit(scaler.transform(X_tr), y_tr)

    X_all_s = scaler.transform(df_plot[[xcol, ycol]].values)
    y_all   = df_plot["discord"].astype(int).values
    y_pred  = svm.predict(X_all_s)

    # background
    x_min, x_max = X_all_s[:, 0].min() - .5, X_all_s[:, 0].max() + .5
    y_min, y_max = X_all_s[:, 1].min() - .5, X_all_s[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    xx_u = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_u = yy * scaler.scale_[1] + scaler.mean_[1]
    ax.contourf(xx_u, yy_u, Z,
                levels=[-0.5, .5, 1.5, 2.5, 3.5],
                cmap=cmap_4, alpha=.35)

    # scatter
    for cid in range(4):
        pts = df_plot[df_plot["discord"] == cid]
        ax.scatter(pts[xcol], pts[ycol],
                   color=class_colors[cid], edgecolors="k", alpha=.9)

    # metrics
    cm   = confusion_matrix(y_all, y_pred, labels=[0,1,2,3])
    prec = precision_score(y_all, y_pred, average=None, zero_division=0)
    sens = recall_score(   y_all, y_pred, average=None, zero_division=0)
    spec = specificity_per_class(cm)

    metrics_df = pd.DataFrame({
        "Precision":   prec,
        "Sensitivity": sens,
        "Specificity": spec,
    }, index=legend_labels).T

    return metrics_df

# ──────────────────────────────────────────────────────────────────────
# 4) SCATTER / DECISION PLOTS
# ──────────────────────────────────────────────────────────────────────
fig_scatter, axes = plt.subplots(1, 3, figsize=(16, 6))
heatmaps = []

for ax, (xc, yc) in zip(axes.flatten(), all_pairs):
    mdf = plot_pair(df, idx_train, idx_test, xc, yc, ax)
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    heatmaps.append((mdf, f"{xc} vs {yc}"))

# legend
handles = [mlines.Line2D([], [], color=class_colors[i], marker='o',
                         markeredgecolor='k', linewidth=0, markersize=10,
                         label=legend_labels[i])
           for i in range(4)]
fig_scatter.legend(handles=handles, loc='lower center',
                   ncol=4, frameon=True, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ──────────────────────────────────────────────────────────────────────
# 5) HEAT‑MAPS
# ──────────────────────────────────────────────────────────────────────
for metrics_df, title in heatmaps:
    plt.figure(figsize=(6, 3.3))
    sns.heatmap(metrics_df, annot=True, fmt=".2f",
                cmap="Reds_r", vmin=.5, vmax=1,
                linewidths=.5, linecolor="grey",
                annot_kws={"size": 9})
    plt.title(f"Per‑Class Metrics – {title}", fontsize=10)
    plt.ylabel("Metric"); plt.xlabel("Class", fontsize=9)
    plt.xticks(rotation=30, fontsize = 6, ha = 'right')
    plt.yticks(fontsize = 9)
    plt.tight_layout()
    plt.show()
