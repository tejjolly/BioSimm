#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary SVM: FFR/CFR Concord vs Discord
* three decision‑region scatter plots
* three matching heat‑maps (one per feature pair)
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
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────────────────────────────────────
# 0) helpers
# ──────────────────────────────────────────────────────────────────────
def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """TN / (TN+FP) for every class (binary → length‑2 array)."""
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

# collapse 4‑way discord → 2‑way
df["discord"] = (pd.to_numeric(df["discord"], errors="coerce")
                   .apply(lambda d: 0 if d in (0, 2) else 1))

feature_candidates = ["P_Loss_Coeff", "BMR/HMR", "HMR"]
df = df.dropna(subset=feature_candidates + ["discord"])

# split (test_size=0 means “all train”)
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
# 2) VIS SETTINGS
# ──────────────────────────────────────────────────────────────────────
# class_colors = ["#5E9096", "#5E2F5C"]                 # Concord, Discord
# cmap_2 = ListedColormap(class_colors)
colors = plt.get_cmap("coolwarm", 2)  # Get 2 discrete colors from viridis
class_colors = [colors(0), colors(1)]  # Concord, Discord
cmap_2 = ListedColormap(class_colors)
all_pairs = list(combinations(feature_candidates, 2)) # 3 pairs

# ──────────────────────────────────────────────────────────────────────
# 3) SVM + PLOTS + PER‑PAIR HEAT‑MAPS
# ──────────────────────────────────────────────────────────────────────
def plot_pair(df, train_idx, test_idx, xcol, ycol, ax):
    """Fit SVM on (xcol,ycol), draw decision region & points, return metrics."""
    # ----- split -----
    df_train = df.loc[train_idx, [xcol, ycol, "discord"]]
    df_test  = df.loc[test_idx,  [xcol, ycol, "discord"]]
    df_plot  = pd.concat([df_train, df_test])

    X_tr = df_train[[xcol, ycol]].values
    y_tr = df_train["discord"].astype(int).values

    # empty test set allowed
    X_te = df_test [[xcol, ycol]].values if len(df_test) else np.empty((0, 2))
    y_te = df_test ["discord"].astype(int).values if len(df_test) else np.empty(0)

    # ----- scale + model -----
    scaler = StandardScaler().fit(X_tr)
    svm    = SVC(kernel="linear", C=.1).fit(scaler.transform(X_tr), y_tr)

    X_all_s = scaler.transform(df_plot[[xcol, ycol]].values)
    y_all   = df_plot["discord"].astype(int).values
    y_pred  = svm.predict(X_all_s)

    # ----- decision background -----
    x_min, x_max = X_all_s[:, 0].min() - .5, X_all_s[:, 0].max() + .5
    y_min, y_max = X_all_s[:, 1].min() - .5, X_all_s[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    xx_u = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_u = yy * scaler.scale_[1] + scaler.mean_[1]
    ax.contourf(xx_u, yy_u, Z, levels=[-0.5, .5, 1.5],
                cmap=cmap_2, alpha=.4)

    # ----- scatter points -----
    for cid in (0, 1):
        pts = df_plot[df_plot["discord"] == cid]
        ax.scatter(pts[xcol], pts[ycol],
                   color=class_colors[cid], edgecolors="k", alpha=.9)

    # ----- metrics for this pair -----
    cm   = confusion_matrix(y_all, y_pred, labels=[0, 1])
    prec = precision_score(y_all, y_pred, average=None, zero_division=0)
    sens = recall_score(   y_all, y_pred, average=None, zero_division=0)
    spec = specificity_per_class(cm)

    metrics_df = pd.DataFrame({
        "Precision":   prec,
        "Sensitivity": sens,
        "Specificity": spec,
    }, index=["Concord", "Discord"]).T           # two columns (0,1) → names

    return metrics_df

# --- scatter / decision regions ---
fig_scatter, axes = plt.subplots(1, 3, figsize=(16, 6))

heatmaps = []  # (metrics_df, title) tuples
for ax, (xc, yc) in zip(axes.flatten(), all_pairs):
    mdf = plot_pair(df, idx_train, idx_test, xc, yc, ax)
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    heatmaps.append((mdf, f"{xc} vs {yc}"))

    if xc != 'BMR/HMR':
        if yc != 'BMR/HMR':
            fig_individual, ax_individual = plt.subplots(figsize=(6, 4))
            plot_pair(df, idx_train, idx_test, xc, yc, ax_individual)
            # ax_individual.set_title(f"{xc} vs {yc}", fontsize=11)
            ax_individual.set_xlabel(r"$\zeta_{\mathrm{L}}$", fontsize=18)
            ax_individual.set_ylabel(f'{yc} [mmHg/cm/s]', fontsize=18)

            # Set tick label font sizes
            ax_individual.tick_params(axis='both', which='major', labelsize=18)

            fig_individual.tight_layout()
            fig_individual.savefig(f"scatter_{xc}_vs_{yc}.svg", transparent=True, format="svg")
            plt.show()
            plt.close(fig_individual)

# --- legend ---
handles = [mlines.Line2D([], [], color=class_colors[i], marker='o',
                         markeredgecolor='k', linewidth=0, markersize=10,
                         label=lbl)
           for i, lbl in enumerate(["Concordant", "Discordant"])]
fig_scatter.legend(handles=handles, loc='lower center',
                   ncol=2, frameon=True, fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

# --- produce one heat‑map per pair ---
for metrics_df, title in heatmaps:
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(metrics_df, annot=True, fmt=".2f",
                cmap="Reds_r", vmin=.5, vmax=1,
                linewidths=.5, linecolor="grey",
                annot_kws={"size": 10})
    plt.title(f"Per‑Class Metrics – {title}", fontsize = 10)
    plt.ylabel("Metric"); plt.xlabel("Class", fontsize = 10)
    plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.show()