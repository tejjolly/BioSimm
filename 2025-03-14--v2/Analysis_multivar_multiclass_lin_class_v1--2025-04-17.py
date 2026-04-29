#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi‑class SVM visualisation (4 classes) with per‑pair metric heat‑maps.

Author: tejjolly   |   Date: 2025‑04‑17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from itertools import combinations

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------------
# 1) LOAD & BASIC CLEAN‑UP
# -------------------------------------------------------------------------
df = pd.read_csv("../data/data.csv")

df['discord']  = pd.to_numeric(df['discord'], errors='coerce')
df['Location_Numeric'] = df['Location'].map({'LAD': 0, 'LCX': 1})

feature_candidates = ['P_Loss_Coeff', 'BMR/HMR', 'HMR']
df = df.dropna(subset=feature_candidates + ['discord'])

# -------------------------------------------------------------------------
# 2) TRAIN / TEST SPLIT  (80‑20 stratified)
# -------------------------------------------------------------------------
idx_train, idx_test = train_test_split(
    df.index,
    test_size=0.20,
    stratify=df['discord'].values,
    random_state=10
)
print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

# -------------------------------------------------------------------------
# 3) COLOUR & LEGEND PREP
# -------------------------------------------------------------------------
class_colors = ["#AAFFAA", "#AAAADD", "#FFDDAA", "#FFAAAA"]
cmap_4       = ListedColormap(class_colors)

legend_labels = [
    "CFR>2, FFR>0.8",  "CFR>2, FFR<0.8",
    "CFR<2, FFR>0.8",  "CFR<2, FFR<0.8"
]

# -------------------------------------------------------------------------
# 4) HELPER: RUN 2‑D SVM + DECISION REGIONS
# -------------------------------------------------------------------------
def run_two_feature_svm(df, idx_tr, idx_te, feat_x, feat_y, ax):
    """Train on (feat_x, feat_y) → discord, draw decision regions, return cm."""
    df_tr = df.loc[idx_tr].dropna(subset=[feat_x, feat_y, 'discord'])
    df_te = df.loc[idx_te].dropna(subset=[feat_x, feat_y, 'discord'])

    X_tr, y_tr = df_tr[[feat_x, feat_y]].values, df_tr['discord'].astype(int).values
    X_te, y_te = df_te[[feat_x, feat_y]].values, df_te['discord'].astype(int).values

    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    svm = SVC(kernel='linear', C=1, decision_function_shape='ovo')
    svm.fit(X_tr_s, y_tr)

    # decision region mesh (un‑scaled)
    x_min, x_max = X_tr_s[:,0].min()-0.5, X_tr_s[:,0].max()+0.5
    y_min, y_max = X_tr_s[:,1].min()-0.5, X_tr_s[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    xx_u = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_u = yy * scaler.scale_[1] + scaler.mean_[1]

    ax.contourf(xx_u, yy_u, Z,
                levels=[-0.5,0.5,1.5,2.5,3.5],
                cmap=cmap_4, alpha=0.4)

    # scatter points
    all_pts = pd.concat([df_tr, df_te])
    for cid in range(4):
        pts = all_pts[all_pts['discord'] == cid]
        ax.scatter(pts[feat_x], pts[feat_y],
                   color=class_colors[cid], edgecolors='k',
                   marker='o', alpha=0.9)

    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)

    # confusion matrix on **test** set
    y_pred = svm.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred, labels=[0,1,2,3])
    acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0
    ax.set_title(f"{feat_x} vs {feat_y}\nAcc = {acc:.2f}")
    return cm

# -------------------------------------------------------------------------
# 5) LOOP THROUGH FEATURE PAIRS
# -------------------------------------------------------------------------
pairs = list(combinations(feature_candidates, 2))
fig_dec, axes_dec = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 6)) \
                     if len(pairs)>1 else (plt.figure(figsize=(6,6)), [plt.gca()])

for ax_dec, (fx, fy) in zip(axes_dec, pairs):
    cm = run_two_feature_svm(df, idx_train, idx_test, fx, fy, ax_dec)

    # ---- derive per‑class metrics ----
    TP = np.diag(cm).astype(float)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec = TP / (TP + FP + 1e-9)
    sens = TP / (TP + FN + 1e-9)          # recall
    spec = TN / (TN + FP + 1e-9)

    # ---- build DataFrame in the *exact* style you want ----
    metrics_df = pd.DataFrame({
        'Precision':   prec,
        'Sensitivity': sens,
        'Specificity': spec
    }, index=[f"Class {i}" for i in range(4)]).T

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        metrics_df,
        annot=True, fmt=".2f",
        cmap="Reds_r", cbar=True,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f"Per‑Class Metrics | {fx} vs {fy}")
    plt.ylabel("Metric")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()             # <‑‑ displays the heat‑map (not saved!)

# -------------------------------------------------------------------------
# 6) ADD LEGEND & SHOW DECISION REGION FIGURE
# -------------------------------------------------------------------------
legend_handles = [mlines.Line2D([], [], color=class_colors[c], marker='o',
                                markersize=9, markeredgecolor='k', linewidth=0,
                                label=legend_labels[c])
                  for c in range(4)]

if len(pairs) > 1:
    fig_dec.legend(legend_handles, legend_labels, loc='lower center',
                   ncol=2, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
else:
    axes_dec[0].legend(legend_handles, legend_labels, loc='best', fontsize=11)

plt.show()   # shows the decision‑boundary subplot(s)
