#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:51:38 2025

@author: tejjolly

Description:
    - Train a multi-class SVM on 'discord' (4-class target).
    - Plot subplots for each pair of features showing decision boundaries and data.
    - Now includes confusion matrix and F1 score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score  # [ADDED]
from itertools import combinations
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, to_hex
# from matplotlib.cm import get_cmap
import seaborn as sns

# ── metrics helpers ─────────────────────────────────────────────────
def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """TN / (TN+FP) for every class."""
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.asarray(spec)

def make_metrics_df(cm: np.ndarray) -> pd.DataFrame:
    """Return a tidy dataframe exactly like the one in NBSP v7.2."""
    prec = np.diag(cm) / (cm.sum(axis=0) + 1e-9)           # precision
    sens = np.diag(cm) / (cm.sum(axis=1) + 1e-9)           # recall / sensitivity
    spec = specificity_per_class(cm)
    return pd.DataFrame(
        {"Precision": prec, "Sensitivity": sens, "Specificity": spec},
        index=[f"Class {i}" for i in range(cm.shape[0])]
    ).T

########################################
# 1) Load summary.csv and filter
########################################
df = pd.read_csv("../data/data.csv")

# # Example: focus only on hyperemic
# df = df[df['Condition'] == 'Hyperemic'].copy()

# Convert 'discord' to integer if needed
df['discord'] = pd.to_numeric(df['discord'], errors='coerce')

# Convert Location to numeric codes if needed
df['Location_Numeric'] = df['Location'].map({'LAD': 0, 'LCX': 1})

# The features to use:
feature_candidates = [
    'P_Loss_Coeff',
    'BMR/HMR',
    'HMR',
]

# Drop rows missing needed features or 'discord'
df = df.dropna(subset=feature_candidates + ['discord'])

########################################
# 2) Global 80/20 train/test split
########################################
all_indices = df.index.to_numpy()
y_dummy = df['discord'].values  # used for stratify
test_size = 0.33

idx_train, idx_test = train_test_split(
    all_indices,
    test_size=test_size,
    stratify=y_dummy,
    random_state=42
)

print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

########################################
# 3) Color Setup
########################################

NUM_CLASSES = 4
base_cmap   = plt.get_cmap('RdYlGn_r')            # continuous diverging map
class_colors = [
    to_hex(base_cmap(i / (NUM_CLASSES - 1)))   # evenly spaced 0‑1
    for i in range(NUM_CLASSES)
]

cmap_3 = ListedColormap(class_colors)       # for contourf / pcolormesh

# If you want to label them in a legend, define them here:
legend_labels = [
    "CFR>2, FFR>0.8",
    "CFR>2, FFR<0.8",
    "CFR<2, FFR>0.8",
    "CFR<2, FFR<0.8"
]  # be sure these match your actual 'discord' class definitions

# We'll do all pairs from the features
all_pairs = list(combinations(feature_candidates, 2))

########################################
# 4) Multi-class SVM function
########################################
def train_multiclass_svm_and_plot(
    df,
    train_idx,
    test_idx,
    xcol,
    ycol,
    ax,
    target_col='discord'
):
    """
    Trains a multi-class SVM (OvO by default) on (xcol, ycol) -> `discord`.
    Plots the 2D decision boundary with a 4-color background.
    Data points are circles with the same color as their class.

    Returns:
      - test_acc: float
      - y_test_true: array of shape (n_test,)
      - y_test_pred: array of shape (n_test,) (for confusion matrix, F1, etc.)
    """
    # 1) Build subsets
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])

    X_train = df_train[[xcol, ycol]].values
    X_test  = df_test[[xcol, ycol]].values
    y_train = df_train[target_col].astype(int).values
    y_test  = df_test[target_col].astype(int).values

    # 2) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3) Train SVC multi-class
    svm_clf = SVC(kernel='linear', C=1, decision_function_shape='ovo')
    svm_clf.fit(X_train_scaled, y_train)

    # 4) Evaluate on test set
    test_acc = svm_clf.score(X_test_scaled, y_test)

    # We'll also keep the predictions for confusion matrix & F1
    y_test_pred = svm_clf.predict(X_test_scaled)

    # 5) Generate mesh for background
    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 6) Inverse transform the mesh grid
    xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

    # 7) Plot background (4 classes => levels for boundaries: -0.5, 0.5, 1.5, 2.5, 3.5, etc.)
    ax.contourf(
        xx_unscaled,
        yy_unscaled,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap=cmap_3,
        alpha=0.4
    )

    # 8) Plot data points
    df_plot = pd.concat([df_train, df_test], axis=0)
    for class_id_raw in sorted(df_plot[target_col].unique()):
        class_id = int(class_id_raw)
        subdf = df_plot[df_plot[target_col] == class_id]
        if subdf.empty:
            continue
        ax.scatter(
            subdf[xcol], subdf[ycol],
            color=class_colors[class_id],
            edgecolors='k',
            marker='o',
            alpha=0.9,
            label=None
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs {ycol}\nAcc = {test_acc:.2f}")

    return test_acc, y_test, y_test_pred  # [ADDED: return ground truth & preds]


########################################
# 5) Generate subplots (only 2 features => 1 subplot here, or more)
########################################
# fig, axes = plt.subplots(1, len(all_pairs), figsize=(6*len(all_pairs),6)) \
#     if len(all_pairs)>1 else (plt.figure(), [plt.gca()])

# 5) Generate subplots (1 plot if 1 pair, else multiple)
if len(all_pairs) == 1:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    axes = [ax]
else:
    fig, axes = plt.subplots(1, len(all_pairs), figsize=(6*len(all_pairs), 6))
    axes = axes.flatten()


for i, (xcol, ycol) in enumerate(all_pairs):
    ax = axes[i] if len(all_pairs) > 1 else axes[0]  # <- use axes[0] not axes

    test_acc, y_test_true, y_test_pred = train_multiclass_svm_and_plot(
        df, idx_train, idx_test, xcol, ycol, ax
    )

    ax = axes[i] if len(all_pairs) > 1 else axes[0]

    # … training / plotting on ax …
    cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1, 2, 3])
    metrics_df = make_metrics_df(cm)

    # ---- stand‑alone confusion‑matrix figure -----------------------
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, ax=ax_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.arange(NUM_CLASSES),
                yticklabels=np.arange(NUM_CLASSES),
                vmin=0.0, vmax=50)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix – {xcol} vs {ycol}")
    fig_cm.tight_layout()
    # plt.close(fig_cm)  # keep notebook uncluttered

    # ---- per‑class metrics figure ----------------------------------
    fig_met, ax_met = plt.subplots(figsize=(6, 4))
    sns.heatmap(metrics_df, ax=ax_met, annot=True, fmt=".2f",
                cmap="Reds_r", linewidths=.5, linecolor="grey",
                vmin=0.0, vmax=1.0)
    ax_met.set_ylabel("Metric")
    ax_met.set_xlabel("Class")
    ax_met.set_title(f"Per‑Class Metrics – {xcol} vs {ycol}")
    fig_met.tight_layout()
    # plt.close(fig_met)

# Build a single legend with 4 colored patches
legend_handles = []
legend_labels_corrected = [
    "Class 0: CFR>2, FFR>0.8",
    "Class 1: CFR>2, FFR<0.8",
    "Class 2: CFR<2, FFR>0.8",
    "Class 3: CFR<2, FFR<0.8"
]

# -------------------------------------------------------------------
# finish composite figure, add legend, THEN show once
if len(all_pairs) > 1:
    fig.legend(legend_handles, legend_labels_corrected,
               loc="lower center", ncol=2, frameon=True, fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
else:
    axes[0].legend(legend_handles, loc="best", fontsize=11)

plt.show()  # ← single global show


for class_id in [0,1,2,3]:
    handle = mlines.Line2D(
        [], [],
        color=class_colors[class_id],
        marker='o',
        markersize=10,
        markeredgecolor='k',
        linewidth=0,
        label=legend_labels_corrected[class_id]
    )
    legend_handles.append(handle)

if len(all_pairs)>1:
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=2,
        frameon=True,
        fontsize=11
    )
    plt.tight_layout(rect=[0,0.07,1,1])
else:
    plt.legend(handles=legend_handles, loc='best', fontsize=11)
