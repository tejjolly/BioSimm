#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:51:38 2025

@author: tejjolly
"""
"""
Train a multi-class SVM on 'discord' (4-class target).
Use 4 distinct colors for each class in both the decision region background
and the data points themselves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap

########################################
# 1) Load summary.csv and filter
########################################
df = pd.read_csv("summary.csv")

# Example: focus only on hyperemic
df = df[df['Condition'] == 'Hyperemic'].copy()

# Convert 'discord' to integer if needed
df['discord'] = pd.to_numeric(df['discord'], errors='coerce')

# Convert Location to numeric codes if needed
df['Location_Numeric'] = df['Location'].map({'LAD': 0, 'LCX': 1})

# The 6 features:
feature_candidates = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'HMR',
    # 'Location_Numeric'
]

# Drop rows missing needed features or 'discord'
df = df.dropna(subset=feature_candidates + ['discord'])
df = df[df['discord'] != 3]

########################################
# 2) Global 80/20 train/test split
########################################
all_indices = df.index.to_numpy()
y_dummy = df['discord'].values  # used for stratify
idx_train, idx_test = train_test_split(
    all_indices,
    test_size=0.05,
    stratify=y_dummy,
    random_state=10
)

print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

########################################
# 3) Setup
########################################

# cmap = get_cmap("cividis", 4)  # Generate 4 distinct colors from Set3 colormap
# class_colors = [cmap(i) for i in range(4)]

# We'll define 4 colors (one per discord class). For example:
class_colors = ["#AAFFAA","#AAAADD","#FFDDAA","#FFAAAA"]

# class 0 -> #AAFFAA (pastel green)
# class 1 -> #AAAADD (pastel blue/purple)
# class 2 -> #FFDDAA (pastel orange-ish)
# class 3 -> #FFAAAA (pastel red)

# We'll use these colors for the background and the data points.
# Build a ListedColormap
cmap_4 = ListedColormap(class_colors)

# We'll do all pairs from the 6 features
all_pairs = list(combinations(feature_candidates, 2))  # 15 pairs total

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
    Trains a multi-class SVM (OvR) on (xcol, ycol) -> `discord`.
    Plots the 2D decision boundary with a 4-color background.
    Data points are circles with the same color as their class.
    Returns the trained model's test accuracy.
    """
    # 1) Build subsets
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])
    df_plot  = pd.concat([df_train, df_test], axis=0)

    X_train = df_train[[xcol, ycol]].values
    X_test  = df_test[[xcol, ycol]].values
    y_train = df_train[target_col].astype(int).values
    y_test  = df_test[target_col].astype(int).values

    # 2) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3) Train SVC multi-class
    svm_clf = SVC(kernel='poly',degree=3, C=1, gamma=10, decision_function_shape='ovo')
    svm_clf.fit(X_train_scaled, y_train)
    test_acc = svm_clf.score(X_test_scaled, y_test)

    # 4) Make mesh for background
    x_min, x_max = X_train_scaled[:, 0].min()-0.5, X_train_scaled[:, 0].max()+0.5
    y_min, y_max = X_train_scaled[:, 1].min()-0.5, X_train_scaled[:, 1].max()+0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 5) Inverse transform the mesh grid
    xx_unscaled = xx*scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy*scaler.scale_[1] + scaler.mean_[1]

    # 6) Plot background
    ax.contourf(
        xx_unscaled,
        yy_unscaled,
        Z,
        levels=[-0.5,0.5,1.5,2.5,3.5],  # boundaries for classes 0..3
        cmap=cmap_4,
        alpha=0.4
    )

    # 7) Plot data points
    # We'll color each point by its 'discord' class using the same 4 colors.
    for class_id in [0,1,2,3]:
        subdf = df_plot[df_plot[target_col] == class_id]
        if subdf.empty:
            continue
        ax.scatter(
            subdf[xcol], subdf[ycol],
            color=class_colors[class_id],
            edgecolors='k',  # black edges
            marker='o',
            alpha=0.9,
            label=None  # we'll do a single legend at the end
        )

    return test_acc

########################################
# 5) Generate subplots (3 x 5 = 15)
########################################
fig, axes = plt.subplots(2, 5, figsize=(16,6))
axes = axes.flatten()

for i, (xcol, ycol) in enumerate(all_pairs):
    ax = axes[i]
    
    # Call the function instead of repeating the logic
    test_acc = train_multiclass_svm_and_plot(df, idx_train, idx_test, xcol, ycol, ax)
    
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Acc={test_acc:.2f}")

# Build a single legend with 4 colored patches
# We'll do it as a set of lines or patches in 4 different colors
legend_handles = []
legend_labels = [
    "CFR>2, FFR>0.8",
    "CFR>2, FFR<0.8",
    "CFR<2, FFR>0.8",
    "CFR<2, FFR<0.8"
]

for class_id in [0,1,2,3]:
    handle = mlines.Line2D(
        [],[],
        color=class_colors[class_id], 
        marker='o',
        markersize=10,
        markeredgecolor='k',
        linewidth=0,
        label=legend_labels[class_id]
    )
    legend_handles.append(handle)

fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=4,
    frameon=True,
    fontsize=11
)

plt.tight_layout(rect=[0,0.04,1,1])  # leave bottom space for legend
plt.savefig("multi_class_svm_subplots_same_color.png", dpi=900)
plt.show()
