#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:19:17 2025

@author: tejjolly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines  # for custom black legend handles

########################################
# 1) LOAD YOUR SUMMARY CSV
########################################
df = pd.read_csv("summary.csv")

# Focus only on hyperemic rows if that's relevant
df = df[df['Condition'] == 'Hyperemic']

# Drop rows that lack needed columns
required_cols = ['P_d/P_a','CFR','CFR/FFR','HSR','HMR','BMR/HMR','Average Flow','discord']
df = df.dropna(subset=required_cols)

# 1a) Convert the original CFR/FFR columns to binary for classification
df['FFR'] = (df['P_d/P_a'] > 0.8).astype(int)     # 1 if FFR>0.8
df['CFR'] = (df['CFR'] > 2.0).astype(int)         # 1 if CFR>2.0
df['CFR_FFR'] = (df['CFR/FFR'] > 2.0).astype(int) # 1 if CFR/FFR>2.0

########################################
# 2) GLOBAL SPLIT (80/20) 
########################################
all_indices = df.index.to_numpy()
y_dummy = df['FFR'].values  # just a placeholder for stratify

idx_train_full, idx_test = train_test_split(
    all_indices, test_size=0.2, stratify=y_dummy, random_state=10
)

print("Train+Val size:", len(idx_train_full), "Test size:", len(idx_test))

########################################
# 3) HELPER: train_svm_and_plot
########################################
def train_svm_and_plot(
    df,
    train_idx, test_idx,
    xcol, ycol,
    target_col, target_name
):
    """
    - df: full dataframe (with 'discord' column)
    - train_idx, test_idx: arrays of row indices
    - xcol, ycol: feature column names (2D)
    - target_col: name of binary target (0 or 1)
    - target_name: for plot title
    """

    # 1) Combine train+test for plotting. We'll still train on train_idx only.
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])
    df_plot  = pd.concat([df_train, df_test], axis=0)  # for plotting all points

    # 2) Extract arrays
    X_train_unscaled = df_train[[xcol, ycol]].values
    X_test_unscaled  = df_test[[xcol, ycol]].values
    y_train = df_train[target_col].values
    y_test  = df_test[target_col].values

    # 3) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test_scaled  = scaler.transform(X_test_unscaled)

    # 4) Train SVM
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X_train_scaled, y_train)

    test_acc = svm_clf.score(X_test_scaled, y_test)
    print(f"[{target_name} | {xcol} vs {ycol}] Test Acc: {test_acc:.3f}")

    # 5) Build decision boundary in scaled space
    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Transform mesh grid back to unscaled space
    xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

    # 6) Plot 
    plt.figure(figsize=(6,5))
    # Show classification regions
    plt.contourf(xx_unscaled, yy_unscaled, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

    # Dictionary for discord categories => marker shapes
    # (Matches your specification)
    discord_markers = {
        0: {'marker':'o', 'label':'CFR>2, FFR>0.8'},
        1: {'marker':'^', 'label':'CFR>2, FFR<0.8'},
        2: {'marker':'v', 'label':'CFR<2, FFR>0.8'},
        3: {'marker':'s', 'label':'CFR<2, FFR<0.8'}
    }

    # Plot each discord category
    # We'll color the points by target_col (0=red, 1=green)
    # Then separate df_plot into train vs test if you *still* want a different edge or alpha
    # but here we do them the same for simplicity.
    for disc_val, info in discord_markers.items():
        subdf = df_plot[df_plot['discord'] == disc_val]
        if subdf.empty:
            continue

        # For subdf, we have two possible classes in target_col: 0 or 1
        # We'll color them differently but keep the same marker
        # Let's do one scatter for class=0 (red), one for class=1 (green)
        subdf_0 = subdf[subdf[target_col] == 0]
        subdf_1 = subdf[subdf[target_col] == 1]

        # For each subset
        plt.scatter(
            subdf_0[xcol],
            subdf_0[ycol],
            c='red',
            edgecolors='k',
            alpha=0.7,
            marker=info['marker'],
            label=None  # We'll add a custom black legend below
        )
        plt.scatter(
            subdf_1[xcol],
            subdf_1[ycol],
            c='green',
            edgecolors='k',
            alpha=0.7,
            marker=info['marker'],
            label=None
        )

    # Build a black legend with 4 handles
    legend_handles = []
    for disc_val, info in discord_markers.items():
        handle = mlines.Line2D(
            [], [], 
            color='black', markerfacecolor='black', markeredgecolor='black',
            marker=info['marker'], linestyle='None',
            label=info['label']
        )
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, loc='upper right', frameon=True)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"Target: {target_name}\nTest Acc: {test_acc:.2f}")
    plt.tight_layout()
    plt.show()

########################################
# 4) CALL train_svm_and_plot
########################################
targets = [
    # ('FFR',     "FFR > 0.8"),
    # ('CFR',     "CFR > 2.0"),
    ('CFR_FFR', "CFR/FFR > 2.0")
]

predictor_pairs = [
    ('HMR','HSR'),          
    ('HSR','BMR/HMR'),
    ('HSR','Average Flow')
]

for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        train_svm_and_plot(
            df, 
            idx_train_full, idx_test, 
            xcol, ycol, 
            target_col, target_name
        )
        print("----------------------------------------------------")

########################################
# 5) MAKE SUBPLOTS (3x3) 
########################################
import matplotlib.lines as mlines
fig, axes = plt.subplots(3, 1, figsize=(4,7))
axes = axes.flatten()

plot_idx = 0

for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        df_train = df.loc[idx_train_full].dropna(subset=[xcol, ycol, target_col])
        df_test  = df.loc[idx_test].dropna(subset=[xcol, ycol, target_col])
        df_plot  = pd.concat([df_train, df_test])

        X_train_unscaled = df_train[[xcol, ycol]].values
        X_test_unscaled  = df_test[[xcol, ycol]].values
        y_train = df_train[target_col].values
        y_test  = df_test[target_col].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_unscaled)
        X_test_scaled  = scaler.transform(X_test_unscaled)

        svm_clf = SVC(kernel='linear', C=1.0)
        svm_clf.fit(X_train_scaled, y_train)
        test_acc = svm_clf.score(X_test_scaled, y_test)

        x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
        y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
        yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

        ax = axes[plot_idx]
        ax.contourf(xx_unscaled, yy_unscaled, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

        # Markers based on discord
        discord_markers = {
            0: {'marker':'o', 'label':'CFR>2, FFR>0.8'},
            1: {'marker':'^', 'label':'CFR>2, FFR<0.8'},
            2: {'marker':'v', 'label':'CFR<2, FFR>0.8'},
            3: {'marker':'s', 'label':'CFR<2, FFR<0.8'}
        }

        for disc_val, info in discord_markers.items():
            subdf = df_plot[df_plot['discord'] == disc_val]
            if subdf.empty:
                continue
            
            subdf_0 = subdf[subdf[target_col] == 0]
            subdf_1 = subdf[subdf[target_col] == 1]

            # Plot class=0 in red, class=1 in green
            ax.scatter(
                subdf_0[xcol], subdf_0[ycol],
                c='red', edgecolors='k',
                alpha=0.7, marker=info['marker'],
                label=None
            )
            ax.scatter(
                subdf_1[xcol], subdf_1[ycol],
                c='green', edgecolors='k',
                alpha=0.7, marker=info['marker'],
                label=None
            )

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{target_name}\nTest Acc: {test_acc:.2f}")

        plot_idx += 1

# Build a single black legend for the entire figure
legend_handles = []
discord_markers = {
    0: {'marker':'o', 'label':'CFR>2, FFR>0.8'},
    1: {'marker':'^', 'label':'CFR>2, FFR<0.8'},
    2: {'marker':'v', 'label':'CFR<2, FFR>0.8'},
    3: {'marker':'s', 'label':'CFR<2, FFR<0.8'}
}
for disc_val, info in discord_markers.items():
    handle = mlines.Line2D(
        [], [],
        color='black', markerfacecolor='black', markeredgecolor='black',
        marker=info['marker'], linestyle='None',
        label=info['label']
    )
    legend_handles.append(handle)

# # Place a single legend below all subplots
# fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=True)

# # plt.tight_layout(rect=[left, bottom, right, top])
plt.tight_layout(rect=[0,0,1,1])  # leave space at bottom for legend
# plt.savefig("svm_subplots.png", dpi=900)
# plt.show()
