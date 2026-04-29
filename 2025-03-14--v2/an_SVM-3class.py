#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:51:38 2025

@author: tejjolly

Description:
    - Train a multi-class SVM on 'discord' (fused to 3-class target).
    - Plot subplots for each pair of features showing decision boundaries and data.
    - Includes confusion matrix, F1 score, and exports to Excel/plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from itertools import combinations
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns

########################################
# 1) Load summary.csv and filter
########################################
df = pd.read_csv("./summary2.csv")
df = df[df['Condition'] == 'Hyperemic'].copy()
df['discord'] = pd.to_numeric(df['discord'], errors='coerce')

# Fuse Class 1 and 2 into one class
df['discord_fused'] = df['discord'].replace({1: 1, 2: 1, 3: 2})

df['Location_Numeric'] = df['Location'].map({'LAD': 0, 'LCX': 1})

df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

feature_candidates = ['v_distal', 'BMR/HMR']
third_features = ['HSR', 'WSS_TE', 'WSS_LE', 'WSS_Area_Bifur', 'WSS_TE_Area_min', 'WSS_LE_Area_min', 'WSS_Area_Bifur_min']  # or any other features you want to sweep
df = df.dropna(subset=feature_candidates + ['discord_fused'])

########################################
# 2) Global 80/20 train/test split
########################################
all_indices = df.index.to_numpy()
y_dummy = df['discord_fused'].values
test_size = 0.5

idx_train, idx_test = train_test_split(
    all_indices,
    test_size=test_size,
    stratify=y_dummy,
    random_state=10
)

print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

########################################
# 3) Color and Label Setup
########################################
class_colors = [
    "#AAFFAA",  # Class 0
    "#AAAADD",  # Fused Class 1/2
    "#FFAAAA",  # Class 3 becomes Class 2
]
cmap_3 = ListedColormap(class_colors)

legend_labels_corrected = [
    "Class 0: CFR>2, FFR>0.8",
    "Class 1: Discordant (CFR>2 & FFR<0.8 OR CFR<2 & FFR>0.8)",
    "Class 2: CFR<2, FFR<0.8"
]

all_pairs = list(combinations(feature_candidates, 2))

########################################
# 4) Multi-class SVM function
########################################
def train_multiclass_svm_and_plot(df, train_idx, test_idx, xcol, ycol, ax, target_col='discord_fused'):
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])

    X_train = df_train[[xcol, ycol]].values
    X_test = df_test[[xcol, ycol]].values
    y_train = df_train[target_col].astype(int).values
    y_test = df_test[target_col].astype(int).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_clf = SVC(kernel='linear', C=0.1, decision_function_shape='ovo')
    svm_clf.fit(X_train_scaled, y_train)

    test_acc = svm_clf.score(X_test_scaled, y_test)
    y_test_pred = svm_clf.predict(X_test_scaled)

    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

    ax.contourf(
        xx_unscaled,
        yy_unscaled,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=ListedColormap(class_colors),
        alpha=0.4
    )

    df_plot = pd.concat([df_train, df_test], axis=0)
    for class_id in sorted(df_plot[target_col].unique()):
        class_id = int(class_id)  # <- convert to plain int
        subdf = df_plot[df_plot[target_col] == class_id]
        ax.scatter(
            subdf[xcol], subdf[ycol],
            color=class_colors[class_id],
            edgecolors='k',
            marker='o',
            alpha=0.9
        )

    return test_acc, y_test, y_test_pred

########################################
# 5) Generate subplots and compute metrics
########################################
results_summary = {}

def compute_per_class_metrics(conf_matrix):
    """
    Given a confusion matrix for multi-class classification,
    compute precision, Sensitivity, specificity, and F1 for each class.

    Parameters:
        conf_matrix (ndarray): square matrix (num_classes x num_classes)

    Returns:
        dict: metrics per class
    """
    num_classes = conf_matrix.shape[0]
    metrics = {}

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"Class {i}"] = {
            "Precision": round(precision, 3),
            "Sensitivity": round(recall, 3),
            "Specificity": round(specificity, 3),
            "F1 Score": round(f1, 3)
        }

    return metrics

for third_feat in third_features:
    print(f"\n=== Adding third dimension: {third_feat} ===")

    # Combine with the existing 2D features
    features_3d = feature_candidates + [third_feat]

    # Drop rows missing any of the selected 3 features
    df_cleaned = df.dropna(subset=features_3d + ['discord_fused']).copy()
    X = df_cleaned[features_3d].values
    y = df_cleaned['discord_fused'].astype(int).values

    # New train/test split just for this setup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=24
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate
    clf = SVC(kernel='linear', C=0.1, decision_function_shape='ovo')
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = clf.score(X_test_scaled, y_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics = compute_per_class_metrics(cm)

    # Print basic info
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)
    print(f"Weighted F1 Score: {f1:.2f}")

    # Store results
    results_summary[third_feat] = {
        "confusion_matrix": cm,
        "accuracy": acc,
        "f1_score": f1,
        "metrics": metrics
    }

    # Optional: save per-metric heatmaps per third dimension
    metrics_df = pd.DataFrame(metrics).T

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=[f"Pred {i}" for i in range(3)],
                yticklabels=[f"True {i}" for i in range(3)])
    plt.title(f"Confusion Matrix\n+ {third_feat}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_3D_{third_feat}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title(f"{feature_candidates[0]} & {feature_candidates[1]} & {third_feat}")
    plt.tight_layout()
    plt.savefig(f"metrics_heatmap_3D_{third_feat}.png", dpi=300)
    plt.show()

########################################
# 6) Metrics Computation
########################################
def compute_per_class_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    metrics = {}
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"Class {i}"] = {
            "Precision": round(precision, 3),
            "Sensitivity": round(recall, 3),
            "Specificity": round(specificity, 3),
            "F1 Score": round(f1, 3)
        }

    return metrics

metrics = compute_per_class_metrics(cm)

for cls, values in metrics.items():
    print(f"\n{cls}")
    for k, v in values.items():
        print(f"  {k}: {v}")

########################################
# 7) Save to Excel and Plot Heatmaps
########################################
metrics_df = pd.DataFrame(metrics).T

# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
#             xticklabels=[f"Pred {i}" for i in range(3)],
#             yticklabels=[f"True {i}" for i in range(3)])
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("confusion_matrix_heatmap_fused.png", dpi=300)
# plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title("Per-Class Performance Metrics")
plt.tight_layout()
plt.savefig("per_class_metrics_heatmap_fused.png", dpi=300)
plt.show()
