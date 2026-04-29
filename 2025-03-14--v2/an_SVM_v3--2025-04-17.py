#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5‑fold CV ─ 3‑feature linear‑SVM (4‑class ‘discord’)

* StratifiedKFold, shuffle=True, random_state=42
* Handles class imbalance with class_weight='balanced'
* Reports CV accuracy & F1 (mean ± std)
* Aggregated confusion‑matrix + per‑class metrics
* 3‑D scatter of **all** samples coloured by true label;
  mis‑classified ones marked with a '^'.
"""

# ── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score
)

# ── 1 | Load & prepare data ────────────────────────────────────────────────
df = pd.read_csv("../data/data.csv")

feature_cols = ["P_Loss_Coeff", "BMR/HMR", "HMR"]
df["discord"] = pd.to_numeric(df["discord"], errors="coerce")
df = df.dropna(subset=feature_cols + ["discord"])

X = df[feature_cols].values.astype(float)
y = df["discord"].astype(int).values
print(f"Dataset size: {X.shape[0]} samples")

# ── 2 | Pipeline & CV setup ────────────────────────────────────────────────
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svc",
            SVC(
                kernel="linear",
                C=1.0,
                decision_function_shape="ovo",
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "acc": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}

cv_results = cross_validate(
    pipe, X, y, cv=skf, scoring=scoring, return_train_score=False
)

print("\n5‑fold cross‑validation results")
for key, label in [
    ("test_acc", "Accuracy"),
    ("test_f1_macro", "Macro‑F1"),
    ("test_f1_weighted", "Weighted‑F1"),
]:
    vals = cv_results[key]
    print(f"  {label:12}: {vals.mean():.3f} ± {vals.std():.3f}")

# ── 3 | Out‑of‑fold predictions for global CM ──────────────────────────────
y_pred_oof = cross_val_predict(pipe, X, y, cv=skf)
cm = confusion_matrix(y, y_pred_oof, labels=[0, 1, 2, 3])

acc_oof      = accuracy_score(y, y_pred_oof)
macro_f1_oof = f1_score(y, y_pred_oof, average="macro")

print("\nOOF metrics (aggregated over all folds)")
print(f"  Accuracy : {acc_oof:.3f}")
print(f"  Macro‑F1 : {macro_f1_oof:.3f}")
print("  Confusion matrix (rows = true, cols = pred):\n", cm)

# ── 4 | Per‑class metrics helper ───────────────────────────────────────────
def per_class_metrics(cm_arr: np.ndarray):
    out = []
    for i in range(cm_arr.shape[0]):
        TP = cm_arr[i, i]
        FN = cm_arr[i, :].sum() - TP
        FP = cm_arr[:, i].sum() - TP
        TN = cm_arr.sum() - (TP + FN + FP)
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec  = TP / (TP + FN) if TP + FN else 0.0
        spec = TN / (TN + FP) if TN + FP else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append([prec, rec, spec, f1])
    return pd.DataFrame(
        out,
        index=[f"Class {i}" for i in range(cm_arr.shape[0])],
        columns=["Precision", "Recall", "Specificity", "F1"],
    ).round(3)


metrics_df = per_class_metrics(cm)

# ── 5 | Plot confusion‑matrix ──────────────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar=False,
    xticklabels=[f"Pred {i}" for i in range(4)],
    yticklabels=[f"True {i}" for i in range(4)],
)
plt.title("Confusion Matrix – 5‑fold OOF predictions")
plt.tight_layout()
plt.show()

# ── 6 | Plot per‑class metrics ─────────────────────────────────────────────
plt.figure(figsize=(7, 3))
sns.heatmap(metrics_df, annot=True, cmap="YlOrRd_r", fmt=".3f")
plt.title("Per‑class metrics (OOF)")
plt.tight_layout()
plt.show()

# ── 7 | 3‑D scatter (true labels, mis‑classified '^') ──────────────────────
class_colors = ["#AAFFAA", "#AAAADD", "#FFDDAA", "#FFAAAA"]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

for cls in range(4):
    mask = y == cls
    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        X[mask, 2],
        label=f"Class {cls} (true)",
        color=class_colors[cls],
        edgecolors="k",
        marker="o",
        alpha=0.8,
    )

mis_mask = y != y_pred_oof
ax.scatter(
    X[mis_mask, 0],
    X[mis_mask, 1],
    X[mis_mask, 2],
    facecolors="none",
    edgecolors="k",
    marker="^",
    s=90,
    linewidths=1.3,
    label="Mis‑classified",
)

ax.set_xlabel(feature_cols[0])
ax.set_ylabel(feature_cols[1])
ax.set_zlabel(feature_cols[2])
ax.set_title("3‑D scatter – OOF predictions vs. true labels")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

from matplotlib.animation import FuncAnimation, FFMpegWriter  # or PillowWriter for .gif

# ------------------------------------------------------------
# 1)  build the static 3‑D scatter once (use your X, y, y_pred_oof)
# ------------------------------------------------------------
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection="3d")

class_colors = ["#AAFFAA", "#AAAADD", "#FFDDAA", "#FFAAAA"]
for cls in range(4):
    m = y == cls
    ax.scatter(X[m, 0], X[m, 1], X[m, 2],
               color=class_colors[cls], edgecolors="k", label=f"Class {cls}",
               s=20, alpha=0.9)
mis = y != y_pred_oof
ax.scatter(X[mis, 0], X[mis, 1], X[mis, 2],
           facecolors="none", edgecolors="k", marker="^", s=60, linewidths=1.3,
           label="Mis‑classified")

ax.set_xlabel(feature_cols[0]); ax.set_ylabel(feature_cols[1]); ax.set_zlabel(feature_cols[2])
ax.set_title("OOF predictions vs. true labels")
ax.legend()

# ------------------------------------------------------------
# 2)  animate: spin the azimuth by Δ° each frame
# ------------------------------------------------------------
def update(frame):
    ax.view_init(elev=35, azim=frame)    # keep elev fixed, spin azim
    return ax,

n_frames   = 360                         # 180° gives a nice half‑turn
fps        = 30
anim = FuncAnimation(fig, update, frames=range(0, 360, 360 // n_frames), blit=False)

# # ------------------------------------------------------------
# # 3)  save: MP4 (needs ffmpeg)  ✱or✱  GIF (needs pillow or ImageMagick)
# # ------------------------------------------------------------
# # --- MP4 ---
# writer = FFMpegWriter(fps=fps, bitrate=2400)
# anim.save("svm_3d_spin.mp4", writer=writer, dpi=200)

# --- GIF (fallback) ---
# from matplotlib.animation import PillowWriter
# anim.save("svm_3d_spin.gif", writer=PillowWriter(fps=fps))
