#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottleneck‑net (multiclass) + out‑of‑fold diagnostics  —NBSPv7.2
============================================================
* 5‑fold (Stratified) CV on the **training** split
* Per‑class Precision / Sensitivity / Specificity
  → **mean ± std dev** printed after CV
  → heat‑map shows the means
* Training‑ vs‑validation loss curves (CV‑averaged ± 1σ bands)
* Final fit on full‑training data → evaluation on the held‑out **test** set
* Two scatter plots on the P_d/P_a – CFR plane
"""

import os, pathlib, matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    classification_report
)
from scipy.stats import chi2_contingency
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ────────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & flags
# ────────────────────────────────────────────────────────────────────────────────
inspect_data        = False #outputs .csv of data
train_model         = False
p_val_flag          = False
user_bottleneck_dim = 1
user_epochs         = 400
user_batch_size     = 8
user_hidden_dim     = 8
user_lr             = 2e-3
user_cos_max        = user_epochs
n_splits            = 5                      # ← use 5‑fold CV
num_classes         = 4
test_size           = 0.333
random_state        = 41
device              = torch.device("cpu")

# ────────────────────────────────────────────────────────────────────────────────
# Save plot function
# ────────────────────────────────────────────────────────────────────────────────
save_dir = f"./{user_bottleneck_dim}-d_{user_epochs}-epoch_plots"
pathlib.Path(save_dir).mkdir(exist_ok=True)          # make it once
def savefig(name, **kw):                             # tiny helper
    plt.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches="tight", **kw)


# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREP DATA
# ────────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(
    "/Users/tejjolly/Documents/BioSimm/Simulations/"
    "Post_Processing/data/data.csv"
)

mask = (
    # (df_raw["Condition"] == "Hyperemic") &
    # df_raw["discord"].notna() &
    # df_raw["P_Loss_Coeff"].notna()
)
# df_raw = df_raw[mask]

feature_cols = ["Length", "Stenosis Percentage"]

plt.figure(figsize=(10, 4))

for i, col in enumerate(feature_cols):
    print(df_raw[col].dtype)
    print(df_raw[col].head())
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    print(f"{col} count:", df_raw[col].count())
    mean = df_raw[col].mean()
    std  = df_raw[col].std()
    median = df_raw[col].median()

    plt.subplot(1, 2, i + 1)
    # sns.kdeplot(df_raw[col], fill=True, bw_adjust=0.8)
    sns.histplot(df_raw[col], bins=7, stat="count", alpha=0.3)
    # sns.kdeplot(df_raw[col], color="blue", linewidth=2)
    # plt.axvline(median, color="red", linestyle="--", linewidth=1.2, label="Median")
    # plt.axvline(mean - 1 * std, color='k', linestyle=":", linewidth=2, label="±1 SD" if i == 0 else "")
    # plt.axvline(mean + 1 * std, color='k', linestyle=":", linewidth=2)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("Count" if i == 0 else "")
    plt.grid(False)

plt.tight_layout()
savefig("sample_distribution.png")
plt.show()

# df_model = df_raw[feature_cols + ["discord"]].dropna()
# X_full   = df_model[feature_cols].values.astype(np.float32)
# y_full   = df_model["discord"].astype(int).values
#
# print("Unique labels:", np.unique(y_full), "| Total samples:", len(y_full))
# if inspect_data:
#     df_model.to_csv("data_cleaned.csv", index=False)