#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sanity‑check script – v7.3
================================
Just prints **how many samples of each class** appear
* in every CV validation fold
* in the final **test set**
No training, metrics or plots.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
feature_cols   = ["HMR", "P_Loss_Coeff", "BMR/HMR"]
num_classes    = 4
n_splits       = 5
random_state   = 42

test_size      = 0.20

# ────────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data.csv")
mask = (
    (df["Condition"] == "Hyperemic") &
    df["discord"].notna() &
    df["P_Loss_Coeff"].notna()
)
df = df[mask]

df_model = df[feature_cols + ["discord"]].dropna()
X = df_model[feature_cols].values.astype(np.float32)
y = df_model["discord"].astype(int).values

print("Total samples:", len(y))
print("Overall class counts:")
print({cls: int(sum(y == cls)) for cls in range(num_classes)})

# ────────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (stratified)
# ────────────────────────────────────────────────────────────────────────────────
_, X_test, _, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state)

test_counts = {cls: int(sum(y_test == cls)) for cls in range(num_classes)}
print("\nTest‑set class counts (size =", len(y_test), "):")
print(test_counts)

# ────────────────────────────────────────────────────────────────────────────────
# K‑FOLD SPLIT  (on *full* dataset; if you want train‑only split first)
# ────────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

print("\nValidation fold class counts:")
for fold, (_, val_idx) in enumerate(skf.split(X, y), 1):
    y_val = y[val_idx]
    counts = {cls: int(sum(y_val == cls)) for cls in range(num_classes)}
    print(f"  Fold {fold}:", counts)
