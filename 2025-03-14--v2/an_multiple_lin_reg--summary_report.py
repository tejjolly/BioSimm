#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single‑output Linear Regression (predicting ‘CFR/FFR’)
• Optional hold‑out: set test_size = 0 to train on 100 % of the data.
• 5‑fold (here 2‑fold for speed) CV on the training split.
• If a test split exists, produce:
      1) Predicted‑vs‑Actual scatter (test set)
      2) Permutation‑importance bar plot (test set)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ───────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN
# ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/data.csv")
df = df[df["Condition"] == "Hyperemic"].copy()
df.rename(columns={"P_d/P_a": "FFR"}, inplace=True)

feature_cols = ["HMR", "BMR/HMR", "P_Loss_Coeff"]
target_col   = "CFR/FFR"

df_model = df.dropna(subset=feature_cols + [target_col])
X, y = df_model[feature_cols], df_model[target_col]
print("Data shape:", X.shape, "| Target:", y.shape)

# ───────────────────────────────────────────────────────────────────────
# 2) TRAIN / TEST SPLIT   (set test_size = 0.0 for “no test set”)
# ───────────────────────────────────────────────────────────────────────
test_size = 0.0                  # ← change here
if test_size and test_size > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=41
    )
else:
    X_train, y_train = X, y
    X_test  = pd.DataFrame(columns=X.columns)   # empty holders
    y_test  = pd.Series(dtype=float)

print(f"Train size: {X_train.shape},  Test size: {X_test.shape}")

# ───────────────────────────────────────────────────────────────────────
# 3) CROSS‑VALIDATION (training split only)
# ───────────────────────────────────────────────────────────────────────
kf   = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("linreg", LinearRegression())])

cv_scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=kf)
print(f"\n5‑fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ───────────────────────────────────────────────────────────────────────
# 4) FINAL FIT ON TRAINING DATA
# ───────────────────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X_train)
linreg = LinearRegression().fit(scaler.transform(X_train), y_train)

# ───────────────────────────────────────────────────────────────────────
# 5) TEST‑SET EVALUATION & PLOTS  (only if test set exists)
# ───────────────────────────────────────────────────────────────────────
if len(X_test):
    X_test_s = scaler.transform(X_test)
    y_pred   = linreg.predict(X_test_s)
    test_r2  = r2_score(y_test, y_pred)
    print(f"Test‑set R² = {test_r2:.3f}")

    # 5a)  Predicted vs Actual
    plt.figure()
    plt.scatter(y_test, y_pred, facecolor="#5E9096", edgecolor="k", alpha=.8)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title("Predicted vs Actual (test)")
    plt.tight_layout(); plt.show()

    # 5b)  Permutation importance
    perm = permutation_importance(
        linreg, X_test_s, y_test, scoring="r2",
        n_repeats=10, random_state=42
    )
    imp_mean = perm.importances_mean
    order    = np.argsort(imp_mean)[::-1]
    plt.figure()
    plt.barh(np.array(feature_cols)[order], imp_mean[order])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (ΔR²)")
    plt.title("Permutation Importance (test)")
    plt.tight_layout(); plt.show()
else:
    print("\nNo test split requested; skipping test‑set plots.")