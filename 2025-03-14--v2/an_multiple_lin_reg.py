#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 05:11:50 2025

@author: tejjolly

Description:
    - Single-Output Linear Regression Example (predicting 'CFR/FFR').
    - Use baseline LinearRegression, no Ridge/Lasso.
    - Includes 5-fold cross-validation on the training set for mean ± std R^2.
    - Produces two plots:
       1) Predicted vs Actual 'CFR/FFR' (test set).
       2) Permutation importance (test).
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

###############################################################################
# 1) LOAD & CLEAN DATA
###############################################################################
df = pd.read_csv('../data/data.csv')

# Keep only hyperemic runs (if that's your usage)
df = df[df['Condition'] == 'Hyperemic']

# Rename columns as needed (optional: if you want to treat 'P_d/P_a' as a feature 'FFR')
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

feature_cols = [
    # 'Stenosis Percentage',
    'Length',
    'Width',
    'HMR',
    'HSR',
    'BMR/HMR',

    'P_Loss_Coeff',
    'WSS_TE',
    'WSS_LE',
    'WSS_TE_Area',
    'WSS_LE_Area',
    'WSS_Area_Bifur',
    'WSS_Bif',
    'WSS_LMB',
    'WSS_min',
    'WSS_TE_min',
    'WSS_LE_min',
    'WSS_TE_Area_min',
    'WSS_Area_Bifur_min',
    'v_distal'
    # 'Average Flow',
    # 'Max Flow',
    # 'FFR',
]
target_col = 'CFR/FFR'  # single target

# Drop rows missing any required columns
df_model = df.dropna(subset=feature_cols + [target_col])

X = df_model[feature_cols]
y = df_model[target_col]  # shape (n_samples, ), a single column

print("Data shape (features):", X.shape, "| Target shape:", y.shape)

###############################################################################
# 2) TRAIN/TEST SPLIT (20% TEST)
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=41
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

###############################################################################
# 3) 5-FOLD CV ON THE TRAINING SET (Mean ± Std R²)
###############################################################################
# We'll build a pipeline that scales X -> linear reg, then compute cross_val_score.

kf = KFold(n_splits=2, shuffle=True, random_state=42)
pipe_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

cv_scores = cross_val_score(
    pipe_baseline,
    X_train,
    y_train,
    scoring='r2',
    cv=kf
)
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)

print("\n=== Baseline Single‐Output Linear Regression ===")
print(f"5-fold CV R^2: mean={cv_mean:.3f}, std={cv_std:.3f}")

###############################################################################
# 4) FINAL TRAIN/TEST TRAINING
###############################################################################
# Fit a model on the entire training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

# Predictions => shape (n_test,)
y_test_pred = linreg.predict(X_test_scaled)

# R² on test set
test_r2  = r2_score(y_test, y_test_pred)
print(f"\nFinal Test R^2 = {test_r2:.3f}")

###############################################################################
# 5) PLOT #1: Predicted vs Actual (Test Set)
###############################################################################
plt.figure()
plt.scatter(y_test, y_test_pred, edgecolors='k', alpha=0.8)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel(f"Actual {target_col}")
plt.ylabel(f"Predicted {target_col}")
plt.title(f"{target_col} (Test) - Linear Regression")
plt.show()

###############################################################################
# 6) PLOT #2: PERMUTATION IMPORTANCE (Test Set)
###############################################################################
perm = permutation_importance(
    linreg,
    X_test_scaled,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

sorted_feats = [feature_cols[i] for i in indices]
sorted_imps  = importances[indices]

plt.figure()
plt.barh(sorted_feats, sorted_imps)
plt.gca().invert_yaxis()
plt.xlabel("Importance (R^2 decrease)")
plt.title(f"Permutation Importance ({target_col})")
plt.tight_layout()
plt.show()
