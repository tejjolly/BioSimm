#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 04:58:38 2025

@author: tejjolly

Description:
    - Multi‐Output Regression Example (predicting [FFR, CFR] simultaneously).
    - Use baseline LinearRegression, no Ridge/Lasso.
    - Include 5-fold cross-validation on the training set for mean ± std R^2.
    - Produce three separate plots:
       1) FFR (test) scatter
       2) CFR (test) scatter
       3) Permutation importance (test)
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

# Rename columns as needed
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

our_data = False # Else, all data (Garcia and ours)

if our_data:
    feature_cols = [
        'Stenosis Percentage',
        'Length',
        'Width',
        'HMR',
        'HSR',
        'BMR/HMR',
        # 'Average Flow',
        # 'Max Flow',
        # 'FFR',
        'HSR',
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
    ]
else:
    feature_cols = [
        'HMR',
        'BMR/HMR',
        'P_Loss_Coeff',
    ]

output_cols = ['CFR','FFR']  # Multi‐output: 1 targets
# output_cols = ['discord']

# Drop rows missing any of the required columns
df_model = df.dropna(subset=feature_cols + output_cols)

X = df_model[feature_cols]
Y = df_model[output_cols]  # shape (n_samples, 2)

print("Data shape (features):", X.shape, "| Outputs shape:", Y.shape)

###############################################################################
# 2) TRAIN/TEST SPLIT (20% TEST)
###############################################################################
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.5
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

###############################################################################
# 3) 5‑FOLD CV ON THE TRAINING SET (separate R² for FFR and CFR)
###############################################################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pipe_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# --- FFR ---
cv_scores_ffr = cross_val_score(
    pipe_baseline,
    X_train,
    Y_train['FFR'],        # single output
    scoring='r2',
    cv=kf
)

# --- CFR ---
cv_scores_cfr = cross_val_score(
    pipe_baseline,
    X_train,
    Y_train['CFR'],        # single output
    scoring='r2',
    cv=kf
)

print("\n=== Baseline Multi‑Output Linear Regression (5‑fold CV) ===")
print(f"FFR  R²: mean={cv_scores_ffr.mean():.3f}, std={cv_scores_ffr.std():.3f}")
print(f"CFR  R²: mean={cv_scores_cfr.mean():.3f}, std={cv_scores_cfr.std():.3f}")

###############################################################################
# 4) FINAL TRAIN/TEST TRAINING
###############################################################################
# We now fit one final model on the entire training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train_scaled, Y_train)

# Predictions => shape (n_test, 2)
Y_test_pred = linreg.predict(X_test_scaled)

# (a) R² averaged over both outputs
test_r2_avg  = r2_score(Y_test, Y_test_pred)
# (b) R² per output
test_r2_raw  = r2_score(Y_test, Y_test_pred, multioutput='raw_values')

print(f"\nFinal Test R^2 (avg): {test_r2_avg:.3f}, per target=[FFR, CFR] = {test_r2_raw}")

###############################################################################
# 5) PLOT #1: Predicted vs Actual FFR (Test Set)
###############################################################################
plt.figure()
plt.scatter(Y_test.iloc[:, 0], Y_test_pred[:, 0], color='tab:blue', edgecolors='k', alpha=0.8)
min_val = min(Y_test.iloc[:, 0].min(), Y_test_pred[:, 0].min())
max_val = max(Y_test.iloc[:, 0].max(), Y_test_pred[:, 0].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")
plt.title("FFR (Test) - Baseline Linear Regression")
plt.show()

###############################################################################
# 6) PLOT #2: Predicted vs Actual CFR (Test Set)
###############################################################################
plt.figure()
plt.scatter(Y_test.iloc[:, 1], Y_test_pred[:, 1], color='tab:blue', edgecolors='k', alpha=0.8)
min_val = min(Y_test.iloc[:, 1].min(), Y_test_pred[:, 1].min())
max_val = max(Y_test.iloc[:, 1].max(), Y_test_pred[:, 1].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")
plt.title("CFR (Test) - Baseline Linear Regression")
plt.show()

###############################################################################
# 7) PLOT #3: PERMUTATION IMPORTANCE (Test Set)
###############################################################################
# For multi-output, by default, 'r2' => average R^2 across both outputs
perm = permutation_importance(
    linreg,
    X_test_scaled,
    Y_test,  # shape (n_test,2)
    scoring='r2',    # average R^2 over FFR & CFR
    n_repeats=10,
    random_state=42
)
importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

sorted_feats = [feature_cols[i] for i in indices]
sorted_imps  = importances[indices]

plt.figure()
plt.barh(sorted_feats, sorted_imps)#, color='k')
plt.gca().invert_yaxis()
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importance (Average R^2 across FFR & CFR)")
plt.tight_layout()
plt.show()

# After fitting
coefs = pd.DataFrame(linreg.coef_, columns=feature_cols, index=['FFR', 'CFR'])
print("Coefficient matrix:")
print(coefs)

print("\nIntercepts:")
print(pd.Series(linreg.intercept_, index=['FFR', 'CFR']))
