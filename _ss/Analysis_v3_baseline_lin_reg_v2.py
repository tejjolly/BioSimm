#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 04:58:38 2025

@author: tejjolly

Description:
    - Perform baseline (LinearRegression) for three targets: [FFR, CFR, CFR_FFR].
    - Drop "Average Flow" from predictors if the target is CFR or CFR_FFR.
    - Do a train/test split (80/20).
    - Perform 5-fold cross-validation on the training set to get mean ± std R².
    - Plot:
       1) Test-set predictions vs actual (3 subplots in one figure)
       2) Permutation importance (3 subplots in one figure)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

###############################################################################
# 1) LOAD & INITIAL CLEANUP
###############################################################################
df = pd.read_csv('summary.csv')

# Focus on hyperemic runs only (if relevant)
df = df[df['Condition'] == 'Hyperemic']

# Rename columns for convenience
df.rename(columns={
    'P_d/P_a':  'FFR',       # original: "P_d/P_a"
    'CFR/FFR':  'CFR_FFR'    # original: "CFR/FFR"
}, inplace=True)

###############################################################################
# 2) DEFINE BASE FEATURE LIST & TARGETS
###############################################################################
base_features = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'HMR',
    'HSR',
    'WSS',
    'BMR/HMR'
]
targets = ['FFR', 'CFR', 'CFR_FFR']

###############################################################################
# 3) FUNCTION: TRAIN BASELINE REGRESSION + PERM IMPORTANCE FOR A SINGLE TARGET
###############################################################################
def train_linear_and_importance(df, target_col):
    """
    - Removes 'Average Flow' if target is CFR or CFR_FFR.
    - Drops NA in [features + target].
    - Splits 80/20 train/test.
    - Performs 5-fold cross-validation on the training set to get mean ± std R^2.
    - Fits LinearRegression (with StandardScaler).
    - Returns (model, X_test, y_test, predictions, features_used, etc.)
    """
    # 1) Decide which features to use
    features = list(base_features)
    if target_col in ['CFR', 'CFR_FFR'] and 'Average Flow' in features:
        features.remove('Average Flow')

    # 2) Drop NA
    df_model = df.dropna(subset=features + [target_col])
    if df_model.empty:
        print(f"\n[WARNING] No valid rows for target={target_col}. Skipping.")
        return None

    X = df_model[features]
    y = df_model[target_col]

    # 3) Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5) 5-fold CV on the training set
    #    We'll do KFold(5) on the scaled training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    linreg_cv = LinearRegression()
    cv_scores = cross_val_score(
        linreg_cv,
        X_train_scaled,
        y_train,
        scoring='r2',
        cv=kf
    )
    mean_cv = np.mean(cv_scores)
    std_cv  = np.std(cv_scores)

    print(f"\n=== {target_col} ===")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"5-fold CV R^2: mean={mean_cv:.3f}, std={std_cv:.3f}")

    # 6) Fit on the entire training set
    linreg = LinearRegression()
    linreg.fit(X_train_scaled, y_train)

    y_pred_test = linreg.predict(X_test_scaled)

    # 7) Evaluate on the test set
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"Test R^2: {test_r2:.3f}")

    # Return everything needed for plotting
    return {
        'model': linreg,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'features_used': features
    }

###############################################################################
# 4) RUN THE PIPELINE FOR ALL THREE TARGETS & STORE RESULTS
###############################################################################
results = {}
for tgt in targets:
    out = train_linear_and_importance(df, tgt)
    if out is not None:
        results[tgt] = out

# If any target had no data, it won't appear in `results`.
available_targets = list(results.keys())
if not available_targets:
    print("No valid targets found. Exiting.")
    exit()

###############################################################################
# 5) MAKE 1 FIGURE WITH 3 SUBPLOTS FOR TEST-SET SCATTER (LINEAR REGRESSION)
###############################################################################
fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(18, 6))
fig_scatter.suptitle("Test-Set Predictions vs. Actual (LinearRegression)")

for i, tgt in enumerate(available_targets):
    ax = axes_scatter[i]
    r = results[tgt]

    y_test = r['y_test']
    y_pred = r['y_pred_test']

    ax.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.set_xlabel(f"Actual {tgt}")
    ax.set_ylabel(f"Predicted {tgt}")
    ax.set_title(f"{tgt}")

plt.tight_layout()
plt.show()

###############################################################################
# 6) MAKE 1 FIGURE WITH 3 SUBPLOTS FOR PERMUTATION IMPORTANCE
#    (Using the fitted baseline LinearRegression for each target).
###############################################################################
fig_import, axes_import = plt.subplots(1, 3, figsize=(18, 6))
fig_import.suptitle("Permutation Importance (LinearRegression)")

for i, tgt in enumerate(available_targets):
    ax = axes_import[i]
    r = results[tgt]
    # We have the fitted model, X_test, y_test, and the feature list:
    model = r['model']
    scaler = r['scaler']
    features = r['features_used']
    X_test = r['X_test']
    y_test = r['y_test']

    # Permutation importance uses the data as the model expects it (scaled).
    X_test_scaled = scaler.transform(X_test)

    perm = permutation_importance(
        model,
        X_test_scaled,
        y_test,
        scoring='r2',
        n_repeats=10,
        random_state=42
    )
    importances = perm.importances_mean
    indices = np.argsort(importances)[::-1]

    sorted_feats = [features[idx] for idx in indices]
    sorted_imps  = importances[indices]

    ax.barh(sorted_feats, sorted_imps)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (R^2 decrease)")
    ax.set_title(f"{tgt}")

plt.tight_layout()
plt.show()
