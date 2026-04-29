#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 04:31:11 2025

@author: tejjolly

Description:
    - Perform linear, Ridge, Lasso regression for each target (FFR, CFR, CFR_FFR).
    - Drop "Average Flow" as a feature if target is CFR or CFR_FFR.
    - Plot 3 test-set scatter plots per target (as before).
    - Collect the 6 permutation importance plots (Ridge & Lasso × 3 targets)
      into 2 figures with 3 subplots each, for a cleaner presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

###############################################################################
# 1) LOAD & INITIAL CLEANUP
###############################################################################
df = pd.read_csv('summary.csv')

# Focus on hyperemic runs only (if that's what you want)
df = df[df['Condition'] == 'Hyperemic']

# Rename columns for convenience
df.rename(columns={
    'P_d/P_a':  'FFR',       # original: P_d/P_a
    'CFR/FFR':  'CFR_FFR'    # original: CFR/FFR
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
    # We do NOT include 'FFR', 'CFR', or 'CFR_FFR' in the feature list itself
]

targets = ['FFR', 'CFR', 'CFR_FFR']

# Repeated K-Fold for cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Range of alphas for Ridge/Lasso
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# We'll store final predictions & permutation importances in dictionaries.
model_results = {}    # predictions
perm_imports_lasso = {}
perm_imports_ridge = {}

###############################################################################
# 3) FUNCTION TO RUN THE PIPELINE FOR A SINGLE TARGET
###############################################################################
def run_regression_pipeline_for_target(df, target_col):
    """
    Runs:
      - Drop "Average Flow" if target is CFR or CFR_FFR
      - Baseline LinearRegression + test-set scatter plot
      - RepeatedKFold cross-val
      - Ridge & Lasso with GridSearchCV + scatter plots
      - Permutation importances for Lasso & Ridge
    Returns:
      - y_test
      - y_pred_baseline, y_pred_ridge, y_pred_lasso
      - importances_lasso (features, importance array)
      - importances_ridge (features, importance array)
    """

    print("\n" + "="*80)
    print(f"    REGRESSION PIPELINE FOR TARGET = {target_col}")
    print("="*80)

    # 1) Determine features
    features = list(base_features)  # make a copy
    if target_col in ['CFR', 'CFR_FFR'] and 'Average Flow' in features:
        features.remove('Average Flow')

    # 2) Drop rows missing in features + target
    df_model = df.dropna(subset=features + [target_col])
    print(f"Data shape for {target_col}: {df_model.shape}")
    if len(df_model) == 0:
        print(f"No data for target={target_col}. Skipping.")
        return None

    X = df_model[features]
    y = df_model[target_col]

    # 3) Train/Test split (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    ############################################################################
    # (A) BASELINE LINEAR REGRESSION
    ############################################################################
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    y_pred_train_lr = lr.predict(X_train_scaled)
    y_pred_test_lr  = lr.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train_lr)
    test_r2  = r2_score(y_test,  y_pred_test_lr)

    print("\n=== Baseline Linear Regression ===")
    print(f"Train R^2: {train_r2:.3f}, Test R^2: {test_r2:.3f}")

    # Plot predictions vs actual (TEST set) for Baseline
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test_lr, edgecolor='k', alpha=0.7)
    min_val = min(y_test.min(), y_pred_test_lr.min())
    max_val = max(y_test.max(), y_pred_test_lr.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Baseline Linear Regression (Test set) - {target_col}")
    plt.show()

    ############################################################################
    # (B) BASELINE CROSS-VAL (Repeated K-Fold)
    ############################################################################
    baseline_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    cv_scores = cross_val_score(
        baseline_pipe, X_train, y_train, 
        scoring='r2', cv=rkf, n_jobs=-1
    )
    print("\n=== Baseline Linear Repeated K-Fold ===")
    print(f"Mean CV R^2: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")

    ############################################################################
    # (C) RIDGE: GridSearchCV with Repeated K-Fold
    ############################################################################
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    ridge_param_grid = {'ridge__alpha': alpha_values}
    ridge_grid = GridSearchCV(
        estimator=ridge_pipe,
        param_grid=ridge_param_grid,
        scoring='r2',
        cv=rkf,
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)

    print("\n=== Ridge: Repeated K-Fold GridSearch ===")
    print("Best alpha:", ridge_grid.best_params_)
    print(f"Best CV mean R^2: {ridge_grid.best_score_:.3f}")

    best_ridge = ridge_grid.best_estimator_
    y_pred_test_ridge = best_ridge.predict(X_test)

    test_r2_ridge  = r2_score(y_test, y_pred_test_ridge)
    test_mse_ridge = mean_squared_error(y_test, y_pred_test_ridge)
    print(f"Ridge Test R^2: {test_r2_ridge:.3f}, Test MSE: {test_mse_ridge:.4f}")

    # Plot predictions vs actual (TEST set) for Ridge
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test_ridge, edgecolor='k', alpha=0.7)
    min_val = min(y_test.min(), y_pred_test_ridge.min())
    max_val = max(y_test.max(), y_pred_test_ridge.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Ridge (Test set) - {target_col}")
    plt.show()

    ############################################################################
    # (D) LASSO: GridSearchCV with Repeated K-Fold
    ############################################################################
    lasso_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=10000))
    ])
    lasso_param_grid = {'lasso__alpha': alpha_values}
    lasso_grid = GridSearchCV(
        estimator=lasso_pipe,
        param_grid=lasso_param_grid,
        scoring='r2',
        cv=rkf,
        n_jobs=-1
    )
    lasso_grid.fit(X_train, y_train)

    print("\n=== Lasso: Repeated K-Fold GridSearch ===")
    print("Best alpha:", lasso_grid.best_params_)
    print(f"Best CV mean R^2: {lasso_grid.best_score_:.3f}")

    best_lasso = lasso_grid.best_estimator_
    y_pred_test_lasso = best_lasso.predict(X_test)

    test_r2_lasso  = r2_score(y_test, y_pred_test_lasso)
    test_mse_lasso = mean_squared_error(y_test, y_pred_test_lasso)
    print(f"Lasso Test R^2: {test_r2_lasso:.3f}, Test MSE: {test_mse_lasso:.4f}")

    # Plot predictions vs actual (TEST set) for Lasso
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test_lasso, edgecolor='k', alpha=0.7)
    min_val = min(y_test.min(), y_pred_test_lasso.min())
    max_val = max(y_test.max(), y_pred_test_lasso.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Lasso (Test set) - {target_col}")
    plt.show()

    ############################################################################
    # (E) PERMUTATION IMPORTANCE (Test Set): Store for subplot grouping later
    ############################################################################
    # Lasso
    print("\n=== Permutation Importance: Best Lasso (Test) ===")
    perm_lasso = permutation_importance(
        best_lasso, X_test, y_test,
        scoring='r2',
        n_repeats=10,
        random_state=42
    )
    importances_lasso = perm_lasso.importances_mean
    indices_lasso = np.argsort(importances_lasso)[::-1]

    print("Feature importances (Test set):")
    for i in indices_lasso:
        print(f"{features[i]:<25}  {importances_lasso[i]:.4f}")

    # Ridge
    print("\n=== Permutation Importance: Best Ridge (Test) ===")
    perm_ridge = permutation_importance(
        best_ridge, X_test, y_test,
        scoring='r2',
        n_repeats=10,
        random_state=42
    )
    importances_ridge = perm_ridge.importances_mean
    indices_ridge = np.argsort(importances_ridge)[::-1]

    print("Feature importances (Test set):")
    for i in indices_ridge:
        print(f"{features[i]:<25}  {importances_ridge[i]:.4f}")

    # Instead of plotting bar charts *now*, we return the sorted results
    # so we can do 3-subplots-later approach.
    # We'll store (features, importances, sorted indices).
    # This way we can produce a single figure for Lasso across all 3 targets,
    # and a single figure for Ridge across all 3 targets.
    lasso_import_data = (features, importances_lasso, indices_lasso)
    ridge_import_data = (features, importances_ridge, indices_ridge)

    # Return actual y_test plus predictions, plus importances
    return {
        'y_test':            y_test,
        'pred_baseline':     y_pred_test_lr,
        'pred_ridge':        y_pred_test_ridge,
        'pred_lasso':        y_pred_test_lasso,
        'lasso_import_data': lasso_import_data,
        'ridge_import_data': ridge_import_data
    }

###############################################################################
# 4) LOOP OVER TARGETS (FFR, CFR, CFR_FFR) & STORE RESULTS
###############################################################################
for tgt in targets:
    out = run_regression_pipeline_for_target(df, tgt)
    if out is not None:
        model_results[tgt] = out

###############################################################################
# 5) COMBINE PERMUTATION IMPORTANCES INTO 2 FIGURES:
#    - 1 figure for Lasso (3 subplots, one per target)
#    - 1 figure for Ridge (3 subplots, one per target)
###############################################################################
fig_lasso, axs_lasso = plt.subplots(1, 3, figsize=(18, 6))
fig_lasso.suptitle("Lasso Permutation Importances (Test)", fontsize=16)

fig_ridge, axs_ridge = plt.subplots(1, 3, figsize=(18, 6))
fig_ridge.suptitle("Ridge Permutation Importances (Test)", fontsize=16)

# We'll consider the targets in this order
target_order = ['FFR', 'CFR', 'CFR_FFR']

for i, tgt in enumerate(target_order):
    if tgt not in model_results:
        continue

    # Unpack Lasso info
    features_l, importances_l, indices_l = model_results[tgt]['lasso_import_data']
    ax_l = axs_lasso[i]
    sorted_feats_l = [features_l[j] for j in indices_l]
    sorted_imps_l  = importances_l[indices_l]

    ax_l.barh(sorted_feats_l, sorted_imps_l)
    ax_l.set_title(f"Lasso - {tgt}")
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Importance (R^2 decrease)")

    # Unpack Ridge info
    features_r, importances_r, indices_r = model_results[tgt]['ridge_import_data']
    ax_r = axs_ridge[i]
    sorted_feats_r = [features_r[j] for j in indices_r]
    sorted_imps_r  = importances_r[indices_r]

    ax_r.barh(sorted_feats_r, sorted_imps_r)
    ax_r.set_title(f"Ridge - {tgt}")
    ax_r.invert_yaxis()
    ax_r.set_xlabel("Importance (R^2 decrease)")

plt.tight_layout()
plt.show()
