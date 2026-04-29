#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi‐Output Regression Example:
 - We now predict both FFR and CFR together as y = [FFR, CFR].
 - Compare baseline LR, then do Ridge/Lasso with repeated k-fold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# =============== 1) LOAD & CLEAN DATA ===============
df = pd.read_csv('summary.csv')

# Keep only hyperemic runs (if that is your usage)
df = df[df['Condition'] == 'Hyperemic']

# Rename P_d/P_a -> 'FFR' if needed
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

# We want to predict both FFR and CFR simultaneously => multi‐output
# So we need both in the dataset, plus features
feature_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    # 'Average Flow',
    'HMR',
    'HSR',
    'WSS',
    'BMR/HMR'
]
# Make sure we have 'CFR' in the DF for the second output
output_cols = ['FFR','CFR']

df_model = df.dropna(subset=feature_cols + output_cols)

X = df_model[feature_cols]
Y = df_model[output_cols]  # shape (n_samples, 2) => multi‐output

print("Data shape:", X.shape, "Outputs shape:", Y.shape)

# =============== 2) TRAIN/TEST SPLIT (20% TEST) ===============
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# =============== 3) BASELINE MULTI‐OUTPUT LINEAR REGRESSION ===============
# We'll scale features, but Y is used as is (no separate scaling by default).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)  # Y_train is (n,2)

# Predictions => shape (n_train,2) or (n_test,2)
Y_train_pred = lr.predict(X_train_scaled)
Y_test_pred  = lr.predict(X_test_scaled)

# (a) R² *averaged* over both outputs
train_r2_avg = r2_score(Y_train, Y_train_pred)   # default = 'uniform_average'
test_r2_avg  = r2_score(Y_test,  Y_test_pred)

# (b) R² *per output*
train_r2_per_target = r2_score(Y_train, Y_train_pred, multioutput='raw_values')
test_r2_per_target  = r2_score(Y_test,  Y_test_pred,  multioutput='raw_values')

print("\n=== Baseline Multi-Output Linear Regression ===")
print(f"Train R^2 (avg over FFR & CFR): {train_r2_avg:.3f}, per target = {train_r2_per_target}")
print(f"Test  R^2 (avg)               : {test_r2_avg:.3f}, per target = {test_r2_per_target}")

# Optional: Plot predictions vs actual for each target
#   e.g. target 0 -> FFR, target 1 -> CFR
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(Y_test.iloc[:,0], Y_test_pred[:,0], color = 'k', edgecolor='k', alpha=1)
plt.plot([Y_test.iloc[:,0].min(), Y_test.iloc[:,0].max()],
         [Y_test.iloc[:,0].min(), Y_test.iloc[:,0].max()], 'k--')
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")
plt.title("FFR (Test)")

plt.subplot(1,2,2)
plt.scatter(Y_test.iloc[:,1], Y_test_pred[:,1], color = 'k', edgecolor='k', alpha=1)
plt.plot([Y_test.iloc[:,1].min(), Y_test.iloc[:,1].max()],
         [Y_test.iloc[:,1].min(), Y_test.iloc[:,1].max()], 'k--')
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")
plt.title("CFR (Test)")

plt.tight_layout()
plt.show()

# =============== 4) REPEATED K-FOLD CROSS-VALIDATION ===============
# We'll do repeated k-fold on (X_train, Y_train).
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# By default, 'r2' scoring will average over both outputs. 
# If you want a custom approach, you can define a custom scoring. 
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

cv_scores = cross_val_score(
    baseline_pipe, X_train, Y_train,
    scoring='r2', cv=rkf, n_jobs=-1
)

print("\n=== Baseline Multi-Output Linear: Repeated K-Fold ===")
print(f"Mean CV R^2 (avg of outputs): {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")

# =============== 5) RIDGE & LASSO (Multi-Output) ===============
# We'll do a pipeline: scale -> ridge (or lasso),
# Then do GridSearchCV with repeated k-fold for alpha.

alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# 5A) Ridge
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
ridge_param_grid = {'ridge__alpha': alpha_values}

ridge_grid = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=ridge_param_grid,
    scoring='r2',    # average R² over outputs
    cv=None,
    n_jobs=-1
)
ridge_grid.fit(X_train, Y_train)

print("\n=== Ridge: Repeated K-Fold GridSearch (Multi-Output) ===")
print("Best alpha:", ridge_grid.best_params_)
print(f"Best CV mean R^2 (avg outputs): {ridge_grid.best_score_:.3f}")

best_ridge = ridge_grid.best_estimator_

# Evaluate on test
Y_test_pred_ridge = best_ridge.predict(X_test)
test_r2_ridge_avg  = r2_score(Y_test, Y_test_pred_ridge)  # average
test_r2_ridge_raw  = r2_score(Y_test, Y_test_pred_ridge, multioutput='raw_values')
print(f"Ridge Test R^2 (avg): {test_r2_ridge_avg:.3f}, per target={test_r2_ridge_raw}")

# 5B) Lasso
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
lasso_param_grid = {'lasso__alpha': alpha_values}

lasso_grid = GridSearchCV(
    estimator=lasso_pipe,
    param_grid=lasso_param_grid,
    scoring='r2',
    cv=None,
    n_jobs=-1
)
lasso_grid.fit(X_train, Y_train)

print("\n=== Lasso: Repeated K-Fold GridSearch (Multi-Output) ===")
print("Best alpha:", lasso_grid.best_params_)
print(f"Best CV mean R^2: {lasso_grid.best_score_:.3f}")

best_lasso = lasso_grid.best_estimator_

Y_test_pred_lasso = best_lasso.predict(X_test)
test_r2_lasso_avg = r2_score(Y_test, Y_test_pred_lasso)
test_r2_lasso_raw = r2_score(Y_test, Y_test_pred_lasso, multioutput='raw_values')
print(f"Lasso Test R^2 (avg): {test_r2_lasso_avg:.3f}, per target={test_r2_lasso_raw}")

# =============== 6) OPTIONAL: Permutation Importance ===============
# For multi-output, scikit-learn's permutation_importance uses the default
# 'r2' metric, i.e. average over outputs. We'll do it for best_lasso:

print("\n=== Permutation Importance: Best Lasso (Test set) ===")

perm = permutation_importance(
    best_lasso,
    X_test,
    Y_test,  # shape (n_test,2)
    scoring='r2',         # again, average R^2 over outputs
    n_repeats=10,
    random_state=42
)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

print("Feature importances (Test set) [avg R^2 decrease across 2 targets]:")
for i in indices:
    print(f"{feature_cols[i]:<25}  {importances[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.barh([feature_cols[i] for i in indices], importances[indices], color='k')
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importances - Lasso (Multi-Output)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
