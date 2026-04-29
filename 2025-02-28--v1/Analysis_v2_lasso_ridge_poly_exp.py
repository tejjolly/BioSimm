#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:34:51 2025

@author: tejjolly
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# SKLEARN imports
# ---------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ===== 1) LOAD DATA & SELECT VARIABLES =====

df = pd.read_csv('summary.csv')
df = df[df['Condition'] == 'Hyperemic']  # Focus on hyperemic runs only

# Rename P_d/P_a -> 'FFR'
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

feature_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'HMR',
    'HSR',
    'WSS',
    #'Rtotal_cor Value',
    #'CFR',
    'BMR/HMR'
]

df_model = df.dropna(subset=feature_cols + ['FFR'])
X = df_model[feature_cols]
y = df_model['FFR']

# ===== 2) SPLIT DATA: 60/20/20 =====
# First, separate 20% for test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# From the remaining 80%, take 25% for validation => 20% overall
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}, Test size: {X_test.shape}")

# (Optional) scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ===== 3) BASELINE MULTIPLE LINEAR REGRESSION =====
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_train = lr.predict(X_train_scaled)
y_pred_val   = lr.predict(X_val_scaled)

train_r2 = r2_score(y_train, y_pred_train)
val_r2   = r2_score(y_val, y_pred_val)

print("\n=== Baseline Linear Regression ===")
print(f"Train R^2: {train_r2:.3f}, Val R^2: {val_r2:.3f}")

# 3A) Plot predicted vs. actual (train + val)
plt.figure(figsize=(10, 5))

# Training
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title("FFR, Baseline LR: Training Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

# Validation
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title("FFR, Baseline LR: Validation Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

plt.tight_layout()
plt.show()

# ===== 4) POLYNOMIAL (degree=2) & INTERACTIONS =====
pol_degree = 2
poly = PolynomialFeatures(degree=pol_degree, include_bias=False)

X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly   = poly.transform(X_val_scaled)
X_test_poly  = poly.transform(X_test_scaled)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)

y_pred_train_poly = lr_poly.predict(X_train_poly)
y_pred_val_poly   = lr_poly.predict(X_val_poly)

train_r2_poly = r2_score(y_train, y_pred_train_poly)
val_r2_poly   = r2_score(y_val, y_pred_val_poly)

print(f"\n=== Polynomial (degree={pol_degree}) Linear Regression ===")
print(f"Train R^2: {train_r2_poly:.3f}, Val R^2: {val_r2_poly:.3f}")

# Plot predicted vs. actual
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_poly, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title(f"FFR, Poly LR (deg={pol_degree}): Training Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val_poly, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title(f"FFR, Poly LR (deg={pol_degree}): Validation Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

plt.tight_layout()
plt.show()

# ===== 5) STEPWISE‐LIKE FEATURE SELECTION =====
lr_for_stepwise = LinearRegression()
sfs = SequentialFeatureSelector(
    lr_for_stepwise,
    n_features_to_select="auto",  # or an int
    direction="forward",
    scoring="neg_mean_squared_error",  # could use 'r2'
    cv=5,
    n_jobs=-1
)
sfs.fit(X_train_poly, y_train)
selected_mask = sfs.get_support()

poly_feature_names = poly.get_feature_names_out(input_features=feature_cols)
chosen_features = poly_feature_names[selected_mask]
print("\n=== Forward Stepwise Selected Features (degree=2) ===")
print(chosen_features)

# Build final stepwise model
X_train_poly_sel = X_train_poly[:, selected_mask]
X_val_poly_sel   = X_val_poly[:, selected_mask]
X_test_poly_sel  = X_test_poly[:, selected_mask]

lr_stepwise = LinearRegression()
lr_stepwise.fit(X_train_poly_sel, y_train)

y_pred_val_stepwise = lr_stepwise.predict(X_val_poly_sel)
val_r2_stepwise = r2_score(y_val, y_pred_val_stepwise)
val_mse_stepwise = mean_squared_error(y_val, y_pred_val_stepwise)

print("\n=== Stepwise Feature Selection Model (Val) ===")
print(f"Val R^2: {val_r2_stepwise:.3f}, Val MSE: {val_mse_stepwise:.3f}")

# 5A) Plot stepwise results
plt.figure(figsize=(10, 5))

# Training predictions
y_pred_train_stepwise = lr_stepwise.predict(X_train_poly_sel)
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title("FFR, Stepwise Poly LR: Training Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

# Validation predictions
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title("FFR, Stepwise Poly LR: Validation Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")

plt.tight_layout()
plt.show()

# ===== 6) EVALUATE ON TEST SET =====
y_pred_test_stepwise = lr_stepwise.predict(X_test_poly_sel)
test_r2_stepwise = r2_score(y_test, y_pred_test_stepwise)
test_mse_stepwise = mean_squared_error(y_test, y_pred_test_stepwise)

print("\n=== FINAL TEST RESULTS (Stepwise + Polynomial) ===")
print(f"Test R^2: {test_r2_stepwise:.3f}, Test MSE: {test_mse_stepwise:.3f}")

# Optional: Plot test predictions vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("FFR, Stepwise Poly LR: Test Set")
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")
plt.show()

# Print sorted coefficients for the stepwise model
final_coefs = lr_stepwise.coef_
final_intercept = lr_stepwise.intercept_
print("\nIntercept:", final_intercept)
sorted_coefs = sorted(zip(chosen_features, final_coefs), key=lambda x: abs(x[1]), reverse=True)
for name, coef in sorted_coefs:
    print(name, ":", coef)


# =============== 7) PERMUTATION IMPORTANCE ON STEPWISE MODEL ===============
print("\n=== Permutation Importance for Stepwise Model ===")
perm = permutation_importance(
    lr_stepwise,
    X_val_poly_sel,
    y_val,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
importances = perm.importances_mean  # average importance
indices = np.argsort(importances)[::-1]  # descending

print("Permutation Importances (Validation Set):")
for i in indices:
    print(f"{chosen_features[i]:<40}  {importances[i]:.4f}")

plt.figure(figsize=(7, 6))
plt.barh([chosen_features[i] for i in indices], importances[indices])
plt.title("Permutation Importances (Val) - Stepwise Model")
plt.xlabel("Importance (R^2 decrease)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# =============== 8) RIDGE & LASSO (GRID SEARCH) ===============
# We'll build a pipeline that does scaling + polynomial + ridge (or lasso).
# Then do cross-validation on X_train,y_train only, and check on val/test.

alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

# -- Ridge Pipeline --
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),  # re-scale inside pipeline
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge())
])

ridge_param_grid = {
    'ridge__alpha': alpha_values
}

ridge_grid = GridSearchCV(
    ridge_pipe,
    ridge_param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1
)
ridge_grid.fit(X_train, y_train)

print("\n=== Ridge Regression Grid Search ===")
print("Best Ridge params:", ridge_grid.best_params_)
best_ridge = ridge_grid.best_estimator_

# Evaluate best ridge on train/val/test
y_train_pred_ridge = best_ridge.predict(X_train)
y_val_pred_ridge   = best_ridge.predict(X_val)
y_test_pred_ridge  = best_ridge.predict(X_test)

print("Ridge Train R^2:", r2_score(y_train, y_train_pred_ridge))
print("Ridge Val   R^2:", r2_score(y_val,   y_val_pred_ridge))
print("Ridge Test  R^2:", r2_score(y_test,  y_test_pred_ridge))

# Permutation importance for best_ridge on validation
perm_ridge = permutation_importance(
    best_ridge,
    X_val,
    y_val,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
importances_ridge = perm_ridge.importances_mean
idx_ridge = np.argsort(importances_ridge)[::-1]
# Get feature names from pipeline
feat_names_ridge = best_ridge.named_steps['poly'].get_feature_names_out(feature_cols)

print("\nPermutation Importance (Val) - Best Ridge:")
for i in idx_ridge:
    print(f"{feat_names_ridge[i]:<40}  {importances_ridge[i]:.4f}")

plt.figure(figsize=(7, 6))
plt.barh([feat_names_ridge[i] for i in idx_ridge], importances_ridge[idx_ridge])
plt.title("Permutation Importances (Val) - Best Ridge")
plt.xlabel("Importance (R^2 decrease)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -- Lasso Pipeline --
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lasso', Lasso(max_iter=10000))  # Increase max_iter if needed
])

lasso_param_grid = {
    'lasso__alpha': alpha_values
}

lasso_grid = GridSearchCV(
    lasso_pipe,
    lasso_param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1
)
lasso_grid.fit(X_train, y_train)

print("\n=== Lasso Regression Grid Search ===")
print("Best Lasso params:", lasso_grid.best_params_)
best_lasso = lasso_grid.best_estimator_

# Evaluate best lasso
y_train_pred_lasso = best_lasso.predict(X_train)
y_val_pred_lasso   = best_lasso.predict(X_val)
y_test_pred_lasso  = best_lasso.predict(X_test)

print("Lasso Train R^2:", r2_score(y_train, y_train_pred_lasso))
print("Lasso Val   R^2:", r2_score(y_val,   y_val_pred_lasso))
print("Lasso Test  R^2:", r2_score(y_test,  y_test_pred_lasso))

# Permutation importance for best_lasso
perm_lasso = permutation_importance(
    best_lasso,
    X_val,
    y_val,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
importances_lasso = perm_lasso.importances_mean
idx_lasso = np.argsort(importances_lasso)[::-1]
feat_names_lasso = best_lasso.named_steps['poly'].get_feature_names_out(feature_cols)

print("\nPermutation Importance (Val) - Best Lasso:")
for i in idx_lasso:
    print(f"{feat_names_lasso[i]:<40}  {importances_lasso[i]:.4f}")

plt.figure(figsize=(7, 6))
plt.barh([feat_names_lasso[i] for i in idx_lasso], importances_lasso[idx_lasso])
plt.title("Permutation Importances (Val) - Best Lasso")
plt.xlabel("Importance (R^2 decrease)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
