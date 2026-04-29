#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:51:14 2025

@author: tejjolly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error

# ===== 1) LOAD DATA & SELECT VARIABLES =====

# 1A) Read your summary CSV
df = pd.read_csv('summary.csv')

df = df[df['Condition'] == 'Hyperemic'] # Dropping non-hyperemic runs

# If your real "FFR" is in a different column, adjust accordingly.
# Example: rename P_d/P_a -> 'FFR'
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

# Choose predictors
feature_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'HMR',
    'HSR',
    'WSS',
    'Rtotal_cor Value',
    'FFR',
    'BMR/HMR'
]

# Drop any rows with missing data in relevant columns
df_model = df.dropna(subset=feature_cols + ['CFR'])

X = df_model[feature_cols]
y = df_model['CFR']

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

# (Optional) Scale
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

# 3A) Plot predicted vs. actual for baseline model
plt.figure(figsize=(10, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         'r--')  # 1:1 line
plt.title("Baseline LR: Training Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

# Validation set
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         'r--')  # 1:1 line
plt.title("Baseline LR: Validation Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

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

# 4A) Plot predicted vs. actual for polynomial model
plt.figure(figsize=(10, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_poly, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         'r--')
plt.title("CFR, Poly LR (deg=2): Training Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

# Validation set
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val_poly, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         'r--')
plt.title("Poly LR (deg=2): Validation Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

plt.tight_layout()
plt.show()


# ===== 5) STEPWISE‐LIKE FEATURE SELECTION =====
# Using SequentialFeatureSelector with forward selection
lr_for_stepwise = LinearRegression()
from sklearn.feature_selection import SequentialFeatureSelector

sfs_forward = SequentialFeatureSelector(
    lr_for_stepwise,
    n_features_to_select="auto",  # or an integer, e.g., 10
    direction="forward",
    scoring="neg_mean_squared_error",  # or 'r2'
    cv=5,  # 5-fold cross-validation for selection
    n_jobs=-1
)

sfs_forward.fit(X_train_poly, y_train)
selected_mask = sfs_forward.get_support()

# Identify which polynomial features were chosen
poly_feature_names = poly.get_feature_names_out(input_features=feature_cols)
chosen_features = poly_feature_names[selected_mask]
print("\n=== Forward Stepwise Selected Features (degree=2) ===")
print(chosen_features)

# Build & evaluate final stepwise model
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

# 5A) Plot predicted vs. actual for stepwise polynomial model
plt.figure(figsize=(10, 5))

# Training predictions
y_pred_train_stepwise = lr_stepwise.predict(X_train_poly_sel)

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         'r--')
plt.title("Stepwise Poly LR: Training Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

# Validation predictions
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         'r--')
plt.title("Stepwise Poly LR: Validation Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")

plt.tight_layout()
plt.show()

# ===== 6) EVALUATE ON TEST SET =====
y_pred_test_stepwise = lr_stepwise.predict(X_test_poly_sel)
test_r2_stepwise = r2_score(y_test, y_pred_test_stepwise)
test_mse_stepwise = mean_squared_error(y_test, y_pred_test_stepwise)

print("\n=== FINAL TEST RESULTS (Stepwise + Polynomial) ===")
print(f"Test R^2: {test_r2_stepwise:.3f}, Test MSE: {test_mse_stepwise:.3f}")

# (Optional) Plot test predictions vs. actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test_stepwise, alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.title("Stepwise Poly LR: Test Set")
plt.xlabel("Actual CFR")
plt.ylabel("Predicted CFR")
plt.show()


# Print sorted coefficients
final_coefs = lr_stepwise.coef_
final_intercept = lr_stepwise.intercept_
# Sort features and coefficients by coefficient magnitude in descending order
sorted_coefs = sorted(zip(chosen_features, final_coefs), key=lambda x: abs(x[1]), reverse=True)
# Print intercept
print("Intercept:", final_intercept)
# Print sorted coefficients
for name, coef in sorted_coefs:
    print(name, ":", coef)


