#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 05:11:50 2025

@author: tejjolly

Description:
    - Polynomial Regression Example (predicting 'CFR/FFR').
    - Uses RepeatedKFold CV, feature selection, and an optional polynomial
      expansion of degree=2.
    - Produces:
        1) Target distribution plot
        2) Predicted vs Actual (Test)
        3) Permutation importance (Test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

###############################################################################
# 1) LOAD & CLEAN DATA
###############################################################################
df = pd.read_csv('./summary3.csv')
df = df[df['Condition'] == 'Hyperemic']
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

df['WSS_TE']      = df['WSS_TE'] / df['WSS_LMB']
df['WSS_LE']      = df['WSS_LE'] / df['WSS_LMB']
df['WSS_Bif']     = df['WSS_Bif'] / df['WSS_LMB']
df['WSS_TE_min']  = df['WSS_TE_min'] / df['WSS_LMB']
df['WSS_LE_min']  = df['WSS_LE_min'] / df['WSS_LMB']


feature_cols = [
    # 'Length',
    # 'Width',
    # 'HMR',
    # 'HSR',
    'BMR/HMR',
    # 'P_Loss_Coeff',
    # 'WSS_TE',
    # 'WSS_LE',
    # 'WSS_TE_Area',
    # 'WSS_LE_Area',
    # 'WSS_Area_Bifur',
    # 'WSS_Bif',
    # # 'WSS_LMB',
    # 'WSS_TE_min',
    # 'WSS_LE_min',
    # 'WSS_TE_Area_min',
    # 'WSS_Area_Bifur_min',
    # 'v_distal'
]
target_col = 'CFR/FFR'

df_model = df.dropna(subset=feature_cols + [target_col])
X = df_model[feature_cols]
y = df_model[target_col]

print("Data shape (features):", X.shape, "| Target shape:", y.shape)

###############################################################################
# 2) Plot target distribution
###############################################################################
plt.figure()
sns.histplot(y, kde=True)
plt.title("Target Distribution: CFR/FFR")
plt.xlabel("CFR/FFR")
plt.ylabel("Frequency")
plt.show()

###############################################################################
# 3) TRAIN/TEST SPLIT
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

###############################################################################
# 4) PIPELINE W/ POLYNOMIAL FEATURES + FEATURE SELECTION + RepeatedKFold CV
###############################################################################
# PolynomialFeatures(degree=2, include_bias=False) creates squared & cross-terms.
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_regression, k=19)),
    ('model', LinearRegression())
])

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

cv_scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=rkf)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print("\n=== Polynomial Regression (degree=2) ===")
print(f"Repeated 5-fold CV R^2: mean={cv_mean:.3f}, std={cv_std:.3f}")

###############################################################################
# 5) FINAL MODEL ON TRAINING SET
###############################################################################
# We'll fit the same pipeline on the entire training set.
pipe.fit(X_train, y_train)

# Predict on the test set
y_test_pred = pipe.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"\nFinal Test R^2 = {test_r2:.3f}")

###############################################################################
# 6) PLOT #1: Predicted vs Actual (Test Set)
###############################################################################
plt.figure()
plt.scatter(y_test, y_test_pred, edgecolors='k', alpha=0.8)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel(f"Actual {target_col}")
plt.ylabel(f"Predicted {target_col}")
plt.title(f"{target_col} (Test) - Polynomial Regression (deg=2)")
plt.show()

###############################################################################
# 7) PLOT #2: PERMUTATION IMPORTANCE (Test Set)
###############################################################################
# This will measure the drop in R^2 when each *transformed* feature is shuffled.
perm = permutation_importance(
    pipe,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=100,
    random_state=19
)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

# (A) get the polynomial feature names
poly_step = pipe.named_steps['poly']
poly_feature_names = poly_step.get_feature_names_out(feature_cols)

# (B) see which features were selected by SelectKBest
select_step = pipe.named_steps['select']
selected_indices = select_step.get_support(indices=True)

# Only keep the selected feature names and corresponding importances
selected_poly_names = [poly_feature_names[i] for i in selected_indices]
selected_importances = importances[:len(selected_poly_names)]  # match length

# Sort by importance
sorted_pairs = sorted(zip(selected_poly_names, selected_importances), key=lambda x: x[1], reverse=True)
sorted_feats, sorted_imps = zip(*sorted_pairs)

# Plot
plt.figure()
plt.barh(sorted_feats, sorted_imps)
plt.gca().invert_yaxis()
plt.xlabel("Importance (R^2 decrease)")
plt.title(f"Permutation Importance ({target_col}, poly deg=2)")
plt.tight_layout()
plt.show()
