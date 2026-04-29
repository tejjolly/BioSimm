#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 05:11:50 2025

@author: tejjolly

Description:
    - Enhanced Linear Regression Example (predicting 'CFR/FFR').
    - Includes RepeatedKFold CV with feature selection.
    - Optional: Ridge regression & PCA (commented).
    - Includes:
        1) Target distribution plot
        2) Predicted vs Actual
        3) Permutation importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # or Ridge
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.linear_model import Ridge
# from sklearn.decomposition import PCA

###############################################################################
# 1) LOAD & CLEAN DATA
###############################################################################
df = pd.read_csv('./summary3.csv')
df = df[df['Condition'] == 'Hyperemic']
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

feature_cols = [
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
# 4) PIPELINE W/ FEATURE SELECTION + RepeatedKFold CV
###############################################################################
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_regression, k=8)),
    # ('pca', PCA(n_components=0.95)),  # Optional PCA instead of feature selection
    ('model', LinearRegression())  # Change to Ridge(alpha=1.0) if needed
])

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

cv_scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=rkf)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print("\n=== Enhanced Linear Regression ===")
print(f"Repeated 5-fold CV R^2: mean={cv_mean:.3f}, std={cv_std:.3f}")

###############################################################################
# 5) FINAL MODEL ON TRAINING SET
###############################################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional feature selection for final model
selector = SelectKBest(score_func=f_regression, k=19)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)
# X_train_sel = X_train_scaled
# X_test_sel = X_test_scaled

model = LinearRegression()
model.fit(X_train_sel, y_train)

y_test_pred = model.predict(X_test_sel)
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
plt.title(f"{target_col} (Test) - Linear Regression")
plt.show()

###############################################################################
# 7) PLOT #2: PERMUTATION IMPORTANCE (Test Set)
###############################################################################
perm = permutation_importance(
    model,
    X_test_sel,
    y_test,
    scoring='r2',
    n_repeats=100,
    random_state=19
)

# Map indices back to selected feature names
selected_indices = selector.get_support(indices=True)
selected_feature_names = [feature_cols[i] for i in selected_indices]

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

sorted_feats = [selected_feature_names[i] for i in indices]
sorted_imps = importances[indices]

plt.figure()
plt.barh(sorted_feats, sorted_imps)
plt.gca().invert_yaxis()
plt.xlabel("Importance (R^2 decrease)")
plt.title(f"Permutation Importance ({target_col})")
plt.tight_layout()
plt.show()
