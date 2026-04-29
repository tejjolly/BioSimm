#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: Polynomial linear regression (predicting CFR/FFR).
 - Uses all numeric columns from 'summary2.csv' except certain excluded columns.
 - Converts Location (LAD, LCX) into numeric {0,1}.
 - Splits data into train/test, does 5-fold cross-val, prints R^2,
   and plots predicted vs actual plus permutation importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

###############################################################################
# 1) LOAD & PREPARE DATA
###############################################################################
df = pd.read_csv('./summary2.csv')

# Keep only hyperemic runs
df = df[df['Condition'] == 'Hyperemic']

# Convert Location to numeric (0 for LAD, 1 for LCX)
# (If you prefer -1 and 1, you can map {'LAD':-1, 'LCX':1} instead.)
df['Location_num'] = df['Location'].map({'LAD': 0, 'LCX': 1})

# The target will be CFR/FFR
target_col = 'CFR/FFR'

# We want to use "all" remaining columns EXCEPT these:
excluded_cols = [
    'Condition',          # we filtered on this, so we don't use it as input
    'Geometry Number',
    'Rtotal_cor Value',
    'discord',
    'source',
    target_col            # must exclude target from features
]

#  We must also exclude columns that are non-numeric (besides 'Location', which we've replaced with Location_num).
#  So let's gather numeric columns only:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Now define the final feature set: numeric cols minus any excluded ones
feature_cols = [c for c in numeric_cols if c not in excluded_cols]

print("Feature columns:", feature_cols)

###############################################################################
# 2) BUILD THE DATASETS
###############################################################################
# X = all chosen features
# y = the single target (CFR/FFR)
df_model = df.dropna(subset=feature_cols + [target_col])  # remove rows with missing feature or target
X = df_model[feature_cols]
y = df_model[target_col]

print(f"Data shape: X={X.shape}, y={y.shape}")

###############################################################################
# 3) TRAIN/TEST SPLIT
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

###############################################################################
# 4) 5-FOLD CROSS-VALIDATION FOR POLYNOMIAL REGRESSION
###############################################################################
# We'll build a pipeline:
#   1) Scale X
#   2) Polynomial features (degree=2 as example)
#   3) Linear regression
pipe_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),  # <---- Polynomial step
    ('linear', LinearRegression())
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipe_poly,
    X_train,
    y_train,
    scoring='r2',
    cv=kf
)
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)

print("\n=== Polynomial Linear Regression (Degree=2) ===")
print(f"5-fold CV R^2: mean={cv_mean:.3f}, std={cv_std:.3f}")

###############################################################################
# 5) FIT FINAL MODEL & EVALUATE ON TEST SET
###############################################################################
pipe_poly.fit(X_train, y_train)
y_test_pred = pipe_poly.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
print(f"\nFinal Test R^2 on CFR/FFR = {test_r2:.3f}")

###############################################################################
# 6) PLOT: Predicted vs Actual (CFR/FFR)
###############################################################################
plt.figure()
plt.scatter(y_test, y_test_pred, edgecolors='k', alpha=0.8)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel("Actual CFR/FFR")
plt.ylabel("Predicted CFR/FFR")
plt.title("Polynomial Regression: CFR/FFR (Test)")
plt.show()

###############################################################################
# 7) PERMUTATION IMPORTANCE
###############################################################################
# For a single output, scoring='r2' is straightforward
perm = permutation_importance(
    pipe_poly['linear'],   # or pipe_poly named steps: pipe_poly.named_steps['linear']
    pipe_poly['poly'].transform(pipe_poly['scaler'].transform(X_test)),
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)

# Sort features by mean importance
importances = perm.importances_mean
indices = np.argsort(importances)[::-1]
sorted_feats = [feature_cols[i] for i in indices]
sorted_imps = importances[indices]

plt.figure()
plt.barh(sorted_feats, sorted_imps)
plt.gca().invert_yaxis()
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importance (Polynomial Regression: CFR/FFR)")
plt.tight_layout()
plt.show()
