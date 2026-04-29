
"""
Example script: 
 - Drop polynomial expansions
 - Do a standard multiple linear regression
 - Also compare Ridge/Lasso with repeated k-fold CV for alpha tuning
 - Finally evaluate chosen models on a hold-out test set
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

# =============== 1) LOAD & CLEAN DATA ===============
df = pd.read_csv('summary.csv')

# Focus on hyperemic runs only
df = df[df['Condition'] == 'Hyperemic']

# Rename P_d/P_a -> 'FFR' if needed
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

# Choose columns (dropping Rtotal_cor Value, CFR, etc.)
feature_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'HMR',
    'HSR',
    'WSS',
    'BMR/HMR',
    'CFR',
]
df_model = df.dropna(subset=feature_cols + ['FFR'])

X = df_model[feature_cols]
y = df_model['FFR']

print("Data shape:", X.shape)

# =============== 2) TRAIN/TEST SPLIT (20% TEST) ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# =============== 3) BASELINE LINEAR REGRESSION (No poly, no stepwise) ===============
# We'll do a quick fit on the training set and evaluate on test set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_train_pred = lr.predict(X_train_scaled)
y_test_pred  = lr.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2  = r2_score(y_test,  y_test_pred)

print("\n=== Baseline Linear Regression ===")
print(f"Train R^2: {train_r2:.3f}, Test R^2: {test_r2:.3f}")

# Optional: Plot predictions vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual FFR")
plt.ylabel("Predicted FFR")
plt.title("Baseline Linear Regression (Test set)")
plt.show()

# =============== 4) REPEATED K-FOLD CROSS-VALIDATION ===============
# We'll do repeated k-fold on (X_train, y_train).
# For example, 5 folds repeated 3 times. Adjust as you like.
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Quick demonstration: cross-val on baseline linear (no hyperparameter)
# We'll do a pipeline with scaling + linear
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

cv_scores = cross_val_score(baseline_pipe, X_train, y_train, 
                            scoring='r2', cv=rkf, n_jobs=-1)
print("\n=== Baseline Linear Repeated K-Fold ===")
print(f"Mean CV R^2: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")

# =============== 5) RIDGE & LASSO (No polynomial expansions) ===============
# We'll do a pipeline: scale -> ridge(or lasso).
# Then do GridSearchCV with repeated k-fold for alpha. 
# Finally, we'll evaluate the best model on the hold-out test set.

alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# 5A) Ridge
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
ridge_param_grid = {
    'ridge__alpha': alpha_values
}
ridge_grid = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=ridge_param_grid,
    scoring='r2',
    cv=rkf,           # repeated k-fold
    n_jobs=-1
)
ridge_grid.fit(X_train, y_train)

print("\n=== Ridge: Repeated K-Fold GridSearch ===")
print("Best alpha:", ridge_grid.best_params_)
print(f"Best CV mean R^2: {ridge_grid.best_score_:.3f}")

best_ridge = ridge_grid.best_estimator_

# Evaluate on test
y_test_pred_ridge = best_ridge.predict(X_test)
test_r2_ridge  = r2_score(y_test, y_test_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
print(f"Ridge Test R^2: {test_r2_ridge:.3f}, Test MSE: {test_mse_ridge:.4f}")

# 5B) Lasso
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
lasso_param_grid = {
    'lasso__alpha': alpha_values
}
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

# Evaluate on test
y_test_pred_lasso = best_lasso.predict(X_test)
test_r2_lasso  = r2_score(y_test, y_test_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
print(f"Lasso Test R^2: {test_r2_lasso:.3f}, Test MSE: {test_mse_lasso:.4f}")

# =============== 6) OPTIONAL: Permutation Importance ===============
# Let's do it for whichever model you consider "final." 
# Suppose we pick best_lasso as final:

print("\n=== Permutation Importance: Best Lasso (Test set) ===")
# We'll need the pipeline to transform X_test as well 
perm = permutation_importance(
    best_lasso,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

print("Feature importances (Test set):")
for i in indices:
    print(f"{feature_cols[i]:<25}  {importances[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.barh([feature_cols[i] for i in indices], importances[indices])
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importances - Lasso (Test)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


print("\n=== Permutation Importance: Best Ridge (Test set) ===")
# We'll need the pipeline to transform X_test as well 
perm = permutation_importance(
    best_ridge,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

print("Feature importances (Test set):")
for i in indices:
    print(f"{feature_cols[i]:<25}  {importances[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.barh([feature_cols[i] for i in indices], importances[indices])
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importances - Ridge (Test)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

