
"""
Extended script that includes:
 - Residual analysis & outlier detection
 - Viewing coefficients
 - Partial dependence plots
 - SHAP interpretability (optional)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# For Cook's distance we can use statsmodels
import statsmodels.api as sm

# SHAP interpretability (install shap if you haven't: pip install shap)
# !pip install shap
import shap

# =============== 1) LOAD & CLEAN DATA ===============
df = pd.read_csv('summary.csv')

# Focus on hyperemic runs only
df = df[df['Condition'] == 'Hyperemic']

# Rename P_d/P_a -> 'FFR'
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

# Choose columns (dropping any others as needed)
feature_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    # 'Average Flow',   # Uncomment if needed
    'HMR',
    'HSR',
    'WSS',
    'BMR/HMR',
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

# =============== 3) BASELINE LINEAR REGRESSION ===============
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

# ----- 3A) Residual Analysis for Baseline LR -----

# Residuals
residuals_train = y_train_pred - y_train
residuals_test  = y_test_pred - y_test

# 1) Residual vs. Predicted
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(y_train_pred, residuals_train, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Baseline LR: Residuals vs. Predicted (Train)")
plt.xlabel("Predicted FFR")
plt.ylabel("Residual")

plt.subplot(1,2,2)
plt.scatter(y_test_pred, residuals_test, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Baseline LR: Residuals vs. Predicted (Test)")
plt.xlabel("Predicted FFR")
plt.ylabel("Residual")

plt.tight_layout()
plt.show()

# 2) Histogram of Residuals (Train & Test)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(residuals_train, bins=10, edgecolor='k')
plt.title("Train Residuals Distribution")

plt.subplot(1,2,2)
plt.hist(residuals_test, bins=10, edgecolor='k')
plt.title("Test Residuals Distribution")

plt.tight_layout()
plt.show()

# ----- 3B) Cook's Distance / Outlier Detection -----
# We'll do a quick fit in statsmodels so we can compute Cook's Distance
# and see if any points are particularly influential on the training set.
# NOTE: Statsmodels wants an intercept column explicitly.

X_train_sm = sm.add_constant(X_train_scaled)
model_sm = sm.OLS(y_train, X_train_sm).fit()
influence = model_sm.get_influence()
cooks_d, pvals = influence.cooks_distance

# Plot Cook's distance
plt.figure(figsize=(6,4))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Cook's Distance (Training Data)")
plt.xlabel("Observation index")
plt.ylabel("Cook's distance")
plt.show()

# A common rule of thumb is to check if Cook's distance > 4/n.
n = len(X_train)
threshold = 4/n
outliers = np.where(cooks_d > threshold)[0]
print(f"\nPotential outliers (Cook's dist > {threshold:.3f}):", outliers)


# =============== 4) REPEATED K-FOLD CROSS-VALIDATION (Baseline) ===============
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])
cv_scores = cross_val_score(baseline_pipe, X_train, y_train, 
                            scoring='r2', cv=rkf, n_jobs=-1)
print("\n=== Baseline Linear Repeated K-Fold ===")
print(f"Mean CV R^2: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")


# =============== 5) RIDGE & LASSO ===============
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# ---- 5A) Ridge
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
ridge_grid = GridSearchCV(
    estimator=ridge_pipe,
    param_grid={'ridge__alpha': alpha_values},
    scoring='r2',
    cv=rkf,
    n_jobs=-1
)
ridge_grid.fit(X_train, y_train)

print("\n=== Ridge: Repeated K-Fold GridSearch ===")
print("Best alpha:", ridge_grid.best_params_)
print(f"Best CV mean R^2: {ridge_grid.best_score_:.3f}")

best_ridge = ridge_grid.best_estimator_
y_test_pred_ridge = best_ridge.predict(X_test)
test_r2_ridge  = r2_score(y_test, y_test_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
print(f"Ridge Test R^2: {test_r2_ridge:.3f}, Test MSE: {test_mse_ridge:.4f}")

# ---- 5B) Lasso
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
lasso_grid = GridSearchCV(
    estimator=lasso_pipe,
    param_grid={'lasso__alpha': alpha_values},
    scoring='r2',
    cv=rkf,
    n_jobs=-1
)
lasso_grid.fit(X_train, y_train)

print("\n=== Lasso: Repeated K-Fold GridSearch ===")
print("Best alpha:", lasso_grid.best_params_)
print(f"Best CV mean R^2: {lasso_grid.best_score_:.3f}")

best_lasso = lasso_grid.best_estimator_
y_test_pred_lasso = best_lasso.predict(X_test)
test_r2_lasso  = r2_score(y_test, y_test_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
print(f"Lasso Test R^2: {test_r2_lasso:.3f}, Test MSE: {test_mse_lasso:.4f}")


# =============== 6) COEFFICIENTS ===============
# For the baseline LR (with scaling), let's retrieve coefficients in original or scaled space.
# The simplest: pipeline approach for LR. But we did manual scaling, so let's show baseline LR's scaled coefs:

print("\n=== Baseline Linear Regression Coefficients ===")
# lr.coef_ are in scaled feature space. 
coefs_lr = lr.coef_
intercept_lr = lr.intercept_

# Print each feature + coef
for feat_name, coef_val in zip(feature_cols, coefs_lr):
    print(f"{feat_name}: {coef_val:.4f}")
print("Intercept (in target's units):", intercept_lr)

# You can do the same for best_ridge and best_lasso:
print("\n=== Best Ridge Coefficients ===")
coefs_ridge = best_ridge.named_steps['ridge'].coef_
# Because that pipeline includes StandardScaler, these coefs are also in scaled space
for feat_name, coef_val in zip(feature_cols, coefs_ridge):
    print(f"{feat_name}: {coef_val:.4f}")

print("\n=== Best Lasso Coefficients ===")
coefs_lasso = best_lasso.named_steps['lasso'].coef_
for feat_name, coef_val in zip(feature_cols, coefs_lasso):
    print(f"{feat_name}: {coef_val:.4f}")

# =============== 7) PERMUTATION IMPORTANCE ===============
print("\n=== Permutation Importance: Best Lasso (Test set) ===")
perm_lasso = permutation_importance(
    best_lasso,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
imps_lasso = perm_lasso.importances_mean
idx_lasso = np.argsort(imps_lasso)[::-1]
for i in idx_lasso:
    print(f"{feature_cols[i]:<25}  {imps_lasso[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.barh([feature_cols[i] for i in idx_lasso], imps_lasso[idx_lasso])
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importances - Lasso (Test)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n=== Permutation Importance: Best Ridge (Test set) ===")
perm_ridge = permutation_importance(
    best_ridge,
    X_test,
    y_test,
    scoring='r2',
    n_repeats=10,
    random_state=42
)
imps_ridge = perm_ridge.importances_mean
idx_ridge = np.argsort(imps_ridge)[::-1]
for i in idx_ridge:
    print(f"{feature_cols[i]:<25}  {imps_ridge[i]:.4f}")

plt.figure(figsize=(7, 5))
plt.barh([feature_cols[i] for i in idx_ridge], imps_ridge[idx_ridge])
plt.xlabel("Importance (R^2 decrease)")
plt.title("Permutation Importances - Ridge (Test)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# =============== 8) PARTIAL DEPENDENCE PLOTS ===============
# We'll do them for best_lasso as an example. 
# partial_dependence is simpler if we wrap best_lasso in a single pipeline with the scaler
# But we already have best_lasso as a pipeline, so we can do:

print("\n=== Partial Dependence Plots: Best Lasso ===")

fig, axs = plt.subplots(1, len(feature_cols), figsize=(4*len(feature_cols),4), sharey=True)

for i, feat_name in enumerate(feature_cols):
    # partial dependence: we need feature indices by their position in X
    # In alphabetical order? Actually let's just find the index:
    feature_idx = feature_cols.index(feat_name)

    disp = PartialDependenceDisplay.from_estimator(
        best_lasso,          # pipeline
        X_train,             # unscaled training data
        [feature_idx],
        ax=axs[i],
        feature_names=feature_cols,
        kind="average"
    )
    axs[i].set_xlabel(feat_name)
    axs[i].set_ylabel("FFR")
    axs[i].set_title(f"PDP: {feat_name}")
plt.tight_layout()
plt.show()


# =============== 9) SHAP EXPLANATIONS ===============
# For linear models, SHAP is basically similar to seeing your coefficients, but let's demonstrate:

# ...
# after you define best_lasso, do this:

print("\n=== SHAP for Best Lasso ===")
# 1) Extract final Lasso from the pipeline
final_lasso_model = best_lasso.named_steps['lasso']

# 2) Transform X_train with the pipeline's scaler
X_train_scaled_for_shap = best_lasso.named_steps['scaler'].transform(X_train)

# 3) Create SHAP explainer on the final Lasso model + scaled data
explainer = shap.Explainer(
    final_lasso_model, 
    X_train_scaled_for_shap,
    feature_names=feature_cols
)

# 4) Transform X_test
X_test_scaled_for_shap = best_lasso.named_steps['scaler'].transform(X_test)

# 5) Compute shap values
shap_values = explainer(X_test_scaled_for_shap)

# 6) Summary plot
shap.summary_plot(shap_values, X_test_scaled_for_shap, feature_names=feature_cols)


# Ensure full display of DataFrame without truncation
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width
pd.set_option('expand_frame_repr', False)  # Prevent wrapping

# 1) Print individual outliers with full feature display
for idx in outliers:
    original_idx = X_train.index[idx]  # Get original index from df_model
    print(f"Outlier in training set at position {idx}, original df index = {original_idx}")
    print(df_model.loc[original_idx].to_string())  # Use `.to_string()` to print fully
    print("---")

# 2) Print full DataFrame of outliers
outlier_df = df_model.loc[X_train.index[outliers]]
print("\nOutlier runs (via Cook's distance > threshold):")
print(outlier_df.to_string())  # Ensures full DataFrame is printed

