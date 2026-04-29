#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi‑output Linear Regression (predicting [FFR, CFR])

• Set test_size = 0.0 to train on 100 % of the data *and* plot on all data.
• Otherwise the script behaves like the original (hold‑out, test‑set plots).

Outputs
  1) Predicted‑vs‑Actual FFR scatter
  2) Predicted‑vs‑Actual CFR scatter
  3) Permutation‑importance bar chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ───────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN DATA
# ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/data.csv")
df = df[df["Condition"] == "Hyperemic"].copy()
df.rename(columns={"P_d/P_a": "FFR"}, inplace=True)

feature_cols = ["HMR", "BMR/HMR", "P_Loss_Coeff"]
output_cols  = ["FFR", "CFR"]             # order: [FFR (col‑0), CFR (col‑1)]

df_model = df.dropna(subset=feature_cols + output_cols)
X, Y = df_model[feature_cols], df_model[output_cols]
print("Data shape:", X.shape, "| Outputs:", Y.shape)

# ───────────────────────────────────────────────────────────────────────
# 2) TRAIN / EVAL SPLIT   (0 ⇒ use all data for both)
# ───────────────────────────────────────────────────────────────────────
test_size = 0.0                 # <‑‑ change here
if test_size and test_size > 0:
    X_train, X_eval, Y_train, Y_eval = train_test_split(
        X, Y, test_size=test_size, random_state=41
    )
    label_eval = "test set"
else:
    X_train, Y_train = X, Y
    X_eval,  Y_eval  = X, Y           # evaluate/plot on all data
    label_eval = "all data"

print(f"Train size: {X_train.shape},  Eval size: {X_eval.shape}")

# ───────────────────────────────────────────────────────────────────────
# 3) CROSS‑VALIDATION ON TRAIN SPLIT
# ───────────────────────────────────────────────────────────────────────
kf   = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("linreg", LinearRegression())])

cv_ffr = cross_val_score(pipe, X_train, Y_train["FFR"], scoring="r2", cv=kf)
cv_cfr = cross_val_score(pipe, X_train, Y_train["CFR"], scoring="r2", cv=kf)
cv_all = cross_val_score(pipe, X, Y, scoring="r2", cv=kf)

print("\n5‑fold CV R²")
print(f"FFR : {cv_ffr.mean():.3f} ± {cv_ffr.std():.3f}")
print(f"CFR : {cv_cfr.mean():.3f} ± {cv_cfr.std():.3f}")
print(f"5‑fold CV R²  (all data, average over both targets)")
print(f"R²  : {cv_all.mean():.3f} ± {cv_all.std():.3f}")

# ───────────────────────────────────────────────────────────────────────
# 4) FIT FINAL MODEL, EVALUATE ON X_eval
# ───────────────────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_eval_s  = scaler.transform(X_eval)

linreg = LinearRegression().fit(X_train_s, Y_train)

Y_pred   = linreg.predict(X_eval_s)
r2_avg   = r2_score(Y_eval, Y_pred)
r2_each  = r2_score(Y_eval, Y_pred, multioutput="raw_values")

print(f"\n{label_eval.capitalize()} R² (avg) : {r2_avg:.3f}")
print(f"Per‑target R²        [FFR, CFR] : {r2_each}")

# ───────────────────────────────────────────────────────────────────────
# 5) PLOT 1 – FFR scatter
# ───────────────────────────────────────────────────────────────────────
plt.figure()
plt.scatter(Y_eval["FFR"], Y_pred[:, 0], facecolor="#5E9096",
            edgecolor="k", alpha=.8)
lo, hi = min(Y_eval["FFR"].min(), Y_pred[:, 0].min()), \
         max(Y_eval["FFR"].max(), Y_pred[:, 0].max())
plt.plot([lo, hi], [lo, hi], "k--")
plt.xlabel("Actual FFR"); plt.ylabel("Predicted FFR")
# plt.title(f"FFR – Linear Regression ({label_eval})")
plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────────────
# 6) PLOT 2 – CFR scatter
# ───────────────────────────────────────────────────────────────────────
plt.figure()
plt.scatter(Y_eval["CFR"], Y_pred[:, 1], facecolor="#5E9096",
            edgecolor="k", alpha=.8)
lo, hi = min(Y_eval["CFR"].min(), Y_pred[:, 1].min()), \
         max(Y_eval["CFR"].max(), Y_pred[:, 1].max())
plt.plot([lo, hi], [lo, hi], "k--")
plt.xlabel("Actual CFR"); plt.ylabel("Predicted CFR")
# plt.title(f"CFR – Linear Regression ({label_eval})")
plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────────────
# 7) PLOT 3 – Permutation Importance
#     (computed on X_eval / Y_eval, which is either the test set
#      or the full dataset if test_size = 0)
# ───────────────────────────────────────────────────────────────────────
perm = permutation_importance(
    linreg, X_eval_s, Y_eval, scoring="r2",
    n_repeats=10, random_state=42
)
imp_mean = perm.importances_mean
order    = np.argsort(imp_mean)[::-1]

plt.figure()
plt.barh(np.array(feature_cols)[order], imp_mean[order])
plt.gca().invert_yaxis()
plt.xlabel("Importance (Delta R2)")
plt.title(f"Permutation Importance")
plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────────────
# 8) COEFFICIENT SUMMARY
# ───────────────────────────────────────────────────────────────────────
coefs = pd.DataFrame(linreg.coef_, columns=feature_cols, index=["FFR", "CFR"])
print("\nCoefficient matrix:")
print(coefs)

print("\nIntercepts:")
print(pd.Series(linreg.intercept_, index=["FFR", "CFR"]))