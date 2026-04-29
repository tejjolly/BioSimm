#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four‑class classification (“quadrant”) version of the baseline script.

Author: tejjolly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREP DATA
# ────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('../data/data.csv')
df = df[df['Condition'] == 'Hyperemic']
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

# ── create the 4‑quadrant label ─────────────────────────────────────────────
def quadrant(row):
    cfr_high  = row['CFR'] >= 2
    ffr_high  = row['FFR'] >= 0.8
    if  cfr_high and  ffr_high:  return 0
    if  cfr_high and not ffr_high: return 1
    if not cfr_high and  ffr_high: return 2
    return 3                           # cfr_low & ffr_low

df['quad'] = df.apply(quadrant, axis=1)

our_data = False             # flip if you want the long feature list
if our_data:
    feature_cols = [
        'Stenosis Percentage','Length','Width','HMR','HSR','BMR/HMR','HSR',
        'P_Loss_Coeff','WSS_TE','WSS_LE','WSS_TE_Area','WSS_LE_Area',
        'WSS_Area_Bifur','WSS_Bif','WSS_LMB','WSS_min','WSS_TE_min',
        'WSS_LE_min','WSS_TE_Area_min','WSS_Area_Bifur_min',
        # 'v_distal'
    ]
else:
    feature_cols = ['HMR','BMR/HMR','P_Loss_Coeff']

df_model = df.dropna(subset=feature_cols + ['quad'])
X = df_model[feature_cols]
y = df_model['quad'].astype(int)       # 0‒3 labels

print("Data shape (features):", X.shape, "| Output shape:", y.shape)

# ────────────────────────────────────────────────────────────────────────────
# 2) TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ────────────────────────────────────────────────────────────────────────────
# 3) 5‑FOLD STRATIFIED CV  (macro‑F1 & accuracy)
# ────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=2, shuffle=True)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(
                    class_weight='balanced',
                    max_iter=2000))
])

from sklearn.metrics import recall_score, confusion_matrix

sensitivity_list = []
specificity_list = []
precision_list   = []

for train_index, val_index in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    pipe.fit(X_tr, y_tr)
    y_val_pred = pipe.predict(X_val)

    cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1, 2, 3])
    sensitivities = []
    specificities = []
    precisions = []

    for i in range(4):  # For each class
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = TP / (TP + FN) if (TP + FN) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0
        precision   = TP / (TP + FP) if (TP + FP) else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)

    sensitivity_list.append(sensitivities)
    specificity_list.append(specificities)
    precision_list.append(precisions)


    sensitivity_list.append(sensitivities)
    specificity_list.append(specificities)
    precision_list.append(precisions)


cv_f1  = cross_val_score(pipe, X_train, y_train,
                         scoring='f1_macro', cv=skf)
cv_acc = cross_val_score(pipe, X_train, y_train,
                         scoring='accuracy',  cv=skf)

print("\n=== Multinomial Logistic Regression (quadrant) ===")
print(f"5‑fold CV  macro‑F1 : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
print(f"5‑fold CV  accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 4) FINAL FIT & TEST METRICS
# ────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

logreg = LogisticRegression(
            class_weight='balanced',
            max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average='macro')

avg_sensitivity = np.mean(sensitivity_list, axis=0)
avg_specificity = np.mean(specificity_list, axis=0)
avg_precision = np.mean(precision_list, axis=0)


print("\nAverage Sensitivity (Recall) per class:")
for i, s in enumerate(avg_sensitivity):
    print(f"  Class {i}: {s:.3f}")

print("\nAverage Specificity per class:")
for i, s in enumerate(avg_specificity):
    print(f"  Class {i}: {s:.3f}")

print(f"\nTest accuracy : {test_acc:.3f}")
print(f"Test macro‑F1 : {test_f1:.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 5) CONFUSION MATRIX PLOT (4×4)
# ────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0','1','2','3'],
            yticklabels=['0','1','2','3'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Quadrant Labels)")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 6) PERMUTATION IMPORTANCE (macro‑F1 drop)
# ────────────────────────────────────────────────────────────────────────────
perm = permutation_importance(
    logreg, X_test_scaled, y_test,
    scoring='f1_macro', n_repeats=10)

importances = perm.importances_mean
idx = np.argsort(importances)[::-1]

plt.figure()
plt.barh(np.array(feature_cols)[idx], importances[idx])
plt.gca().invert_yaxis()
plt.xlabel("Mean decrease in macro‑F1")
plt.title("Permutation Importance (quadrant classifier)")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 4.5) HEATMAP OF SENSITIVITY AND SPECIFICITY
# ────────────────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    'Precision':   avg_precision,
    'Sensitivity':      avg_sensitivity,
    'Specificity': avg_specificity
}, index=[f"Class {i}" for i in range(4)]).T

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="Reds_r", cbar=True,
            linewidths=0.5, linecolor='gray')
plt.title("Average Per-Class Metrics (CV)")
plt.ylabel("Metric")
plt.xlabel("Class")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# PRINT COEFFICIENTS & INTERCEPTS (multinomial logits)
# ---------------------------------------------------
coef_df = pd.DataFrame(
    logreg.coef_,                       # shape = (4, n_features)
    columns=feature_cols,
    index=[f"Class {c}" for c in logreg.classes_]
).round(6)                             # adjust precision to taste

print("\nCoefficient matrix:")
print(coef_df)

intercepts = pd.Series(
    logreg.intercept_,
    index=[f"Class {c}" for c in logreg.classes_]
).round(6)

print("\nIntercepts:")
print(intercepts)

