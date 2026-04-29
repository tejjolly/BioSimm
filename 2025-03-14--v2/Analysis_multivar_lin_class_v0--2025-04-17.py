#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary classification version of the original multi‑output regression.
Predicts `discord` (0 = concordant, 1 = discordant).

Author: tejjolly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                      # only for pretty confusion‑matrix

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression          ### CHANGED ###
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ────────────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN DATA
# ────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('../data/data.csv')
df = df[df['Condition'] == 'Hyperemic']           # keep hyperemic rows
df.rename(columns={'P_d/P_a': 'FFR'}, inplace=True)

our_data = True  # set True if you want the long feature list

if our_data:
    feature_cols = [
        'Stenosis Percentage', 'Length', 'Width', 'HMR', 'HSR', 'BMR/HMR',
        'HSR', 'P_Loss_Coeff', 'WSS_TE', 'WSS_LE', 'WSS_TE_Area',
        'WSS_LE_Area', 'WSS_Area_Bifur', 'WSS_Bif', 'WSS_LMB', 'WSS_min',
        'WSS_TE_min', 'WSS_LE_min', 'WSS_TE_Area_min',
        'WSS_Area_Bifur_min', 'v_distal'
    ]
else:
    feature_cols = ['HMR', 'BMR/HMR', 'P_Loss_Coeff']

output_col = 'discord'                                ### CHANGED ###

df_model = df.dropna(subset=feature_cols + [output_col])

X = df_model[feature_cols]
y = df_model[output_col].astype(int)                  ### CHANGED ###

print("Data shape (features):", X.shape, "| Output shape:", y.shape)

# ────────────────────────────────────────────────────────────────────────────
# 2) TRAIN / TEST SPLIT  (stratified because classes may be imbalanced)
# ────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y)     ### CHANGED ###
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ────────────────────────────────────────────────────────────────────────────
# 3) 5‑FOLD STRATIFIED CV (ROC‑AUC)
# ────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(class_weight='balanced', max_iter=1000))
])

cv_auc = cross_val_score(pipe, X_train, y_train,
                         scoring='roc_auc', cv=skf)
cv_acc = cross_val_score(pipe, X_train, y_train,
                         scoring='accuracy', cv=skf)

print("\n=== Baseline Logistic Regression (discord) ===")
print(f"5‑fold CV  ROC‑AUC : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"5‑fold CV  Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 4) FINAL TRAIN / TEST FIT
# ────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred      = logreg.predict(X_test_scaled)
y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]

test_acc  = accuracy_score(y_test, y_pred)
test_auc  = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest Accuracy : {test_acc:.3f}")
print(f"Test ROC‑AUC  : {test_auc:.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 5) PLOT #1 ─ Confusion Matrix
# ────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Concordant', 'Discord'],
            yticklabels=['Concordant', 'Discord'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────────────────
# 6) PLOT #2 ─ Permutation Importance (ROC‑AUC drop)
# ────────────────────────────────────────────────────────────────────────────
perm = permutation_importance(
    logreg, X_test_scaled, y_test,
    scoring='roc_auc', n_repeats=10, random_state=42)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1]

sorted_feats = [feature_cols[i] for i in indices]
sorted_imps  = importances[indices]

plt.figure()
plt.barh(sorted_feats, sorted_imps)
plt.gca().invert_yaxis()
plt.xlabel("Mean decrease in ROC‑AUC")
plt.title("Permutation Importance (discord classifier)")
plt.tight_layout()
plt.show()
