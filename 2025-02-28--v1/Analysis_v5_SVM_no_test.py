#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:10:24 2025

@author: tejjolly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

########################################
# 1) LOAD YOUR SUMMARY CSV
########################################
df = pd.read_csv("summary.csv")

# We assume your CSV has columns:
# 'P_d/P_a' (FFR), 'CFR', 'CFR/FFR', plus 'HSR', 'HMR', 'BMR/HMR', 'Average Flow', etc.

# Optionally focus only on Hyperemic rows if that’s relevant:
df = df[df['Condition'] == 'Hyperemic']

# Drop rows that lack needed columns
required_cols = ['P_d/P_a','CFR','CFR/FFR','HSR','HMR','BMR/HMR','Average Flow']
df = df.dropna(subset=required_cols)

########################################
# 2) DEFINE BINARY TARGETS
########################################
# y_FFR = 1 if FFR>0.8 else 0
df['y_FFR'] = (df['P_d/P_a'] > 0.8).astype(int)

# y_CFR = 1 if CFR>2.0 else 0
df['y_CFR'] = (df['CFR'] > 2.0).astype(int)

# y_CFR_FFR = 1 if CFR/FFR>2.0 else 0
df['y_CFR_FFR'] = (df['CFR/FFR'] > 2.0).astype(int)

########################################
# 3) SET UP OUR 3 TARGETS & 3 PREDICTOR PAIRS
########################################
targets = [
    ('y_FFR',  "FFR > 0.8"),
    ('y_CFR',  "CFR > 2.0"),
    ('y_CFR_FFR', "CFR/FFR > 2.0")
]

predictor_pairs = [
    ('HSR','HMR'),          # (x1, x2)
    ('HSR','BMR/HMR'),
    ('HSR','Average Flow')
]

########################################
# 4) HELPER FUNCTION TO TRAIN SVM & PLOT 2D DECISION BOUNDARY
########################################
def train_svm_and_plot(df, xcol, ycol, target_col, target_name):
    """
    df: DataFrame with columns xcol, ycol, target_col
    xcol, ycol: strings for the feature columns
    target_col: string for the binary target column
    target_name: string label for plot title

    This function:
      1) Extracts data
      2) Scales features
      3) Trains an SVM
      4) Plots the 2D decision boundary & data points
    """
    X = df[[xcol, ycol]].values
    y = df[target_col].values
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM (use RBF kernel for illustration)
    svm_clf = SVC(kernel='linear', C=1.0, gamma='scale')
    svm_clf.fit(X_scaled, y)

    # Plot decision boundary
    # 1) Create a mesh grid covering the range of X_scaled
    x_min, x_max = X_scaled[:,0].min() - 0.5, X_scaled[:,0].max() + 0.5
    y_min, y_max = X_scaled[:,1].min() - 0.5, X_scaled[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_clf.predict(mesh_points).reshape(xx.shape)

    # 2) Plot the contour
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # 3) Plot data points
    #    We'll color them by their actual class (0 or 1)
    #    We can also mark correct vs. incorrect if we want
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1],
                color='blue', edgecolors='k', label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1],
                color='red', edgecolors='k', label='Class 1')

    # Titles & legend
    plt.xlabel(f"{xcol} (scaled)")
    plt.ylabel(f"{ycol} (scaled)")
    plt.title(f"Target: {target_name}  |  Features: {xcol} vs. {ycol}")
    plt.legend()
    plt.tight_layout()
    plt.show()


########################################
# 5) LOOP OVER (TARGET, PREDICTOR-PAIR) TO PLOT
########################################
for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        # Subset data to valid rows (in case of any leftover NaNs)
        sub_df = df.dropna(subset=[xcol, ycol, target_col])
        # Train & plot
        train_svm_and_plot(sub_df, xcol, ycol, target_col, target_name)
