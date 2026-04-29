#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:46:07 2025

@author: tejjolly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: SVM + cross-validation to get mean accuracy, plus a final fit for 2D boundary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

########################################
# 1) LOAD YOUR SUMMARY CSV
########################################
df = pd.read_csv("../data/data.csv")

# Focus only on hyperemic rows if that's relevant
df = df[df['Condition'] == 'Hyperemic']

# Drop rows that lack needed columns
required_cols = ['P_d/P_a','CFR','CFR/FFR','HSR','HMR','BMR/HMR','Average Flow']
df = df.dropna(subset=required_cols)

########################################
# 2) DEFINE BINARY TARGETS
########################################
df['FFR'] = (df['P_d/P_a'] > 0.8).astype(int)       # 1 if FFR>0.8
df['CFR'] = (df['CFR'] > 2.0).astype(int)           # 1 if CFR>2.0
df['CFR_FFR'] = (df['CFR/FFR'] > 2.0).astype(int)    # 1 if CFR/FFR>2.0

########################################
# 3) SPECIFY TARGETS & PREDICTOR PAIRS
########################################
targets = [
    ('FFR',     "FFR > 0.8"),
    ('CFR',     "CFR > 2.0"),
    ('CFR_FFR', "CFR/FFR > 2.0")
]

predictor_pairs = [
    ('HSR','HMR'),
    ('HSR','BMR/HMR'),
    ('HSR','Average Flow')
]

########################################
# 4) FUNCTION: CROSS-VALIDATION + PLOT
########################################
def crossval_and_plot_svm(df, xcol, ycol, target_col, target_name):
    """
    1) Does cross-validation on sub_df for (xcol, ycol) => target_col.
    2) Prints mean ± std of accuracy.
    3) Fits SVM on the entire sub_df, plots 2D boundary + points.
    """
    # Subset DataFrame
    sub_df = df.dropna(subset=[xcol, ycol, target_col])
    X = sub_df[[xcol, ycol]].values
    y = sub_df[target_col].values
    
    # A) CROSS-VALIDATION with pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=1.0))
    ])
    
    # 5-fold stratified CV

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=38)
    
    cv_scores = cross_val_score(pipeline, X, y, 
                                scoring='accuracy', 
                                cv=skf, 
                                n_jobs=-1)
    mean_acc = np.mean(cv_scores)
    std_acc  = np.std(cv_scores)
    print(f"\n[Target: {target_name} | Features: {xcol}, {ycol}]")
    print(f"CV Accuracy: {mean_acc:.3f} ± {std_acc:.3f}  (n= {len(X)})")
    
    # B) FINAL FIT FOR VISUALIZATION
    #    We'll fit on the entire sub_df to show the boundary of "full data" SVM
    pipeline.fit(X, y)
    
    # Retrieve the scaler & classifier from the pipeline
    scaler = pipeline.named_steps['scaler']
    svm_clf = pipeline.named_steps['svc']
    
    # Scale the entire dataset
    X_scaled = scaler.transform(X)
    
    # Make a mesh for the boundary
    x_min, x_max = X_scaled[:,0].min() - 0.5, X_scaled[:,0].max() + 0.5
    y_min, y_max = X_scaled[:,1].min() - 0.5, X_scaled[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_clf.predict(grid_data).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot points
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1],
                c='blue', edgecolors='k', label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1],
                c='red', edgecolors='k', label='Class 1')
    
    plt.xlabel(f"{xcol} (scaled)")
    plt.ylabel(f"{ycol} (scaled)")
    plt.title(f"SVM (Full fit) | {target_name}\n"
              f"Accuracy (CV mean ± std): {mean_acc:.2f} ± {std_acc:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


########################################
# 5) LOOP OVER TARGETS & FEATURE-PAIRS
########################################
for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        crossval_and_plot_svm(df, xcol, ycol, target_col, target_name)


import matplotlib.patches as mpatches

# Generate a horizontal bar visualization for the 5-fold cross-validation splits
def plot_cv_splits_horizontal(df, target_col):
    """
    Plots a horizontal bar visualization showing the 5-fold CV splits.
    Each bar represents one fold, with colored sections showing test/train positions.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    X = np.arange(len(df))
    y = df[target_col].values
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        split_bar = np.full(len(df), np.nan)  # Initialize with NaN (no assignment)
        split_bar[test_idx] = i  # Mark test set for fold i
        
        # Create a horizontal bar for each fold
        ax.barh(i, len(df), left=0, color='lightgray', edgecolor='black', height=0.5)
        
        # Overlay test set with corresponding color
        for idx in test_idx:
            ax.barh(i, 1, left=idx, color=colors[i], edgecolor='black', height=0.5)

    # Formatting
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_xlabel("Sample Index")
    ax.set_title(f"5-Fold Cross-Validation Splits ({target_col})")
    
    # Create a legend
    patches = [mpatches.Patch(color=colors[i], label=f"Fold {i+1} Test Set") for i in range(5)]
    ax.legend(handles=patches, loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.show()

# Example visualization for one of the target columns
plot_cv_splits_horizontal(df, 'FFR')

