#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:27:17 2025

@author: tejjolly
"""

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

# Focus only on hyperemic rows if that's relevant
df = df[df['Condition'] == 'Hyperemic']

# Drop rows that lack needed columns
required_cols = ['P_d/P_a','CFR','CFR/FFR','HSR','HMR','BMR/HMR','Average Flow']
df = df.dropna(subset=required_cols)

########################################
# 2) DEFINE BINARY TARGETS
########################################
df['FFR'] = (df['P_d/P_a'] > 0.8).astype(int)        # 1 if FFR>0.8
df['CFR'] = (df['CFR'] > 2.0).astype(int)            # 1 if CFR>2.0
df['CFR_FFR'] = (df['CFR/FFR'] > 2.0).astype(int)     # 1 if CFR/FFR>2.0

########################################
# 3) GLOBAL SPLIT (60/20/20)
########################################
# We'll do it once for the entire DataFrame, 
# then pick columns for each classification inside the loop.

# Let's define some columns we definitely need for splitting:
# We'll keep all rows (df.index) in a single array.
all_indices = df.index.to_numpy()

# For demonstration, let's do X_temp, X_test, y_temp, y_test 
# with "some" target. But we have 3 different targets, 
# so the simplest approach is to create a "dummy" y just for splitting. 
# We'll just pick 'y_FFR' as a placeholder to define the splits 
# so each row consistently goes to train/val/test.

y_dummy = df['FFR'].values  # used only for splitting row indices

# # Step 1) split off 20% for test
# idx_temp, idx_test = train_test_split(
#     all_indices, test_size=0.2, stratify=y_dummy, random_state=10
# )

# # Step 2) from the remaining 80%, take 25% => 20% overall for val
# y_dummy_temp = df.loc[idx_temp, 'FFR'].values
# idx_train, idx_val = train_test_split(
#     idx_temp, test_size=0.25, stratify=y_dummy_temp, random_state=10
# )

# Step 1) split off 20% for test
idx_temp, idx_test = train_test_split(
    all_indices, test_size=0.2, random_state=10
)

# Step 2) from the remaining 80%, take 25% => 20% overall for val
y_dummy_temp = df.loc[idx_temp, 'FFR'].values
idx_train, idx_val = train_test_split(
    idx_temp, test_size=0.25, random_state=10
)

print("Train size:", len(idx_train), "Val size:", len(idx_val), "Test size:", len(idx_test))

########################################
# 4) HELPER FUNCTION: train_svm_and_plot
########################################
def train_svm_and_plot(
    df,
    train_idx, val_idx, test_idx,
    xcol, ycol,
    target_col, target_name
):
    """
    - df: full dataframe
    - train_idx, val_idx, test_idx: arrays of row indices for each split
    - xcol, ycol: feature column names (2D)
    - target_col: name of binary target
    - target_name: string for plot title
    """

    # 1) Build separate DataFrames for train/val/test
    df_train = df.loc[train_idx].dropna(subset=[xcol,ycol,target_col])
    df_val   = df.loc[val_idx].dropna(subset=[xcol,ycol,target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol,ycol,target_col])

    # 2) Extract X, y for each subset
    X_train = df_train[[xcol, ycol]].values
    y_train = df_train[target_col].values

    X_val   = df_val[[xcol, ycol]].values
    y_val   = df_val[target_col].values

    X_test  = df_test[[xcol, ycol]].values
    y_test  = df_test[target_col].values

    # 3) Scale based on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # 4) Train SVM on (train)
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X_train_scaled, y_train)

    # Evaluate on val
    val_acc = svm_clf.score(X_val_scaled, y_val)
    print(f"[{target_name} | {xcol} vs {ycol}] Validation Acc: {val_acc:.3f}")

    # Evaluate on test
    test_acc = svm_clf.score(X_test_scaled, y_test)
    print(f"[{target_name} | {xcol} vs {ycol}] Test Acc: {test_acc:.3f}")

    # 5) Plot decision boundary (train + test points on the same figure)
    #    a) combine train + test data for a single boundary
    #       We'll combine them just for plotting, 
    #       or you could do the boundary from train only 
    #       (though it's typically the same model).
    X_all = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.concatenate((y_train, y_test))

    # Build mesh around all scaled data
    x_min, x_max = X_all[:,0].min() - 0.5, X_all[:,0].max() + 0.5
    y_min, y_max = X_all[:,1].min() - 0.5, X_all[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_clf.predict(mesh_points).reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

    # Plot train points
    plt.scatter(
        X_train_scaled[y_train==0, 0],
        X_train_scaled[y_train==0, 1],
        c='red', edgecolors='k', label='Train class 0', marker='o', alpha=0.3
    )
    plt.scatter(
        X_train_scaled[y_train==1, 0],
        X_train_scaled[y_train==1, 1],
        c='green', edgecolors='k', label='Train class 1', marker='o', alpha=0.3
    )

    # Plot test points
    plt.scatter(
        X_test_scaled[y_test==0, 0],
        X_test_scaled[y_test==0, 1],
        c='red', edgecolors='k', label='Test class 0', marker='X'
    )
    plt.scatter(
        X_test_scaled[y_test==1, 0],
        X_test_scaled[y_test==1, 1],
        c='green', edgecolors='k', label='Test class 1', marker='X'
    )

    plt.xlabel(f"{xcol} (scaled)")
    plt.ylabel(f"{ycol} (scaled)")
    # plt.title(f"Target: {target_name}\n({xcol} vs. {ycol})")
    plt.title(f"Target: {target_name}\nTest Acc: {test_acc:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


########################################
# 5) LOOP OVER (TARGET, PREDICTOR-PAIR) 
########################################
targets = [
    ('FFR',     "FFR > 0.8"),
    ('CFR',     "CFR > 2.0"),
    ('CFR_FFR', "CFR/FFR > 2.0")
]

predictor_pairs = [
    ('HMR','HSR'),          
    ('HSR','BMR/HMR'),
    ('HSR','Average Flow')
]

for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        train_svm_and_plot(
            df, 
            idx_train, idx_val, idx_test, 
            xcol, ycol, 
            target_col, target_name
        )
        print("----------------------------------------------------")
