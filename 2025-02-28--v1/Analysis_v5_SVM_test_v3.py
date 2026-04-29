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
# 3) GLOBAL SPLIT (80/20, merging train + val)
########################################
all_indices = df.index.to_numpy()
y_dummy = df['FFR'].values  # Placeholder for consistent splitting

# Step 1: Split 20% for testing
idx_train_full, idx_test = train_test_split(
    all_indices, test_size=0.2, stratify=y_dummy, random_state=10
)

print("Train+Val size:", len(idx_train_full), "Test size:", len(idx_test))

########################################
# 4) HELPER FUNCTION: train_svm_and_plot
########################################
def train_svm_and_plot(
    df,
    train_idx, test_idx,
    xcol, ycol,
    target_col, target_name
):
    """
    - df: full dataframe
    - train_idx, test_idx: arrays of row indices for each split
    - xcol, ycol: feature column names (2D)
    - target_col: name of binary target
    - target_name: string for plot title
    """

    # 1) Build separate DataFrames for train/test
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])

    # 2) Extract X, y for each subset (store original unscaled values)
    X_train_unscaled = df_train[[xcol, ycol]].values
    X_test_unscaled  = df_test[[xcol, ycol]].values

    y_train = df_train[target_col].values
    y_test  = df_test[target_col].values

    # 3) Scale based on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test_scaled  = scaler.transform(X_test_unscaled)

    # 4) Train SVM on full training set
    svm_clf = SVC(kernel='linear', C=1.0)
    svm_clf.fit(X_train_scaled, y_train)

    # Evaluate on test set
    test_acc = svm_clf.score(X_test_scaled, y_test)
    print(f"[{target_name} | {xcol} vs {ycol}] Test Acc: {test_acc:.3f}")

    # 5) Generate decision boundary **in scaled space**
    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_clf.predict(mesh_points).reshape(xx.shape)

    # 6) Transform mesh grid back to **unscaled space**
    xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

    # 7) Plot using **unscaled data**
    plt.figure(figsize=(6,5))
    plt.contourf(xx_unscaled, yy_unscaled, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

    # Plot train points (using unscaled data)
    plt.scatter(
        X_train_unscaled[y_train == 0, 0],
        X_train_unscaled[y_train == 0, 1],
        c='red', edgecolors='k', label='Train class 0', marker='o', alpha=0.6
    )
    plt.scatter(
        X_train_unscaled[y_train == 1, 0],
        X_train_unscaled[y_train == 1, 1],
        c='green', edgecolors='k', label='Train class 1', marker='o', alpha=0.6
    )

    # Plot test points (using unscaled data)
    plt.scatter(
        X_test_unscaled[y_test == 0, 0],
        X_test_unscaled[y_test == 0, 1],
        c='red', edgecolors='k', label='Test class 0', marker='X'
    )
    plt.scatter(
        X_test_unscaled[y_test == 1, 0],
        X_test_unscaled[y_test == 1, 1],
        c='green', edgecolors='k', label='Test class 1', marker='X'
    )

    plt.xlabel(f"{xcol}")
    plt.ylabel(f"{ycol}")
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
            idx_train_full, idx_test, 
            xcol, ycol, 
            target_col, target_name
        )
        print("----------------------------------------------------")

# Adjusting the plotting code to arrange in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(11,9))  # 16:9 aspect ratio for PowerPoint

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Counter to track subplot index
plot_idx = 0

for (target_col, target_name) in targets:
    for (xcol, ycol) in predictor_pairs:
        # Prepare data
        df_train = df.loc[idx_train_full].dropna(subset=[xcol, ycol, target_col])
        df_test = df.loc[idx_test].dropna(subset=[xcol, ycol, target_col])

        X_train_unscaled = df_train[[xcol, ycol]].values
        X_test_unscaled = df_test[[xcol, ycol]].values

        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_unscaled)
        X_test_scaled = scaler.transform(X_test_unscaled)

        svm_clf = SVC(kernel='linear', C=1.0)
        svm_clf.fit(X_train_scaled, y_train)

        test_acc = svm_clf.score(X_test_scaled, y_test)

        x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
        y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm_clf.predict(mesh_points).reshape(xx.shape)

        xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
        yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

        ax = axes[plot_idx]
        ax.contourf(xx_unscaled, yy_unscaled, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

        ax.scatter(
            X_train_unscaled[y_train == 0, 0], X_train_unscaled[y_train == 0, 1],
            c='red', edgecolors='k', label='Train class 0', marker='o', alpha=0.6
        )
        ax.scatter(
            X_train_unscaled[y_train == 1, 0], X_train_unscaled[y_train == 1, 1],
            c='green', edgecolors='k', label='Train class 1', marker='o', alpha=0.6
        )

        ax.scatter(
            X_test_unscaled[y_test == 0, 0], X_test_unscaled[y_test == 0, 1],
            c='red', edgecolors='k', label='Test class 0', marker='X'
        )
        ax.scatter(
            X_test_unscaled[y_test == 1, 0], X_test_unscaled[y_test == 1, 1],
            c='green', edgecolors='k', label='Test class 1', marker='X'
        )

        ax.set_xlabel(f"{xcol}")
        ax.set_ylabel(f"{ycol}")
        ax.set_title(f"{target_name}\nTest Acc: {test_acc:.2f}")

        plot_idx += 1  # Move to the next subplot

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("svm_subplots.png", dpi=900)  # High-res for PowerPoint
plt.show()
