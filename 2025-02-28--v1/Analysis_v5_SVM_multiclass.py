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

# Optionally filter to only "Hyperemic" rows
df = df[df['Condition'] == 'Hyperemic']

# Define which columns you need
required_cols = ['P_d/P_a', 'CFR', 'CFR/FFR', 'HSR', 'HMR', 'BMR/HMR', 'Average Flow']
df = df.dropna(subset=required_cols)

########################################
# 2) DEFINE THE 3-CLASS FFR TARGET
########################################
# Class 0: FFR < 0.70
# Class 1: 0.70 <= FFR < 0.80
# Class 2: FFR >= 0.80
df['FFR_3class'] = np.where(
    df['P_d/P_a'] < 0.70, 
    0, 
    np.where(df['P_d/P_a'] < 0.80, 1, 2)
)

########################################
# 3) GLOBAL SPLIT (80/20) WITH STRATIFICATION
########################################
all_indices = df.index.to_numpy()
y_3class = df['FFR_3class'].values

idx_train_full, idx_test = train_test_split(
    all_indices, 
    test_size=0.20,
    random_state=40,   # for reproducibility
    stratify=y_3class  # ensures class proportions are preserved
)

print("Train+Val size:", len(idx_train_full), 
      "Test size:", len(idx_test))

########################################
# 4) HELPER FUNCTION: train_svm_three_class_and_plot
########################################
def train_svm_three_class_and_plot(
    df,
    train_idx, test_idx,
    xcol, ycol,
    target_col, target_name
):
    """
    - df: full dataframe
    - train_idx, test_idx: arrays of row indices for each split
    - xcol, ycol: feature column names (2D) for plotting
    - target_col: name of the 3-class target
    - target_name: string for the plot title
    """

    # 1) Build separate DataFrames for train/test
    df_train = df.loc[train_idx].dropna(subset=[xcol, ycol, target_col])
    df_test  = df.loc[test_idx].dropna(subset=[xcol, ycol, target_col])

    # 2) Extract X, y for each subset
    X_train_unscaled = df_train[[xcol, ycol]].values
    X_test_unscaled  = df_test[[xcol, ycol]].values

    y_train = df_train[target_col].values
    y_test  = df_test[target_col].values

    # 3) Scale based on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test_scaled  = scaler.transform(X_test_unscaled)

    # 4) Train Multi-Class SVM (linear kernel)
    svm_clf = SVC(kernel='linear', C=5)
    svm_clf.fit(X_train_scaled, y_train)

    # Evaluate on test set
    test_acc = svm_clf.score(X_test_scaled, y_test)
    print(f"[{target_name} | {xcol} vs {ycol}] Test Acc: {test_acc:.3f}")

    # 5) Generate decision boundary in scaled space
    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_clf.predict(mesh_points).reshape(xx.shape)

    # 6) Transform mesh grid back to unscaled space
    xx_unscaled = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_unscaled = yy * scaler.scale_[1] + scaler.mean_[1]

    # 7) Plot using unscaled data
    plt.figure(figsize=(6,5))

    # Create a 3-color colormap for classes 0,1,2 (pastel red, pastel yellow, pastel green)
    from matplotlib.colors import ListedColormap
    cmap_3class = ListedColormap(['#FFCCCC', '#FFFFCC', '#CCFFCC'])  # background fill

    plt.contourf(xx_unscaled, yy_unscaled, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

    # For convenience, define color/marker arrays (Class 0 → red, Class 1 → yellow, Class 2 → green)
    colors        = ['red','yellow','green']
    markers_train = ['o','o','o']
    labels_train  = ['Train class 0','Train class 1','Train class 2']

    # Plot the train points, colored by class:
    for cidx in [0,1,2]:
        plt.scatter(
            X_train_unscaled[y_train == cidx, 0],
            X_train_unscaled[y_train == cidx, 1],
            c=colors[cidx],
            edgecolors='k',
            label=labels_train[cidx],
            marker=markers_train[cidx],
            alpha=0.6
        )

    # Plot the test points with a different marker
    markers_test = ['X','X','X']
    labels_test  = ['Test class 0','Test class 1','Test class 2']
    for cidx in [0,1,2]:
        plt.scatter(
            X_test_unscaled[y_test == cidx, 0],
            X_test_unscaled[y_test == cidx, 1],
            c=colors[cidx],
            edgecolors='k',
            label=labels_test[cidx],
            marker=markers_test[cidx]
        )

    plt.xlabel(f"{xcol}")
    plt.ylabel(f"{ycol}")
    plt.title(f"Target: {target_name}\nTest Acc: {test_acc:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

########################################
# 5) LOOP OVER (PREDICTOR-PAIR) 
########################################
predictor_pairs = [
    ('HMR','HSR'),
    ('HSR','BMR/HMR'),
    ('HSR','Average Flow')
]

for (xcol, ycol) in predictor_pairs:
    train_svm_three_class_and_plot(
        df, 
        idx_train_full, idx_test, 
        xcol, ycol, 
        'FFR_3class', 
        "FFR<0.7  |  0.7<FFR<0.8  |  0.8<FFR"
    )
    print("----------------------------------------------------")
