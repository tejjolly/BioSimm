# USE ATER AN_BOTTLENECK_V4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text


features = ["HMR", "P_Loss_Coeff", "BMR/HMR", "discord"]

target_col = "discord"
feature_cols = features[:-1]

# Load the saved arrays
X_all = np.load("X_all.npy")
z_all = np.load("z_all.npy")
y_all = np.load("y_all.npy")

rf = RandomForestRegressor()
rf.fit(X_all, z_all)
print("Random Forest R²:", rf.score(X_all, z_all))
z_rf_pred = rf.predict(X_all)
plt.scatter(z_all, z_rf_pred, alpha=0.5)
plt.xlabel("True z")
plt.ylabel("Predicted z (RF)")
plt.title("Random Forest on z")
plt.grid(True)
plt.plot([z_all.min(), z_all.max()], [z_all.min(), z_all.max()], 'k--')
plt.show()

import numpy as np

importances = rf.feature_importances_
for i, col in enumerate(feature_cols):
    print(f"Feature: {col}, Importance: {importances[i]:.4f}")

from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_all, z_all, n_repeats=5, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

for i in perm_sorted_idx:
    print(f"Feature: {feature_cols[i]}, Importance Mean: {result.importances_mean[i]:.4f}")

from sklearn.inspection import partial_dependence, PartialDependenceDisplay

for i, col in enumerate(feature_cols):
    disp = PartialDependenceDisplay.from_estimator(rf, X_all, [i])
    disp.figure_.suptitle(f"Partial dependence of z on {col}")
    plt.show()

# # import matplotlib.pyplot as plt
# plt.hist(z_all[:, 0], bins=50)
# plt.title("Distribution of z[0]")
# plt.grid(True)
# plt.show()
#
# plt.hist(z_all[:, 1], bins=50)
# plt.title("Distribution of z[1]")
# plt.grid(True)
# plt.show()

# plt.hist(z_all[:, 2], bins=50)
# plt.title("Distribution of z[2]")
# plt.grid(True)
# plt.show()

for i in range(z_all.shape[1]):
    z_col = z_all[:, i]
    print(f"\nSummary for z[{i}]:")
    print(f"  Min     : {np.min(z_col):.4f}")
    print(f"  Max     : {np.max(z_col):.4f}")
    print(f"  Mean    : {np.mean(z_col):.4f}")
    print(f"  Std Dev : {np.std(z_col):.4f}")
    print(f"  Median  : {np.median(z_col):.4f}")
    print(f"  25%     : {np.percentile(z_col, 25):.4f}")
    print(f"  75%     : {np.percentile(z_col, 75):.4f}")

    plt.hist(z_col, bins=50)
    plt.title(f"Distribution of z[{i}]")
    plt.grid(True)
    plt.show()

if z_all.shape[1] > 1:
    plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='coolwarm')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Bottleneck colored by class label")
    plt.grid(True)
    plt.show()
else:
    z = z_all.ravel()
    y = y_all

    plt.figure(figsize=(8, 4))
    plt.scatter(z, np.random.normal(0, 0.03, size=len(z)), c=y, cmap='coolwarm', alpha=0.6, edgecolors='k')
    plt.xlabel("z[0]")
    plt.yticks([])  # Hide y-axis since it's fake
    plt.title("1D Bottleneck colored by class")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
