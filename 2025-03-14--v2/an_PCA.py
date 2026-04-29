import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# 1) LOAD CSV AND PREPARE DATA
# --------------------------
csv_path = "./summary3.csv"
df = pd.read_csv(csv_path)

# Specify the 20 feature columns and the target column ("CFR/FFR")
feature_cols = [
    "Stenosis Percentage", "Length", "Width", "HMR", "HSR",
    "P_Loss_Coeff", "WSS_TE", "WSS_LE", "WSS_TE_Area", "WSS_LE_Area",
    "BMR/HMR", "WSS_Area_Bifur", "WSS_Bif", "WSS_LMB",
    "WSS_min", "WSS_TE_min", "WSS_LE_min", "WSS_TE_Area_min",
    "WSS_LE_Area_min", "WSS_Area_Bifur_min"
]
target_col = "CFR/FFR"

X = df[feature_cols].values  # shape: [num_samples, 20]
y = df[target_col].values  # shape: [num_samples]

# Convert to PyTorch tensors (features and target as floats)
X_tensor = torch.from_numpy(X).float()
# For regression, we make y shape [N,1]
y_tensor = torch.from_numpy(y).float().unsqueeze(-1)


# --------------------------
# 2) DEFINE A SIMPLE MODEL WITH DROPOUT
# --------------------------
# We use a very simple network:
#  - fc1: input -> 10 neurons, followed by ReLU and dropout (p=0.3)
#  - fc2: 10 -> 1 (output) with linear activation (for regression)
#
# We will record the hidden representation from fc1.

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)  # output for regression

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # We'll record the output of fc1 (after ReLU) as the hidden representation.
        hidden = x.clone()
        x = self.dropout(x)
        x = self.fc2(x)
        return x, hidden  # return both final output and hidden rep


# --------------------------
# 3) SET UP CROSS-VALIDATION, EARLY STOPPING, AND DATALOADERS
# --------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

num_epochs = 1000  # maximum epochs (early stopping will likely trigger before this)
patience_limit = 10  # epochs to wait for improvement

# We'll record validation losses for each fold for reporting
fold_val_losses = []

# For visualization, we will record PCA history for the hidden representations in the first fold only.
pca2d_history = []  # list of 2D PCA snapshots
pca3d_history = []  # list of 3D PCA snapshots
epoch_record = []  # which epochs we recorded from fold 0

# For convenience, create a TensorDataset for the whole dataset.
dataset = TensorDataset(X_tensor, y_tensor)

fold_index = 0
for train_index, val_index in kf.split(X):
    print(f"\n----- Fold {fold_index + 1} / {n_splits} -----")
    # Create DataLoaders for the fold
    train_subset = torch.utils.data.Subset(dataset, train_index)
    val_subset = torch.utils.data.Subset(dataset, val_index)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    # Instantiate a new model for this fold
    model = SimpleRegressor(input_dim=X.shape[1], hidden_dim=10)
    # Use Adam with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # For fold 0, we will record PCA snapshots from the training set
    fold0_pca2d = []
    fold0_pca3d = []
    fold0_epochs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # --- Training Loop ---
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds, _ = model(batch_x)  # we ignore the hidden rep here for training
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        train_loss = running_loss / len(train_subset)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds, _ = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)
        val_loss = val_loss / len(val_subset)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping: check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # For fold 0, record PCA snapshots for the training set every epoch.
        if fold_index == 0:
            # Run a forward pass on the whole training set (without shuffling) to collect hidden reps.
            hidden_list = []
            model.eval()
            with torch.no_grad():
                for bx, by in DataLoader(train_subset, batch_size=16, shuffle=False):
                    _, hidden = model(bx)
                    hidden_list.append(hidden)
            hidden_all = torch.cat(hidden_list, dim=0)  # shape [n_train, hidden_dim]
            H_np = hidden_all.cpu().numpy()

            # PCA to 2D and 3D
            pca_2d = PCA(n_components=2).fit_transform(H_np)
            pca_3d = PCA(n_components=3).fit_transform(H_np)

            fold0_pca2d.append(pca_2d)
            fold0_pca3d.append(pca_3d)
            fold0_epochs.append(epoch + 1)

    # End of epochs for current fold
    fold_val_losses.append(best_val_loss)
    print(f"Best Val Loss for fold {fold_index + 1}: {best_val_loss:.4f}")

    # If this is the first fold, save its PCA history for later plotting.
    if fold_index == 0:
        pca2d_history = fold0_pca2d
        pca3d_history = fold0_pca3d
        epoch_record = fold0_epochs
    fold_index += 1

# Report cross-validation results
avg_val_loss = np.mean(fold_val_losses)
print("\n----- Cross-Validation Results -----")
print(f"Fold validation losses: {fold_val_losses}")
print(f"Average validation loss: {avg_val_loss:.4f}")

# --------------------------
# 4) PLOTTING PCA SNAPSHOTS
# --------------------------
# We want 10 snapshots evenly spaced from the recorded epochs.
total_epochs_recorded = len(epoch_record)
if total_epochs_recorded < 10:
    snapshot_indices = list(range(total_epochs_recorded))
else:
    step = total_epochs_recorded // 10
    snapshot_indices = [i * step for i in range(10)]

# --- (A) 2D PCA: Plot 10 subplots in one figure ---
fig2d, axes2d = plt.subplots(2, 5, figsize=(12, 8), subplot_kw={'aspect': 'equal'})
axes2d = axes2d.flatten()

for i, idx in enumerate(snapshot_indices):
    ax = axes2d[i]
    coords_2d = pca2d_history[idx]
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=None,  # no colormap for regression target; you can change if desired
                    alpha=0.7, cmap="viridis")
    # Here, we'll color-code by the target value from the training set.
    # Since the training set is small, we can get the y values from the first fold.
    # (Note: The order in the DataLoader should be consistent here.)
    # We reload the training subset from fold 0:
    first_fold_indices = list(kf.split(X))[0][0]  # training indices of fold 0
    y_train_fold0 = y_tensor[first_fold_indices].squeeze(-1).numpy()
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=y_train_fold0, alpha=0.7, cmap="viridis")
    ax.set_title(f"Epoch {epoch_record[idx]}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
fig2d.suptitle("2D PCA of Hidden Layer (Snapshots)", fontsize=16)
fig2d.colorbar(sc, ax=axes2d, shrink=0.7, label="CFR/FFR")
plt.tight_layout()
plt.show()

# --- (B) 3D PCA: Plot 10 separate 3D figures ---
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots

for i, idx in enumerate(snapshot_indices):
    coords_3d = pca3d_history[idx]
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(projection='3d')

    # Use the same training target values for color-coding
    p = ax3d.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                     c=y_train_fold0, alpha=0.7, cmap="viridis")
    ax3d.set_title(f"3D PCA at Epoch {epoch_record[idx]}")
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    fig3d.colorbar(p, ax=ax3d, fraction=0.03, pad=0.1, label="CFR/FFR")
    plt.tight_layout()
    plt.show()
