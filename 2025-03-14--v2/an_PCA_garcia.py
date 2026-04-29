import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# 1) LOAD CSV AND PREPARE DATA
# --------------------------
csv_path = "./summary_garcia.csv"
df = pd.read_csv(csv_path)

# Specify the feature columns (3 features) and the target column ("CFR/FFR")
feature_cols = ["HMR", "P_Loss_Coeff", "BMR"]
target_col = "CFR/FFR"

X = df[feature_cols].values  # shape: [num_samples, 3]
y = df[target_col].values    # shape: [num_samples]

# ---- NORMALIZATION STEP ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # shape: [num_samples, 3]

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(-1)  # shape: [N,1]

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# --------------------------
# 2) DEFINE A SIMPLE MLP WITH DROPOUT
# --------------------------
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
        return x, hidden  # final output and hidden rep

# --------------------------
# 3) SET UP CROSS-VALIDATION & EARLY STOPPING
# --------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

num_epochs = 10000  # max epochs
patience_limit = 100

fold_val_losses = []
fold_metrics = []  # to store final metrics from each fold

# For visualization/logging, we record data for fold 0
fold0_train_losses = []
fold0_val_losses = []
hidden_history_fold0 = []      # hidden layer snapshots (fold 0)
epoch_history_fold0 = []       # epoch number for each snapshot
preds_train_fold0 = []         # store final predictions on training set (fold 0)
targets_train_fold0 = []
preds_val_fold0 = []
targets_val_fold0 = []

fold_index = 0
for train_index, val_index in kf.split(X):
    print(f"\n----- Fold {fold_index + 1} / {n_splits} -----")

    # Create DataLoaders
    train_subset = torch.utils.data.Subset(dataset, train_index)
    val_subset = torch.utils.data.Subset(dataset, val_index)
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    # Instantiate a new model for this fold
    model = SimpleRegressor(input_dim=X_tensor.shape[1], hidden_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # For fold 0, store the per-epoch train/val losses
    train_losses_epoch = []
    val_losses_epoch = []

    fold0_hidden_snapshots = []
    fold0_epochs = []

    # 3A) TRAINING LOOP
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        train_loss = running_loss / len(train_subset)

        # 3B) VALIDATION LOOP
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds, _ = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)
        val_loss = val_loss / len(val_subset)

        # For fold 0, record epoch-by-epoch losses
        if fold_index == 0:
            train_losses_epoch.append(train_loss)
            val_losses_epoch.append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Record hidden layer for fold 0
        if fold_index == 0:
            hidden_list = []
            model.eval()
            with torch.no_grad():
                for bx, by in DataLoader(train_subset, batch_size=16, shuffle=False):
                    _, hidden = model(bx)
                    hidden_list.append(hidden)
            hidden_all = torch.cat(hidden_list, dim=0)  # [n_train_fold0, hidden_dim]

            fold0_hidden_snapshots.append(hidden_all)
            fold0_epochs.append(epoch + 1)

    # 3C) LOAD BEST MODEL (early stopping)
    model.load_state_dict(best_model_state)

    # Compute final metrics for this fold
    # We'll gather entire train & val predictions
    model.eval()

    # --- Train predictions (for fold) ---
    train_x_all = torch.stack([dataset[i][0] for i in train_index], dim=0)  # shape [n_train_fold, 3]
    train_y_all = torch.stack([dataset[i][1] for i in train_index], dim=0)  # shape [n_train_fold, 1]
    with torch.no_grad():
        train_preds, _ = model(train_x_all)
    train_preds_np = train_preds.squeeze().numpy()
    train_targets_np = train_y_all.squeeze().numpy()

    # --- Validation predictions (for fold) ---
    val_x_all = torch.stack([dataset[i][0] for i in val_index], dim=0)
    val_y_all = torch.stack([dataset[i][1] for i in val_index], dim=0)
    with torch.no_grad():
        val_preds, _ = model(val_x_all)
    val_preds_np = val_preds.squeeze().numpy()
    val_targets_np = val_y_all.squeeze().numpy()

    # MSE on train/val
    train_mse = np.mean((train_preds_np - train_targets_np)**2)
    val_mse   = np.mean((val_preds_np - val_targets_np)**2)

    # R² on train/val
    train_r2 = r2_score(train_targets_np, train_preds_np)
    val_r2   = r2_score(val_targets_np, val_preds_np)

    # Pearson correlation on train/val
    train_corr, _ = pearsonr(train_preds_np, train_targets_np)
    val_corr, _   = pearsonr(val_preds_np, val_targets_np)

    # Store final fold metrics
    fold_val_losses.append(best_val_loss)
    fold_metrics.append({
        'fold': fold_index,
        'best_val_loss': best_val_loss,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_corr': train_corr,
        'val_corr': val_corr,
    })

    print(f"Best Val Loss for fold {fold_index + 1}: {best_val_loss:.4f}")
    print(f"Final Train MSE: {train_mse:.4f}, R^2: {train_r2:.4f}, Corr: {train_corr:.4f}")
    print(f"Final Val   MSE: {val_mse:.4f}, R^2: {val_r2:.4f}, Corr: {val_corr:.4f}")

    # If this is the first fold, store full data
    if fold_index == 0:
        fold0_train_losses = train_losses_epoch
        fold0_val_losses = val_losses_epoch
        hidden_history_fold0 = fold0_hidden_snapshots
        epoch_history_fold0 = fold0_epochs
        preds_train_fold0 = train_preds_np
        targets_train_fold0 = train_targets_np
        preds_val_fold0 = val_preds_np
        targets_val_fold0 = val_targets_np

    fold_index += 1

# --------------------------
# 4) CROSS-VALIDATION REPORT
# --------------------------
avg_val_loss = np.mean(fold_val_losses)
print("\n----- Cross-Validation Results -----")
print("Fold metrics (MSE, R², Corr) by fold:")
for fm in fold_metrics:
    print(f"Fold {fm['fold']+1}: best_val_loss={fm['best_val_loss']:.4f}, "
          f"train_mse={fm['train_mse']:.4f}, val_mse={fm['val_mse']:.4f}, "
          f"train_r2={fm['train_r2']:.4f}, val_r2={fm['val_r2']:.4f}, "
          f"train_corr={fm['train_corr']:.4f}, val_corr={fm['val_corr']:.4f}")

print(f"\nAverage validation loss across folds: {avg_val_loss:.4f}")

# --------------------------
# 5) PLOT TRAIN vs VAL LOSS (FOLD 0)
# --------------------------
if len(fold0_train_losses) > 0:
    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(fold0_train_losses)+1), fold0_train_losses, label="Train Loss")
    plt.plot(range(1, len(fold0_val_losses)+1), fold0_val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Fold 0: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------
# 6) VISUALIZE THE HIDDEN LAYER IN 3D (FOLD 0)
# --------------------------
if len(epoch_history_fold0) == 0:
    print("\nNo epochs were recorded for fold 0. Possibly your dataset is small or early stopping triggered too soon.")
    import sys; sys.exit()

# Sample 10 snapshots max
total_epochs_recorded = len(epoch_history_fold0)
if total_epochs_recorded < 10:
    snapshot_indices = list(range(total_epochs_recorded))
else:
    step = total_epochs_recorded // 10
    snapshot_indices = [i * step for i in range(10)]

# Retrieve the training targets for fold 0 in the correct order
train_indices_fold0 = list(kf.split(X))[0][0]  # training indices of fold 0
y_train_fold0 = y_tensor[train_indices_fold0].squeeze(-1).numpy()

# Create 2x5 3D subplots in a single figure
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
fig = plt.figure(figsize=(12, 8))
axes = [fig.add_subplot(2, 5, i+1, projection='3d') for i in range(len(snapshot_indices))]

for i, idx in enumerate(snapshot_indices):
    ax = axes[i]
    # hidden_history_fold0[idx] -> tensor [n_train_fold0, 10]
    # We slice out the first 3 hidden neurons for 3D plotting
    coords_3d = hidden_history_fold0[idx][:, :3].detach().numpy()

    sc = ax.scatter(coords_3d[:, 0],
                    coords_3d[:, 1],
                    coords_3d[:, 2],
                    c=y_train_fold0,
                    alpha=0.7, cmap="viridis")

    ax.set_title(f"Epoch {epoch_history_fold0[idx]}")
    ax.set_xlabel("Hidden1")
    ax.set_ylabel("Hidden2")
    ax.set_zlabel("Hidden3")

fig.suptitle("Gradual Separation in Hidden Representation (Fold 0, 3D)", fontsize=16)
# One colorbar for all subplots
cbar = fig.colorbar(sc, ax=axes, shrink=0.6, label="CFR/FFR")
plt.tight_layout()
plt.show()

# # --------------------------
# # 7) (OPTIONAL) WRITE HIDDEN REPRESENTATION OR PREDICTIONS TO CSV
# # --------------------------
# # Let's say you want to write the final hidden representation of the
# # first fold's training data to a CSV file. We'll use the last epoch's hidden layer:
# final_hidden_fold0 = hidden_history_fold0[-1].detach().numpy()  # shape [n_train_fold0, 10]
# df_hidden = pd.DataFrame(final_hidden_fold0, columns=[f"Hidden_{i+1}" for i in range(10)])
# df_hidden["Target"] = y_train_fold0
# df_hidden.to_csv("hidden_representation_fold0.csv", index=False)
#
# print("Wrote hidden_representation_fold0.csv. You can inspect it in e.g. Excel or Python to see exact hidden-layer values.")
