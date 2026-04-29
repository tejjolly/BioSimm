import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

###############################################################################
# 1) LOAD CSV AND PREPARE DATA
###############################################################################
csv_path = "./summary3.csv"
df = pd.read_csv(csv_path)

# 20 feature columns
feature_cols = [
    "Stenosis Percentage", "Length", "Width", "HMR", "HSR",
    "P_Loss_Coeff", "WSS_TE", "WSS_LE", "WSS_TE_Area", "WSS_LE_Area",
    "BMR/HMR", "WSS_Area_Bifur", "WSS_Bif", "WSS_LMB",
    "WSS_min", "WSS_TE_min", "WSS_LE_min", "WSS_TE_Area_min",
    "WSS_LE_Area_min", "WSS_Area_Bifur_min"
]
target_col = "CFR/FFR"

# <-- IMPORTANT: The column that identifies your geometry
geom_col = "Geometry Number"  # adjust if your CSV uses a different name

X = df[feature_cols].values         # shape: [num_samples, 20]
y = df[target_col].values           # shape: [num_samples]
geom_numbers = df[geom_col].values  # shape: [num_samples]

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(-1)  # [N,1] for regression

# We'll keep geom_numbers in a NumPy array, not in the dataset. We'll slice it with fold indices.

###############################################################################
# 2) DEFINE A SIMPLE MODEL WITH DROPOUT
###############################################################################
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 1 output for regression

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        hidden = x.clone()   # We'll treat this as the "hidden representation"
        x = self.dropout(x)
        x = self.fc2(x)      # final linear output
        return x, hidden

###############################################################################
# 3) CROSS-VALIDATION, EARLY STOPPING, WEIGHT DECAY
###############################################################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)

num_epochs = 1000
patience_limit = 10
fold_val_losses = []

fold_index = 0
for train_index, val_index in kf.split(X):
    print(f"\n----- Fold {fold_index+1} / {kf.n_splits} -----")

    # Prepare data subsets
    train_dataset = Subset(TensorDataset(X_tensor, y_tensor), train_index)
    val_dataset   = Subset(TensorDataset(X_tensor, y_tensor), val_index)

    # We'll also slice the geometry array for easy referencing
    geom_train = geom_numbers[train_index]
    geom_val   = geom_numbers[val_index]

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

    # Instantiate new model, optimizer, loss
    model = SimpleRegressor(input_dim=len(feature_cols), hidden_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds, _ = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
        train_loss = running_loss / len(train_dataset)

        # --- VALIDATION ---
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds_val, _ = model(bx)
                loss_val = criterion(preds_val, by)
                val_loss_accum += loss_val.item() * bx.size(0)
        val_loss = val_loss_accum / len(val_dataset)

        # Print progress every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    fold_val_losses.append(best_val_loss)
    print(f"Best Val Loss for fold {fold_index+1}: {best_val_loss:.4f}")

    # -----------------------------
    # 4) FIND OUTLIER ON TRAIN SET
    # -----------------------------
    # Load best model state
    model.load_state_dict(best_model_state)

    # Forward pass (no shuffle) on the entire train subset in a known order
    train_loader_noshuffle = DataLoader(train_dataset, batch_size=1, shuffle=False)
    hidden_list = []
    with torch.no_grad():
        for bx, by in train_loader_noshuffle:
            _, hidden_vec = model(bx)
            hidden_list.append(hidden_vec)
    # hidden_list is a list of [1, hidden_dim] tensors
    hidden_all = torch.cat(hidden_list, dim=0)  # shape [train_size, hidden_dim]

    # Convert to NumPy
    H_np = hidden_all.cpu().numpy()

    # PCA to 2D
    pca_2d = PCA(n_components=2).fit_transform(H_np)
    # We'll define a simple "outlier" as the point farthest from the cluster mean
    center = pca_2d.mean(axis=0)
    dists = np.linalg.norm(pca_2d - center, axis=1)
    max_idx = np.argmax(dists)

    # Retrieve geometry number, plus Stenosis% and Length for that outlier
    # The index in 'geom_train' should align with the order we iterated above.
    outlier_geom = geom_train[max_idx]
    outlier_stenosis = X_tensor[train_index[max_idx], feature_cols.index("Stenosis Percentage")].item()
    outlier_length   = X_tensor[train_index[max_idx], feature_cols.index("Length")].item()

    print(f"Potential outlier in fold {fold_index+1}:")
    print(f"  Geometry Number = {outlier_geom}")
    print(f"  Stenosis% = {outlier_stenosis:.4f}, Length = {outlier_length:.4f}")

    fold_index += 1

###############################################################################
# 5) REPORT CROSS-VALIDATION RESULTS
###############################################################################
print("\n----- Cross-Validation Results -----")
print(f"Fold validation losses: {fold_val_losses}")
print(f"Average validation loss: {np.mean(fold_val_losses):.4f}")
