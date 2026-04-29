import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------
# 1) LOAD DATA
# ----------------------------------------------------------------
df = pd.read_csv("summary_imp_comb_garcia_v3.csv")

drop_cols = [
    "CFR","P_d/P_a","CFR/FFR","Condition",
    "Geometry Number","R_scale","R_micro",
    "R_total","Location","source"
]
df = df.drop(columns=drop_cols)

# For demonstration, user selected these columns:
features = ["HMR","P_Loss_Coeff","BMR/HMR","discord"]
df = df[features]

target_col = "discord"
feature_cols = [c for c in df.columns if c != target_col]

print("Data shape:", df.shape)
print("Columns:", df.columns)
print("Feature columns:", feature_cols)
print(df[feature_cols].describe())
print(df[feature_cols].isna().sum())

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values  # ensure these are integer-coded [0,1] or [-1, +1] for classification

# ----------------------------------------------------------------
# 2) DEFINE MODEL
# ----------------------------------------------------------------
class BottleneckNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes=2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # Hidden
        h = self.relu(self.hidden(x))
        # Bottleneck
        z = self.relu(self.bottleneck(h))
        # Output
        out = self.classifier(z)
        return out, z

# ----------------------------------------------------------------
# 3) TRAINING FUNCTION FOR ONE FOLD
# ----------------------------------------------------------------
def train_one_fold(X_train_fold, y_train_fold,
                   X_val_fold,   y_val_fold,
                   epochs=3000,
                   batch_size=8,
                   hidden_dim=3,
                   bottleneck_dim=1,
                   lr=1e-3):
    """
    Trains BottleneckNet on one fold's (X_train_fold, y_train_fold),
    then evaluates on (X_val_fold, y_val_fold).
    Returns: final accuracy on the validation set, plus the trained model.
    """

    # Convert to torch tensors
    X_train_torch = torch.from_numpy(X_train_fold)
    y_train_torch = torch.from_numpy(y_train_fold).long()

    X_val_torch   = torch.from_numpy(X_val_fold)
    y_val_torch   = torch.from_numpy(y_val_fold).long()

    input_dim = X_train_fold.shape[1]
    model_fold = BottleneckNet(input_dim, hidden_dim, bottleneck_dim)
    optimizer = optim.Adam(model_fold.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=1, eta_min=1e-7
    )

    criterion = nn.CrossEntropyLoss()

    # Simple minibatch iterator
    def iterate_minibatches(X_t, y_t, batch_sz):
        idx = torch.randperm(X_t.size(0))
        for start_idx in range(0, X_t.size(0), batch_sz):
            excerpt = idx[start_idx:start_idx+batch_sz]
            yield X_t[excerpt], y_t[excerpt]

    for epoch in range(epochs):
        model_fold.train()
        running_loss = 0.0

        for Xb, yb in iterate_minibatches(X_train_torch, y_train_torch, batch_size):
            optimizer.zero_grad()
            outputs, _ = model_fold(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        epoch_loss = running_loss / X_train_torch.size(0)

        # (Optional) track val accuracy each epoch
        model_fold.eval()
        with torch.no_grad():
            val_outputs, _ = model_fold(X_val_torch)
            _, val_preds = torch.max(val_outputs, dim=1)
            val_acc = (val_preds == y_val_torch).float().mean().item()

        # Print progress every ~100 epochs if desired:
        # if (epoch+1) % 100 == 0:
        #     print(f"[Epoch {epoch+1}/{epochs}] "
        #           f"Train Loss: {epoch_loss:.4f} "
        #           f"Val Acc: {val_acc:.2f}")

        scheduler.step()

    # Final val accuracy
    model_fold.eval()
    with torch.no_grad():
        val_outputs, _ = model_fold(X_val_torch)
        _, val_preds = torch.max(val_outputs, dim=1)
        val_acc = (val_preds == y_val_torch).float().mean().item()

    return val_acc, model_fold

# ----------------------------------------------------------------
# 4) K-FOLD CROSS VALIDATION
# ----------------------------------------------------------------
from sklearn.model_selection import KFold

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n========== Fold {fold+1} / {n_splits} ==========")

    # Split into train/val subsets for this fold
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Scale based on the fold's training set
    scaler_fold = StandardScaler()
    X_tr_scaled = scaler_fold.fit_transform(X_tr)
    X_val_scaled = scaler_fold.transform(X_val)

    # Train
    fold_acc, fold_model = train_one_fold(
        X_train_fold=X_tr_scaled,
        y_train_fold=y_tr,
        X_val_fold=X_val_scaled,
        y_val_fold=y_val,
        epochs=300,            # or your chosen # of epochs
        batch_size=8,
        hidden_dim=16,
        bottleneck_dim=1,
        lr=1e-3
    )

    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")
    fold_accuracies.append(fold_acc)

# ----------------------------------------------------------------
# 5) AVERAGE FOLD RESULTS
# ----------------------------------------------------------------
mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies)
print("\nCross-Validation Results:")
print(f"Fold Accuracies: {fold_accuracies}")
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
