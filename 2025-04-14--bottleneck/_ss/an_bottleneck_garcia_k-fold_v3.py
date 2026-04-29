# Stratifiying k folds
# Switching to just cosine annealing, no restarts

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# User inputs
user_epoch_input = 1500
user_batch_size = 8
user_hidden_dim = 6
user_lr = 1e-3
user_cos_max = user_epoch_input
n_runs = 3
n_splits = 3

# ----------------------------------------------------------------
# 1) LOAD DATA
# ----------------------------------------------------------------
df = pd.read_csv("summary_imp_comb_garcia_v3.csv")

# Drop columns that imply the label
drop_cols = [
    "CFR","P_d/P_a","CFR/FFR","Condition",
    "Geometry Number","R_scale","R_micro",
    "R_total","Location","source"
]
df = df.drop(columns=drop_cols)

features = ["HMR", "P_Loss_Coeff", "BMR/HMR", "discord"]
df = df[features]

target_col = "discord"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values

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
        h = self.relu(self.hidden(x))
        z = self.relu(self.bottleneck(h))
        out = self.classifier(z)
        return out, z

# ----------------------------------------------------------------
# 3) TRAINING FUNCTION
# ----------------------------------------------------------------
def train_one_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                   X_train_df, epochs=user_epoch_input, batch_size=user_batch_size,
                   hidden_dim=user_hidden_dim, bottleneck_dim=1, lr=user_lr):

    X_train_torch = torch.from_numpy(X_train_fold)
    y_train_torch = torch.from_numpy(y_train_fold).long()
    X_val_torch = torch.from_numpy(X_val_fold)
    y_val_torch = torch.from_numpy(y_val_fold).long()

    input_dim = X_train_fold.shape[1]
    model = BottleneckNet(input_dim, hidden_dim, bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=user_cos_max, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    def iterate_minibatches(X_t, y_t, batch_sz):
        idx = torch.randperm(X_t.size(0))
        for start_idx in range(0, X_t.size(0), batch_sz):
            excerpt = idx[start_idx:start_idx+batch_sz]
            yield X_t[excerpt], y_t[excerpt]

    for epoch in range(epochs):
        model.train()
        for Xb, yb in iterate_minibatches(X_train_torch, y_train_torch, batch_size):
            optimizer.zero_grad()
            outputs, _ = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs, val_bottleneck = model(X_val_torch)
        _, val_preds = torch.max(val_outputs, dim=1)
        val_acc = (val_preds == y_val_torch).float().mean().item()

        _, train_bottleneck = model(X_train_torch)
        train_bottleneck_np = train_bottleneck.numpy().flatten()

    df_bottleneck = pd.DataFrame({"z": train_bottleneck_np}).reset_index(drop=True)
    df_combined = pd.concat([X_train_df.reset_index(drop=True), df_bottleneck], axis=1)
    corr_matrix = df_combined.corr()
    z_corrs = corr_matrix["z"][feature_cols].values.tolist()

    return val_acc, model, z_corrs

# ----------------------------------------------------------------
# 4) CROSS VALIDATION + CORRELATION COLLECTION OVER MULTIPLE RUNS
# ----------------------------------------------------------------
all_run_accuracies = []
all_run_correlations = []

for run in range(n_runs):
    print(f"\n================ Run {run+1} / {n_runs} ================")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + run)

    fold_accuracies = []
    fold_correlations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== Fold {fold+1} / {n_splits} ==========")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        df_train_features = df[feature_cols].iloc[train_idx, :].copy()

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        acc, model, corr_vec = train_one_fold(
            X_tr_scaled, y_tr, X_val_scaled, y_val, df_train_features,
            epochs=user_epoch_input, batch_size=user_batch_size, hidden_dim=user_hidden_dim, bottleneck_dim=1, lr=user_lr
        )

        fold_accuracies.append(acc)
        fold_correlations.append(corr_vec)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    all_run_accuracies.append(np.mean(fold_accuracies))
    all_run_correlations.append(np.mean(fold_correlations, axis=0))

# ----------------------------------------------------------------
# 5) RESULTS SUMMARY
# ----------------------------------------------------------------
print("\n================ Overall Summary ================")
print(f"Average CV Accuracy Across Runs: {np.mean(all_run_accuracies):.4f} ± {np.std(all_run_accuracies):.4f}")

avg_corr_df = pd.DataFrame(all_run_correlations, columns=feature_cols)
avg_corr_df.loc["MeanAcrossRuns"] = avg_corr_df.mean(axis=0)
print("\nAverage Correlation of z with Features Across Runs:")
print(avg_corr_df.round(3))
