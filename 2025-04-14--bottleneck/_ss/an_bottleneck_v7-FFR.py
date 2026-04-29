#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottleneck‑net (multiclass) + out‑of‑fold diagnostics  — v7.2
============================================================
* 5‑fold (Stratified) CV on the **training** split
* Per‑class Precision / Sensitivity / Specificity
  → **mean ± std dev** printed after CV
  → heat‑map shows the means
* Training‑ vs‑validation loss curves (CV‑averaged ± 1σ bands)
* Final fit on full‑training data → evaluation on the held‑out **test** set
* Two scatter plots on the P_d/P_a – CFR plane
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ────────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & flags
# ────────────────────────────────────────────────────────────────────────────────
inspect_data        = True
user_bottleneck_dim = 4
user_epochs         = 200
user_batch_size     = 4
user_hidden_dim     = 8
user_lr             = 2e-3
user_cos_max        = user_epochs
n_splits            = 5                      # ← use 5‑fold CV
num_classes         = 1 # 1 'class' for regression
test_size           = 0.4
random_state        = 41
device              = torch.device("cpu")

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREP DATA
# ────────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(
    "/Users/tejjolly/Documents/BioSimm/Simulations/"
    "Post_Processing/data/data.csv"
)
mask = (
    (df_raw["Condition"] == "Hyperemic")
    # df_raw["discord"].notna() &
    # df_raw["P_Loss_Coeff"].notna()
)
df_raw = df_raw[mask]

feature_cols = ["Stenosis Percentage", "Width", "Length"]

df_model = df_raw[feature_cols + ["P_d/P_a"]].dropna()
X_full   = df_model[feature_cols].values.astype(np.float32)
y_full   = df_model["P_d/P_a"].values

print("Unique labels:", np.unique(y_full), "| Total samples:", len(y_full))
if inspect_data:
    df_model.to_csv("data_cleaned.csv", index=False)

# ────────────────────────────────────────────────────────────────────────────────
# 2) TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_full, y_full, df_model.index.values,
    test_size=test_size, random_state=random_state
)
print(f"Train size: {len(idx_train)},  Test size: {len(idx_test)}")
print("Test‑set class counts:", {cls: int(sum(y_test == cls))
                                 for cls in range(num_classes)})

# ────────────────────────────────────────────────────────────────────────────────
# 3)  NETWORK DEFINITION
# ────────────────────────────────────────────────────────────────────────────────
class BottleneckNet(nn.Module):
    def __init__(self, inp, hid, bottleneck, num_classes):
        super().__init__()
        self.hidden     = nn.Linear(inp, hid)
        self.bottleneck = nn.Linear(hid, bottleneck)
        self.classifier = nn.Linear(bottleneck, num_classes)
        self.act        = nn.LeakyReLU()

    def forward(self, x):
        h = self.act(self.hidden(x))
        z = self.act(self.bottleneck(h))
        return self.classifier(z).squeeze(1), z

# ────────────────────────────────────────────────────────────────────────────────
# 4)  HELPERS
# ────────────────────────────────────────────────────────────────────────────────
# def specificity_per_class(cm: np.ndarray) -> np.ndarray:
#     spec = []
#     for i in range(len(cm)):
#         tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
#         fp = cm[:, i].sum() - cm[i, i]
#         spec.append(tn / (tn + fp) if (tn + fp) else 0.)
#     return np.asarray(spec)

def iterate_minibatches(x_t, y_t, bs):
    order = torch.randperm(x_t.size(0))
    for start in range(0, x_t.size(0), bs):
        sl = order[start:start+bs]
        yield x_t[sl], y_t[sl]

# ────────────────────────────────────────────────────────────────────────────────
# 5)  K‑FOLD CV ON TRAINING SPLIT
# ────────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# metric_lists = {"prec": [], "sens": [], "spec": []}
per_fold_train_curves, per_fold_val_curves = [], []
fold_metrics = []
for fold, (tr_id, val_id) in enumerate(skf.split(X_train_s, y_train)):
    print(f"\n── Fold {fold+1}/{n_splits} ──")
    X_tr, y_tr = X_train_s[tr_id], y_train[tr_id]
    X_val, y_val = X_train_s[val_id], y_train[val_id]

    net = BottleneckNet(X_tr.shape[1], user_hidden_dim,
                        user_bottleneck_dim, num_classes).to(device)
    opt = optim.Adam(net.parameters(), lr=user_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=user_cos_max)
    loss_fn = nn.MSELoss()

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32,   device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    train_curve, val_curve = [], []

    for _ in tqdm(range(user_epochs), leave=False):
        # ----- training -----
        net.train()
        running = 0.0
        for xb, yb in iterate_minibatches(X_tr_t, y_tr_t, user_batch_size):
            opt.zero_grad()
            out, _ = net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_curve.append(running / len(X_tr))

        # ----- validation -----
        net.eval()
        with torch.no_grad():
            val_out, _ = net(X_val_t)
            val_loss = loss_fn(val_out, torch.tensor(y_val,
                                                     dtype=torch.long,
                                                     device=device))
        val_curve.append(val_loss.item())
        sched.step()

    per_fold_train_curves.append(np.asarray(train_curve))
    per_fold_val_curves.append(np.asarray(val_curve))

    # # ----- fold metrics -----
    net.eval()
    # pred = net(X_val_t)[0].cpu().numpy()
    pred = net(X_val_t)[0].cpu().detach().numpy()

    # cm   = confusion_matrix(y_val, pred, labels=np.arange(num_classes))
    #
    # metric_lists["mae"].append(
    #     precision_score(y_val, pred, average=None, zero_division=0))
    # metric_lists["sens"].append(
    #     recall_score(   y_val, pred, average=None, zero_division=0))
    # metric_lists["spec"].append(specificity_per_class(cm))
    #
    # print(classification_report(y_val, pred, digits=3))

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    fold_metrics.append([mae, rmse, r2])

# ────────────────────────────────────────────────────────────────────────────────
# 5b)  PRINT MEAN ± STD METRICS ACROSS FOLDS
# ────────────────────────────────────────────────────────────────────────────────
fold_metrics = np.asarray(fold_metrics)    # shape (k, 3)
print(f"\nMAE   : {fold_metrics[:,0].mean():.4f} ± {fold_metrics[:,0].std():.4f}")
print(f"RMSE  : {fold_metrics[:,1].mean():.4f} ± {fold_metrics[:,1].std():.4f}")
print(f"R²    : {fold_metrics[:,2].mean():.4f} ± {fold_metrics[:,2].std():.4f}")


# ────────────────────────────────────────────────────────────────────────────────
# 6)  LOSS CURVES (mean ± 1σ over folds)
# ────────────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, user_epochs + 1)
train_stack = np.vstack(per_fold_train_curves)
val_stack   = np.vstack(per_fold_val_curves)

mean_train, std_train = train_stack.mean(axis=0), train_stack.std(axis=0)
mean_val,   std_val   = val_stack.mean(axis=0),   val_stack.std(axis=0)

plt.figure(figsize=(7, 4))
plt.plot(epochs, mean_train, label="Train loss")
plt.plot(epochs, mean_val,   label="Val loss")
plt.fill_between(epochs, mean_train-std_train, mean_train+std_train,
                 alpha=.15, linewidth=0)
plt.fill_between(epochs, mean_val-std_val, mean_val+std_val,
                 alpha=.15, linewidth=0)
plt.xlabel("Epoch"); plt.ylabel("MSE loss")
plt.title("Training vs Validation loss (CV average)")
plt.legend(); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 7)  HEAT‑MAP OF AVERAGE CV METRICS
# ────────────────────────────────────────────────────────────────────────────────
# metrics_df = pd.DataFrame({
#     "Precision":   mean_prec,
#     "Sensitivity": mean_sens,
#     "Specificity": mean_spec,
# }, index=[f"Class {i}" for i in range(num_classes)]).T
#
# plt.figure(figsize=(8, 4))
# sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="Reds_r",
#             linewidths=.5, linecolor="grey")
# plt.title("Average Per‑Class Metrics (CV, training data)")
# plt.ylabel("Metric"); plt.xlabel("Class"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 8)  FINAL MODEL → TEST SET
# ────────────────────────────────────────────────────────────────────────────────
X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Y_tr_t = torch.tensor(y_train,   dtype=torch.float32,   device=device)
X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)

final_net = BottleneckNet(
    X_train.shape[1], user_hidden_dim, user_bottleneck_dim, num_classes).to(device)
opt_final = optim.Adam(final_net.parameters(), lr=user_lr)
crit      = nn.MSELoss()

for _ in tqdm(range(user_epochs), desc="Final model", leave=False):
    final_net.train()
    for xb, yb in iterate_minibatches(X_tr_t, Y_tr_t, user_batch_size):
        opt_final.zero_grad()
        out, _ = final_net(xb)
        loss = crit(out, yb)
        loss.backward()
        opt_final.step()

final_net.eval()
y_pred_test = final_net(X_te_t)[0].cpu().detach().numpy()
print("\nTest-set MAE :", mean_absolute_error(y_test, y_pred_test))
print("Test-set RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Test-set R²  :", r2_score(y_test, y_pred_test))

plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'k--', lw=1)
plt.xlabel("True FFR"); plt.ylabel("Predicted FFR")
plt.title("Test-set parity plot"); plt.tight_layout(); plt.show()
