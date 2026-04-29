#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottleneck‑net (multiclass) + out‑of‑fold diagnostics  — v7.1
============================================================
* Train/test split → 5‑fold CV on the **train** portion
* Heat‑map of mean per‑class metrics (precision / sensitivity / specificity)
* Final fit on full training data → evaluation on the **held‑out test set**
* Scatter plots of **true** vs **NN‑predicted** labels on the P_d/P_a‑CFR plane
  now fetch P_d/P_a & CFR directly from the original dataframe so they’re
  available even though they’re not used as features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & flags
# ────────────────────────────────────────────────────────────────────────────────
inspect_data        = True
user_bottleneck_dim = 2
user_epochs         = 600
user_batch_size     = 8
user_hidden_dim     = 16
user_lr             = 2e-3
user_cos_max        = user_epochs
n_splits            = 2
num_classes         = 4

test_size           = 0.333
random_state        = 42

device              = torch.device("cpu")

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREP DATA
# ────────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data.csv")
mask = (
    (df_raw["Condition"] == "Hyperemic") &
    df_raw["discord"].notna() &
    df_raw["P_Loss_Coeff"].notna()
)
df_raw = df_raw[mask]

feature_cols = ["HMR", "P_Loss_Coeff", "BMR/HMR"]
# keep only features + target for model input, but preserve original for plots

df_model = df_raw[feature_cols + ["discord"]].dropna()
X_full   = df_model[feature_cols].values.astype(np.float32)
y_full   = df_model["discord"].astype(int).values

print("Unique labels:", np.unique(y_full), "| Total samples:", len(y_full))

if inspect_data:
    df_model.to_csv("data_cleaned.csv", index=False)

# ────────────────────────────────────────────────────────────────────────────────
# 2) TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_full, y_full, df_model.index.values,
    test_size=test_size, stratify=y_full, random_state=random_state)

print(f"Train size: {len(idx_train)},  Test size: {len(idx_test)}")
test_counts = {cls: int(sum(y_test == cls)) for cls in range(num_classes)}
print("\nTest‑set class counts (size =", len(y_test), "):")
print(test_counts)

# ────────────────────────────────────────────────────────────────────────────────
# 3)  NETWORK DEFINITION
# ────────────────────────────────────────────────────────────────────────────────
class BottleneckNet(nn.Module):
    def __init__(self, inp, hid, bottleneck, num_classes):
        super().__init__()
        self.hidden     = nn.Linear(inp, hid)
        self.bottleneck = nn.Linear(hid, bottleneck)
        self.classifier = nn.Linear(bottleneck, num_classes)
        self.act        = nn.Tanh()

    def forward(self, x):
        h = self.act(self.hidden(x))
        z = self.act(self.bottleneck(h))
        return self.classifier(z), z

# ────────────────────────────────────────────────────────────────────────────────
# 4)  HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.)
    return np.asarray(spec)


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

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

metric_lists = {"prec": [], "sens": [], "spec": []}

# ── add right before the CV loop ───────────────────────────────────────────────
per_fold_train_curves = []   # list of 1‑D arrays, length = user_epochs
per_fold_val_curves   = []   # same, but validation loss


for fold, (tr_id, val_id) in enumerate(skf.split(X_train_s, y_train)):
    print(f"\n── Fold {fold+1}/{n_splits} ──")
    X_tr, y_tr = X_train_s[tr_id], y_train[tr_id]
    X_val, y_val = X_train_s[val_id], y_train[val_id]

    counts = {cls: int(sum(y_val == cls)) for cls in range(num_classes)}
    print(f"  Fold {fold}:", counts)

    net = BottleneckNet(X_tr.shape[1], user_hidden_dim,
                        user_bottleneck_dim, num_classes).to(device)
    opt = optim.Adam(net.parameters(), lr=user_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=user_cos_max)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long,   device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    train_curve = []     # losses for this fold
    val_curve   = []

    for _ in tqdm(range(user_epochs), leave=False):
        # ---- training ----
        net.train()
        running = 0.0
        for xb, yb in iterate_minibatches(X_tr_t, y_tr_t, user_batch_size):
            opt.zero_grad()
            out, _ = net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_curve.append(running / len(X_tr))  # epoch‑avg train loss

        # ---- validation ----
        net.eval()
        with torch.no_grad():
            val_out, _ = net(X_val_t)
            val_loss = loss_fn(val_out, torch.tensor(y_val,
                                                     dtype=torch.long,
                                                     device=device))
        val_curve.append(val_loss.item())

        sched.step()

    per_fold_train_curves.append(np.asarray(train_curve))
    per_fold_val_curves  .append(np.asarray(val_curve))

    net.eval(); pred = net(X_val_t)[0].argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(y_val, pred, labels=np.arange(num_classes))

    metric_lists["prec"].append(precision_score(y_val, pred, average=None, zero_division=0))
    metric_lists["sens"].append(recall_score(   y_val, pred, average=None, zero_division=0))
    metric_lists["spec"].append(specificity_per_class(cm))

    print(classification_report(y_val, pred, digits=3))
# ────────────────────────────────────────────────────────────────────────────────
# 6b)  TRAIN vs VALIDATION LOSS CURVES (averaged over the CV folds)
# ────────────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, user_epochs + 1)

train_stack = np.vstack(per_fold_train_curves)   # shape: (n_splits, epochs)
val_stack   = np.vstack(per_fold_val_curves)

mean_train = train_stack.mean(axis=0)
mean_val   = val_stack.mean(axis=0)
std_train  = train_stack.std(axis=0)
std_val    = val_stack.std(axis=0)

plt.figure(figsize=(7,4))
plt.plot(epochs, mean_train, label="Train loss")
plt.plot(epochs, mean_val,   label="Val loss")
# 95 % band (±2 × std) – optional
plt.fill_between(epochs, mean_train-std_train, mean_train+std_train,
                 alpha=.15, linewidth=0)
plt.fill_between(epochs, mean_val-std_val, mean_val+std_val,
                 alpha=.15, linewidth=0)
plt.xlabel("Epoch"); plt.ylabel("Cross‑entropy loss")
plt.title("Training vs Validation loss (CV average)")
plt.legend(); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 6)  HEAT‑MAP OF CV METRICS
# ────────────────────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    "Precision":   np.mean(metric_lists["prec"], axis=0),
    "Sensitivity": np.mean(metric_lists["sens"], axis=0),
    "Specificity": np.mean(metric_lists["spec"], axis=0),
}, index=[f"Class {i}" for i in range(num_classes)]).T

plt.figure(figsize=(8,4))
sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="Reds_r",
            linewidths=.5, linecolor="grey")
plt.title("Average Per‑Class Metrics (CV, training data)")
plt.ylabel("Metric"); plt.xlabel("Class"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 7)  FINAL MODEL → TEST SET
# ────────────────────────────────────────────────────────────────────────────────
X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Y_tr_t = torch.tensor(y_train,   dtype=torch.long,   device=device)
X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)

final_net = BottleneckNet(X_train.shape[1], user_hidden_dim,
                          user_bottleneck_dim, num_classes).to(device)
opt = optim.Adam(final_net.parameters(), lr=user_lr)
crit = nn.CrossEntropyLoss()

for _ in tqdm(range(user_epochs), desc="Final model", leave=False):
    final_net.train()
    for xb, yb in iterate_minibatches(X_tr_t, Y_tr_t, user_batch_size):
        opt.zero_grad(); out, _ = final_net(xb); loss = crit(out, yb);
        loss.backward(); opt.step()

final_net.eval(); y_pred_test = final_net(X_te_t)[0].argmax(dim=1).cpu().numpy()
print("\nTest set classification report:\n", classification_report(y_test, y_pred_test, digits=3))

cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5,4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
plt.xlabel("Predicted"); plt.ylabel("Actual");
plt.title("Confusion Matrix – Test set"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 8)  TWO SCATTER PLOTS:
#     (a) P_d/P_a‑CFR coloured by **true** class
#     (b) same points coloured by **predicted** class
#     triangles (^)  → mis‑predictions in either view
# ────────────────────────────────────────────────────────────────────────────────
if {"P_d/P_a", "CFR"}.issubset(df_raw.columns):
    # Build once → reuse for both figures
    df_plot = df_raw.loc[idx_test, ["P_d/P_a", "CFR", "discord"]].copy()
    df_plot["pred"]     = y_pred_test
    df_plot["mispred"]  = df_plot["pred"] != df_plot["discord"]

    def make_scatter(ax, hue_col, title):
        """
        hue_col : "discord"  (true labels)  or "pred" (network predictions)
        """
        # correct → circles
        sns.scatterplot(
            data=df_plot[~df_plot["mispred"]],
            x="P_d/P_a", y="CFR", hue=hue_col,
            palette="coolwarm", marker="o", alpha=.5, s=60,
            edgecolor=None, legend=False, ax=ax
        )
        # mis‑pred → triangles
        sns.scatterplot(
            data=df_plot[df_plot["mispred"]],
            x="P_d/P_a", y="CFR", hue=hue_col,
            palette="coolwarm", marker="^", alpha=1, s=70,
            edgecolor="k", legend=True, ax=ax
        )

        ax.axhline(2.0, ls="--", c="gray", lw=1)
        ax.axvline(0.8, ls="--", c="gray", lw=1)
        ax.set_xlabel("FFR (P_d/P_a)")
        ax.set_ylabel("CFR")
        ax.set_title(title)
        ax.grid(False)

    # (a) TRUE‑label view
    fig1, ax1 = plt.subplots(figsize=(6,4))
    make_scatter(ax1, "discord", "Test set • coloured by **true** class")
    plt.tight_layout(); plt.show()

    # (b) PREDICTED‑label view
    fig2, ax2 = plt.subplots(figsize=(6,4))
    make_scatter(ax2, "pred", "Test set • coloured by **predicted** class")
    plt.tight_layout(); plt.show()
else:
    print("\nP_d/P_a and/or CFR not in dataframe – scatter plots skipped.")

# ────────────────────────────────────────────────────────────────────────────────
# 9)  (optional) SAVE ARRAYS / EMBEDDINGS
# ────────────────────────────────────────────────────────────────────────────────
# np.save("X_train_s.npy", X_train_s); np.save("X_test_s.npy", X_test_s)
