#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottleneck‑net (multiclass) + out‑of‑fold diagnostics  —NBSPv7.2
============================================================
* 5‑fold (Stratified) CV on the **training** split
* Per‑class Precision / Sensitivity / Specificity
  → **mean ± std dev** printed after CV
  → heat‑map shows the means
* Training‑ vs‑validation loss curves (CV‑averaged ± 1σ bands)
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    classification_report
)
from scipy.stats import chi2_contingency
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
# ────────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & flags
# ────────────────────────────────────────────────────────────────────────────────
inspect_data        = True #outputs .csv of data
train_model         = False
p_val_flag          = False
user_bottleneck_dim = 3
user_epochs         = 100
user_batch_size     = 8
user_hidden_dim     = 8
user_lr             = 2e-3
user_cos_max        = user_epochs
n_splits            = 5                      # ← use 5‑fold CV
num_classes         = 4
test_size           = 0.333
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
    (df_raw["Condition"] == "Hyperemic") &
    df_raw["discord"].notna() &
    df_raw["P_Loss_Coeff"].notna()
)
df_raw = df_raw[mask]

feature_cols = ["HMR", "P_Loss_Coeff", "BMR/HMR"]

plt.figure(figsize=(10, 4))

for i, col in enumerate(feature_cols):
    mean = df_raw[col].mean()
    std  = df_raw[col].std()
    median = df_raw[col].median()

    plt.subplot(1, 3, i + 1)
    sns.histplot(df_raw[col], kde=True, bins=25)
    plt.axvline(median, color="red", linestyle="--", linewidth=1.2, label="Median")
    plt.axvline(mean - 1 * std, color='k', linestyle=":", linewidth=2, label="±1 SD" if i == 0 else "")
    plt.axvline(mean + 1 * std, color='k', linestyle=":", linewidth=2)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("Frequency" if i == 0 else "")
    plt.grid(False)

plt.tight_layout()
plt.show()

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
    test_size=test_size, stratify=y_full, random_state=random_state
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

metric_lists = {"prec": [], "sens": [], "spec": [], "p_val": []}
per_fold_train_curves, per_fold_val_curves = [], []

for fold, (tr_id, val_id) in enumerate(skf.split(X_train_s, y_train)):
    print(f"\n── Fold {fold+1}/{n_splits} ──")
    X_tr, y_tr = X_train_s[tr_id], y_train[tr_id]
    X_val, y_val = X_train_s[val_id], y_train[val_id]

    net = BottleneckNet(X_tr.shape[1], user_hidden_dim,
                        user_bottleneck_dim, num_classes).to(device)
    opt = optim.Adam(net.parameters(), lr=user_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=user_cos_max)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long,   device=device)
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

    # ----- fold metrics -----
    net.eval()
    pred = net(X_val_t)[0].argmax(dim=1).cpu().numpy()
    cm   = confusion_matrix(y_val, pred, labels=np.arange(num_classes))
    # chi2, p_val, _, _ = chi2_contingency(cm)

    metric_lists["prec"].append(
        precision_score(y_val, pred, average=None, zero_division=0))
    metric_lists["sens"].append(
        recall_score(   y_val, pred, average=None, zero_division=0))
    metric_lists["spec"].append(specificity_per_class(cm))
    # metric_lists["p_val"].append(p_val)


    print(classification_report(y_val, pred, digits=3))

# ────────────────────────────────────────────────────────────────────────────────
# 5b)  PRINT MEAN ± STD METRICS ACROSS FOLDS
# ────────────────────────────────────────────────────────────────────────────────
prec_arr = np.vstack(metric_lists["prec"])
sens_arr = np.vstack(metric_lists["sens"])
spec_arr = np.vstack(metric_lists["spec"])
# p_val_arr = np.vstack(metric_lists["p_val"])


mean_prec, std_prec = prec_arr.mean(axis=0), prec_arr.std(axis=0)
mean_sens, std_sens = sens_arr.mean(axis=0), sens_arr.std(axis=0)
mean_spec, std_spec = spec_arr.mean(axis=0), spec_arr.std(axis=0)

print("\nAverage Precision per class (CV):")
for c, (m, s) in enumerate(zip(mean_prec, std_prec)):
    print(f"  Class {c}: {m:.3f} ± {s:.3f}")

print("\nAverage Sensitivity (Recall) per class (CV):")
for c, (m, s) in enumerate(zip(mean_sens, std_sens)):
    print(f"  Class {c}: {m:.3f} ± {s:.3f}")

print("\nAverage Specificity per class (CV):")
for c, (m, s) in enumerate(zip(mean_spec, std_spec)):
    print(f"  Class {c}: {m:.3f} ± {s:.3f}")

# print(f"\nChi-squared p-value across folds: {p_val_arr.mean()} ± {p_val_arr.std()}")

# ────────────────────────────────────────────────────────────────────────────────
# 6)  LOSS CURVES (mean ± 1σ over folds)
# ────────────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, user_epochs + 1)
train_stack = np.vstack(per_fold_train_curves)
val_stack   = np.vstack(per_fold_val_curves)

mean_train, std_train = train_stack.mean(axis=0), train_stack.std(axis=0)
mean_val,   std_val   = val_stack.mean(axis=0),   val_stack.std(axis=0)

plt.figure(figsize=(7, 4))
plt.plot(epochs, mean_train, label="Train loss")
plt.plot(epochs, mean_val,   label="Val loss")
plt.fill_between(epochs, mean_train-std_train, mean_train+std_train,
                 alpha=.15, linewidth=0)
plt.fill_between(epochs, mean_val-std_val, mean_val+std_val,
                 alpha=.15, linewidth=0)
plt.xlabel("Epoch"); plt.ylabel("Cross‑entropy loss")
plt.title("Training vs Validation loss (CV average)")
plt.legend(); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 7)  HEAT‑MAP OF AVERAGE CV METRICS
# ────────────────────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    "Precision":   mean_prec,
    "Sensitivity": mean_sens,
    "Specificity": mean_spec,
}, index=[f"Class {i}" for i in range(num_classes)]).T

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="Reds_r",
            linewidths=.5, linecolor="grey")
plt.title("Average Per‑Class Metrics (CV, training data)")
plt.ylabel("Metric"); plt.xlabel("Class"); plt.tight_layout(); plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 8)  FINAL MODEL → TEST SET
# ────────────────────────────────────────────────────────────────────────────────
X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Y_tr_t = torch.tensor(y_train,   dtype=torch.long,   device=device)
X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)

final_net = BottleneckNet(
    X_train.shape[1], user_hidden_dim, user_bottleneck_dim, num_classes).to(device)
opt_final = optim.Adam(final_net.parameters(), lr=user_lr)
crit      = nn.CrossEntropyLoss()

for _ in tqdm(range(user_epochs), desc="Final model", leave=False):
    final_net.train()
    for xb, yb in iterate_minibatches(X_tr_t, Y_tr_t, user_batch_size):
        opt_final.zero_grad()
        out, _ = final_net(xb)
        loss = crit(out, yb)
        loss.backward()
        opt_final.step()

torch.save(final_net.state_dict(), "final_model.pt") # saving model

final_net.eval()
y_pred_test = final_net(X_te_t)[0].argmax(dim=1).cpu().numpy()
print("\nTest set classification report:\n",
      classification_report(y_test, y_pred_test, digits=3))

cm_test = confusion_matrix(y_test, y_pred_test)
if p_val_flag:
    chi2, p_val, _, _ = chi2_contingency(cm_test)
    print(f"Test set chi2 p-value: {p_val}")
plt.figure(figsize=(5, 4))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.arange(num_classes),
            yticklabels=np.arange(num_classes))
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix – Test set")
plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# 9)  SINGLE SCATTER PLOT – FFR vs CFR COLOURED BY PREDICTED CLASS
# ────────────────────────────────────────────────────────────────────────────────
# Custom circular legend markers
correct_circle = Line2D([0], [0], marker='o', color='w',
                        label='Correctly predicted',
                        markerfacecolor='lightgray', markersize=8, alpha=0.6)

wrong_circle = Line2D([0], [0], marker='o', color='k', linestyle='None',
                      label='Incorrectly predicted',
                      markerfacecolor='lightgray', markeredgewidth=0.8,
                      markersize=8, alpha=1.0)
if {"P_d/P_a", "CFR"}.issubset(df_raw.columns):
    df_plot = df_raw.loc[idx_test, ["P_d/P_a", "CFR", "discord"]].copy()
    df_plot["pred"]    = y_pred_test
    df_plot["mispred"] = df_plot["pred"] != df_plot["discord"]   # <- drop if not needed

    plt.figure(figsize=(6, 4))

    # correct predictions
    # Plot correct predictions (no edgecolor)
    sns.scatterplot(data=df_plot[~df_plot["mispred"]],
                    x="P_d/P_a", y="CFR",
                    hue="pred", palette="RdYlGn_r",
                    marker="o", s=60, alpha=0.6, edgecolor=None, legend=False)

    # Plot mispredictions (with black edgecolor)
    sns.scatterplot(data=df_plot[df_plot["mispred"]],
                    x="P_d/P_a", y="CFR",
                    hue="pred", palette="RdYlGn_r",
                    marker="o", s=60, alpha=1,
                    edgecolor="k", linewidth=0.9, legend=False)


    # Add the full legend with class labels + marker explanations
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([correct_circle, wrong_circle] + handles,
               ["Correctly predicted", "Incorrectly predicted"] + labels)
               # title="Prediction Legend")

    plt.axhline(2.0, ls="--", c="gray", lw=1)
    plt.axvline(0.8, ls="--", c="gray", lw=1)
    plt.xlabel("FFR (P_d / P_a)")
    plt.ylabel("CFR")
    plt.title("Test set predictions")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 9b)  MONTE-CARLO DECISION-BOUNDARY VISUALISATION
# ────────────────────────────────────────────────────────────────────────────────
# Choose the 2-D slice
feat_x, feat_y   = "BMR/HMR", "P_Loss_Coeff"        # axes you want to see
fixed_feat       = ("HMR", np.median(X_full[:, feature_cols.index("HMR")]))

# 1) Build sampling box (slightly wider than data range)
mins = X_full[:, [feature_cols.index(feat_x),
                   feature_cols.index(feat_y)]].min(axis=0)
maxs = X_full[:, [feature_cols.index(feat_x),
                   feature_cols.index(feat_y)]].max(axis=0)
pad  = 0.05 * (maxs - mins)
mins, maxs = mins - pad, maxs + pad

print(f'mins: {mins}')
print(f'maxs: {maxs}')
eps = 0.2  # or try 0.05
h_idx = feature_cols.index("HMR")
mask = np.abs(X_full[:, h_idx] - fixed_feat[1]) < eps

# 2) Draw Monte-Carlo points
N_mc = 40000
rng  = np.random.default_rng(0)
mc_xy   = rng.uniform(mins, maxs, size=(N_mc, 2))          # (N,2)
mc_full = np.zeros((N_mc, len(feature_cols)), dtype=np.float32)
mc_full[:, feature_cols.index(feat_x)]  = mc_xy[:, 0]
mc_full[:, feature_cols.index(feat_y)]  = mc_xy[:, 1]
mc_full[:, feature_cols.index(fixed_feat[0])] = fixed_feat[1]

# 3) Scale & predict
scaler = StandardScaler()
X_full_s = scaler.fit_transform(X_full)
mc_full_s = scaler.transform(mc_full)
final_net = BottleneckNet(
    X_full.shape[1], user_hidden_dim, user_bottleneck_dim, num_classes).to(device)
final_net.load_state_dict(torch.load("final_model.pt", map_location=device))
with torch.no_grad():
    probs, _ = final_net(torch.tensor(mc_full_s, dtype=torch.float32))
mc_pred = probs.argmax(dim=1).cpu().numpy()

# 4) Plot
plt.figure(figsize=(6,5))
# light dots define decision regions
plt.scatter(mc_xy[:,0], mc_xy[:,1], c=mc_pred,
            cmap="RdYlGn_r", alpha=0.15, s=5, linewidths=0)

# overlay real training points for reference
sns.scatterplot(x=X_full[mask, feature_cols.index(feat_x)],
                y=X_full[mask, feature_cols.index(feat_y)],
                hue=y_full[mask], palette="RdYlGn_r", edgecolor="k",
                linewidth=0.3, s=45, legend=True) # Actual class labels being plotted

plt.xlabel(feat_x); plt.ylabel(feat_y)
plt.title(f"Monte-Carlo decision map\n(slice at {fixed_feat[0]} = {fixed_feat[1]:.3g} +/- {eps})")
plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# 9c)  MONTE-CARLO DECISION-BOUNDARY VISUALISATION
# ────────────────────────────────────────────────────────────────────────────────

feat_x = "P_Loss_Coeff"
feat_y = "BMR/HMR"
sweep_feat = "HMR"

# Define sweep range
sweep_vals = np.linspace(
    X_full[:, feature_cols.index(sweep_feat)].min(),
    # X_full[:, feature_cols.index(sweep_feat)].max(),
    5, # Max value of HMR to be swept
    num = 10  # number of slices
)

for val in sweep_vals:
    # Build grid of inputs
    mins = X_full[:, [feature_cols.index(feat_x), feature_cols.index(feat_y)]].min(axis=0)
    maxs = X_full[:, [feature_cols.index(feat_x), feature_cols.index(feat_y)]].max(axis=0)
    pad  = 0.05 * (maxs - mins)
    mins, maxs = mins - pad, maxs + pad

    N_mc = 40000
    rng  = np.random.default_rng(0)
    mc_xy = rng.uniform(mins, maxs, size=(N_mc, 2))
    mc_full = np.zeros((N_mc, len(feature_cols)), dtype=np.float32)
    mc_full[:, feature_cols.index(feat_x)] = mc_xy[:, 0]
    mc_full[:, feature_cols.index(feat_y)] = mc_xy[:, 1]
    mc_full[:, feature_cols.index(sweep_feat)] = val

    mc_full_s = scaler.transform(mc_full)
    with torch.no_grad():
        probs, _ = final_net(torch.tensor(mc_full_s, dtype=torch.float32))
    mc_pred = probs.argmax(dim=1).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(mc_xy[:, 0], mc_xy[:, 1], c=mc_pred,
                cmap="RdYlGn_r", alpha=0.15, s=5, linewidths=0)

    # Overlay training points where HMR ~ val ± ε
    eps = 0.2  # tolerance for overlay
    mask = np.abs(X_full[:, feature_cols.index(sweep_feat)] - val) < eps
    sns.scatterplot(x=X_full[mask, feature_cols.index(feat_x)],
                    y=X_full[mask, feature_cols.index(feat_y)],
                    hue=y_full[mask], palette="RdYlGn_r",
                    edgecolor="k", linewidth=0.3, s=45, legend=False)

    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.title(f"Slice at {sweep_feat} = {val:.2f}")
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 9d)  3-D Visualization
# ────────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────
# Feature setup
feat_x, feat_y, feat_z = "BMR/HMR", "P_Loss_Coeff", "HMR"
x_idx, y_idx, z_idx = [feature_cols.index(f) for f in (feat_x, feat_y, feat_z)]
num_classes = 4

# ──────────────────────────────────────────────────────
# Monte Carlo sampling in 3D
N_mc = 1000000
rng = np.random.default_rng(0)

mins = X_full[:, [x_idx, y_idx, z_idx]].min(axis=0)
maxs = X_full[:, [x_idx, y_idx, z_idx]].max(axis=0)
pad = 0.05 * (maxs - mins)
mins, maxs = mins - pad, maxs + pad

maxs[2] = 5 #set HMR max to be 5

mc_xyz = rng.uniform(mins, maxs, size=(N_mc, 3))  # (N,3)

mc_full = np.zeros((N_mc, len(feature_cols)), dtype=np.float32)
mc_full[:, x_idx] = mc_xyz[:, 0]
mc_full[:, y_idx] = mc_xyz[:, 1]
mc_full[:, z_idx] = mc_xyz[:, 2]

scaler = StandardScaler()
scaler.fit(X_full)
mc_full_scaled = scaler.transform(mc_full)

final_net = BottleneckNet(X_full.shape[1], user_hidden_dim, user_bottleneck_dim, num_classes).to(device)
final_net.load_state_dict(torch.load("final_model.pt", map_location=device))
final_net.eval()

with torch.no_grad():
    probs, _ = final_net(torch.tensor(mc_full_scaled, dtype=torch.float32))
mc_pred = probs.argmax(dim=1).cpu().numpy()

# ──────────────────────────────────────────────────────
# 3D Scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(mc_xyz[:, 0], mc_xyz[:, 1], mc_xyz[:, 2],
                c=mc_pred, cmap="RdYlGn_r", alpha=1, s=3, linewidth=0)

ax.set_xlabel(feat_x)
ax.set_ylabel(feat_y)
ax.set_zlabel(feat_z)
ax.set_title("3D decision regions (colored by predicted class)")

plt.tight_layout()
plt.show()
# # ──────────────────────────────────────────────────────
# # Animation function (same as yours)
# def rotate(angle):
#     ax.view_init(elev=30, azim=angle)
#
# # Create animation
# n_frames = 36         # number of frames (full 360° turn = smoother, more frames)
# fps      = 1          # frames per second
#
# rot_anim = FuncAnimation(fig, rotate,
#                          frames=np.arange(0, 360, 360 // n_frames),
#                          interval=1000 / fps,
#                          blit=False)
#
# # Save to MP4
# writer = FFMpegWriter(fps=fps, bitrate=1000)  # bitrate controls quality vs size
# rot_anim.save("3d_decision_boundary.mp4", writer=writer, dpi=200)
#
# ────────────────────────────────────────────────────────────────────────────────
# 10)  (optional) SAVE ARRAYS / EMBEDDINGS
# ────────────────────────────────────────────────────────────────────────────────
# np.save("X_train_s.npy", X_train_s)
# np.save("X_test_s.npy",  X_test_s)
