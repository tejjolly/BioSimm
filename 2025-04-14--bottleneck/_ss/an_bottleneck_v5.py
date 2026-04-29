import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns


# Flags
inspect_data = True
# User inputs
user_bottleneck_dim = 1
user_epochs = 500
user_batch_size = 8
user_hidden_dim = 8
user_lr = 1e-3
user_cos_max = user_epochs
n_runs = 3
n_splits = 2
num_classes = 2

# ----------------------------------------------------------------
# 1) LOAD DATA
# ----------------------------------------------------------------
df = pd.read_csv("/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data.csv")

# Keep only hyperemic runs
df = df[df['Condition'] == 'Hyperemic']
df = df[df['discord'].notna()]
df = df[df['P_Loss_Coeff'].notna()] # For geometry #5

# features = ["HMR", "P_Loss_Coeff", "BMR/HMR", "discord"]
features = ["P_d/P_a","CFR","discord"]
# features = ["HMR", "BMR/HMR", "discord"]
# features = ["P_Loss_Coeff", "BMR/HMR", "discord"]
# features = ["HMR", "P_Loss_Coeff", "discord"]
# features = ["P_Loss_Coeff", "discord"]
# features = ["HMR", "discord"]

df = df[features]


target_col = "discord"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values
print(np.unique(y))

if inspect_data:
    df.to_csv("data_cleaned.csv", index=False)

# ----------------------------------------------------------------
# 2) DEFINE MODEL
# ----------------------------------------------------------------
class BottleneckNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes=num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.hidden(x))
        z = self.act(self.bottleneck(h))
        out = self.classifier(z)
        return out, z

# ----------------------------------------------------------------
# 3) TRAINING FUNCTION
# ----------------------------------------------------------------
def train_one_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                   X_train_df, run_idx, fold_idx,
                   epochs=user_epochs, batch_size=user_batch_size,
                   hidden_dim=user_hidden_dim, bottleneck_dim=user_bottleneck_dim, lr=user_lr):

    X_train_torch = torch.from_numpy(X_train_fold)
    y_train_torch = torch.from_numpy(y_train_fold).long()
    X_val_torch = torch.from_numpy(X_val_fold)
    y_val_torch = torch.from_numpy(y_val_fold).long()

    input_dim = X_train_fold.shape[1]
    model = BottleneckNet(input_dim, hidden_dim, bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=user_cos_max, eta_min=1e-8)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies_all = []

    def iterate_minibatches(X_t, y_t, batch_sz):
        idx = torch.randperm(X_t.size(0))
        for start_idx in range(0, X_t.size(0), batch_sz):
            excerpt = idx[start_idx:start_idx+batch_sz]
            yield X_t[excerpt], y_t[excerpt]

    for epoch in tqdm(range(epochs), desc=f"Run {run_idx+1} Fold {fold_idx+1}", leave=False):
        model.train()
        running_loss = 0.0
        for Xb, yb in iterate_minibatches(X_train_torch, y_train_torch, batch_size):
            optimizer.zero_grad()
            outputs, _ = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        epoch_train_loss = running_loss / X_train_torch.size(0)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(X_val_torch)
            _, val_preds_epoch = torch.max(val_outputs, dim=1)
            val_acc_epoch = (val_preds_epoch == y_val_torch).float().mean().item()
            val_accuracies_all.append(val_acc_epoch)

        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs, val_bottleneck = model(X_val_torch)
        _, val_preds = torch.max(val_outputs, dim=1)
        val_acc = (val_preds == y_val_torch).float().mean().item()

        y_true = y_val_torch.cpu().numpy()
        y_pred = val_preds.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = float('nan')

        _, train_bottleneck = model(X_train_torch)
        train_bottleneck_np = train_bottleneck.numpy().flatten()

    df_bottleneck = pd.DataFrame({"z": train_bottleneck_np}).reset_index(drop=True)
    df_combined = pd.concat([X_train_df.reset_index(drop=True), df_bottleneck], axis=1)
    corr_matrix = df_combined.corr()
    z_corrs = corr_matrix["z"][feature_cols].values.tolist()

    # return val_acc, model, z_corrs, X_train_fold, train_bottleneck_np, y_train_fold, train_losses, val_accuracies_all
    return (val_acc, model, z_corrs, X_train_fold, train_bottleneck_np, y_train_fold, train_losses, val_accuracies_all,
            sensitivity, specificity, f1)


# ----------------------------------------------------------------
# 4) CROSS VALIDATION + CORRELATION COLLECTION OVER MULTIPLE RUNS
# ----------------------------------------------------------------
all_run_accuracies = []
all_run_correlations = []
all_zs = []
all_Xs = []
all_ys = []
all_fold_accuracies = []

all_fold_sensitivities = []
all_fold_specificities = []
all_fold_f1s = []


for run in range(n_runs):
    print(f"\n================ Run {run+1} / {n_runs} ================")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + run)

    fold_accuracies = []
    fold_correlations = []
    train_losses_all = []
    val_accs_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== Fold {fold+1} / {n_splits} ==========")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        df_train_features = df[feature_cols].iloc[train_idx, :].copy()

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        (acc, model, corr_vec, X_tr_fold, z_vals, y_vals,
         train_losses, val_accuracies, sens, spec, f1) = train_one_fold(
            X_tr_scaled, y_tr, X_val_scaled, y_val, df_train_features,
            run_idx=run, fold_idx=fold,
            epochs=user_epochs, batch_size=user_batch_size,
            hidden_dim=user_hidden_dim, bottleneck_dim=user_bottleneck_dim, lr=user_lr
        )

        fold_accuracies.append(acc)
        all_fold_accuracies.append(acc)
        fold_correlations.append(corr_vec)
        z_vals = z_vals.reshape(-1, user_bottleneck_dim)
        all_zs.append(z_vals)
        all_Xs.append(X_tr_fold)
        all_ys.append(y_vals)
        train_losses_all.append(train_losses)
        val_accs_all.append(val_accuracies)

        all_fold_accuracies.append(acc)
        all_fold_sensitivities.append(sens)
        all_fold_specificities.append(spec)
        all_fold_f1s.append(f1)

        print(f"\n"
              f"Fold {fold+1} Accuracy: {acc:.4f}")
        print(f"    Sensitivity: {sens:.4f}")
        print(f"    Specificity: {spec:.4f}")
        print(f"             f1: {f1:.4f}")




    # Save training and validation loss plots
    # Show training and validation loss plots
    for i in range(n_splits):
        plt.plot(train_losses_all[i], label=f"Fold {i+1}")
    plt.title(f"Run {run+1} - Train Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()

    for i in range(n_splits):
        plt.plot(val_accs_all[i], label=f"Fold {i+1}")
    plt.title(f"Run {run+1} - Val Acc per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Acc.")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()

    all_run_accuracies.append(np.mean(fold_accuracies))
    all_run_correlations.append(np.mean(fold_correlations, axis=0))

# ----------------------------------------------------------------
# 5) RESULTS SUMMARY
# ----------------------------------------------------------------
print("\n================ Overall Summary ================")
# print(f"\nOverall 9-fold Accuracy and Var: {np.mean(all_fold_accuracies):.4f} ± {np.std(all_fold_accuracies, ddof=1):.4f}")
print(f"\nOverall Accuracy   : {np.mean(all_fold_accuracies):.4f} ± {np.std(all_fold_accuracies, ddof=1):.4f}")
print(f"Overall Sensitivity  : {np.mean(all_fold_sensitivities):.4f} ± {np.std(all_fold_sensitivities, ddof=1):.4f}")
print(f"Overall Specificity  : {np.mean(all_fold_specificities):.4f} ± {np.std(all_fold_specificities, ddof=1):.4f}")
print(f"Overall F1 Score     : {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s, ddof=1):.4f}")

avg_corr_df = pd.DataFrame(all_run_correlations, columns=feature_cols)
avg_corr_df.loc["MeanAcrossRuns"] = avg_corr_df.mean(axis=0)
print("\nAverage Correlation of z with Features Across Runs:")
print(avg_corr_df.round(3))

# ----------------------------------------------------------------
# 6) PCA, t-SNE, and Surrogate Regressor on z
# ----------------------------------------------------------------
X_all = np.vstack(all_Xs)

if user_bottleneck_dim > 1:
    z_all = np.vstack(all_zs)  # shape: (N, bottleneck_dim), 2-D version
else:
    z_all = np.concatenate(all_zs) #1-D bottleneck
y_all = np.concatenate(all_ys)

# Linear regression surrogate
reg = LinearRegression()
reg.fit(X_all, z_all)
z_pred = reg.predict(X_all)
r_squared = reg.score(X_all, z_all)

print(f"\nSurrogate Linear R^2 on z: {r_squared:.4f}")

plt.scatter(z_all, z_pred, alpha=0.6)
plt.xlabel("True z")
plt.ylabel("Predicted z")
plt.title("Surrogate Linear Regression on z")
plt.grid(True)
plt.plot([z_all.min(), z_all.max()], [z_all.min(), z_all.max()], 'k--')
plt.show()

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

np.save("X_all.npy", X_all)
np.save("z_all.npy", z_all)
np.save("y_all.npy", y_all)

sns.scatterplot(data=df, x="P_d/P_a", y="CFR", hue="discord", palette="coolwarm")
plt.title("True Discord Labels")
plt.grid(True)
plt.show()


# # PCA visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_all)
# # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=z_all, cmap='viridis', s=20) # 1--D
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=z_all[:,0], cmap='viridis', s=20) # 2-D
# plt.colorbar(label='z (bottleneck)')
# plt.title("PCA of Input Colored by z")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()

# # t-SNE visualization
# tsne = TSNE(n_components=2, perplexity=10, random_state=42)
# X_tsne = tsne.fit_transform(X_all)
# plt.figure()
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=z_all, cmap='plasma', s=20)
# plt.colorbar(label='z (bottleneck)')
# plt.title("t-SNE of Input Colored by z")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.show()