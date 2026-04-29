import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------------------
df = pd.read_csv("summary_imp_comb.csv")

# Drop the columns that directly define or imply the label
drop_cols = ["CFR",
             "P_d/P_a",
             "CFR/FFR",
             "Condition",
             "Geometry Number",
             "R_scale",
             "R_micro",
             "R_total",
             "Location",
             "source"
             ]  # Remove your 'CFR' & 'FFR' columns
df = df.drop(columns=drop_cols)

# Separate features from the label
# The label is 'discord', with -1 for discordant, +1 for accordant
target_col = "discord"
feature_cols = [c for c in df.columns if c != target_col]

"""Debug"""
print("Data shape:", df.shape)
print("Columns:", df.columns)
print("Feature columns:", feature_cols)
print(df[feature_cols].describe())
print(df[feature_cols].isna().sum())


X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values

all_indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    all_indices, test_size=0.4, random_state=42
)

X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

df_train_features = df[feature_cols].iloc[train_idx, :]
df_test_features  = df[feature_cols].iloc[test_idx, :]

# --------------------------------------------------------------
# 2) TRAIN/TEST SPLIT
# --------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.4, random_state=42
# )

# --------------------------------------------------------------
# 3) SCALING
# --------------------------------------------------------------
print("Before scaling:", np.isnan(X_train).sum(), np.isinf(X_train).sum())
nan_rows = df[df[feature_cols].isna().any(axis=1)]
print("Rows with NaNs:")
print(nan_rows)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("After scaling:", np.isnan(X_train_scaled).sum(), np.isinf(X_train_scaled).sum())

# Convert to torch tensors
X_train_torch = torch.from_numpy(X_train_scaled)
y_train_torch = torch.from_numpy(y_train).long()  # long for classification
X_test_torch = torch.from_numpy(X_test_scaled)
y_test_torch = torch.from_numpy(y_test).long()


# --------------------------------------------------------------
# 4) DEFINE A BOTTLENECK NETWORK
# --------------------------------------------------------------
# We'll do a shallow net:
# [input_dim -> hidden_dim -> bottleneck_dim -> 2 classes]

class BottleneckNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes=2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Hidden
        h = self.relu(self.hidden(x))
        # Bottleneck
        z = self.relu(self.bottleneck(h))
        # Classification
        out = self.classifier(z)
        return out, z


input_dim = X_train_torch.shape[1]  # number of features
hidden_dim = 16  # adjustable
bottleneck_dim = 1  # 1-D discovered parameter
num_classes = 2  # binary classification (0 or 1)

model = BottleneckNet(
    input_dim, hidden_dim, bottleneck_dim, num_classes
)

# --------------------------------------------------------------
# 5) TRAINING SETUP
# --------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 300
batch_size = 12


def iterate_minibatches(X_t, y_t, batch_size):
    indices = torch.randperm(X_t.size(0))
    for start_idx in range(0, X_t.size(0), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X_t[excerpt], y_t[excerpt]

test_accs = []
train_losses = []
# --------------------------------------------------------------
# 6) TRAINING LOOP
# --------------------------------------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for Xb, yb in iterate_minibatches(X_train_torch, y_train_torch, batch_size):
        optimizer.zero_grad()
        outputs, _ = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)

    epoch_loss = running_loss / X_train_torch.size(0)

    # (Optional) Evaluate on test set each epoch
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(X_test_torch)
        _, test_preds = torch.max(test_outputs, dim=1)
        test_acc = (test_preds == y_test_torch).float().mean().item()

    # if epoch % 10 == 0:
    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Test Acc: {test_acc:.2f}")

    test_accs.append(test_acc)
    train_losses.append(epoch_loss)

# --------------------------------------------------------------
# 7) EXTRACT BOTTLENECK PARAMETER
# --------------------------------------------------------------
model.eval()
with torch.no_grad():
    _, train_bottleneck = model(X_train_torch)
    _, test_bottleneck = model(X_test_torch)

# These are the discovered parameters for each sample
train_bottleneck_np = train_bottleneck.numpy().flatten()
test_bottleneck_np = test_bottleneck.numpy().flatten()

print("Train label distribution:", np.bincount(y_train))
print("Test label distribution:", np.bincount(y_test))

plt.plot(range(num_epochs), test_accs)
plt.title('Test Accuracy')
plt.show()

plt.plot(range(num_epochs), np.log(train_losses))
plt.title('Training Loss')
plt.show()

# Now you can analyze the distribution of 'z' vs. your 0/1 label
# or see how well 'z' alone separates classes.

train_bottleneck_np = train_bottleneck.numpy().flatten()
test_bottleneck_np  = test_bottleneck.numpy().flatten()

y_train_np = y_train_torch.numpy()
y_test_np  = y_test_torch.numpy()


plt.figure()
for label_val in [0, 1]:
    subset = train_bottleneck_np[y_train_np == label_val]
    plt.hist(subset, bins=10, alpha=0.5, label=f"Class {label_val}")

plt.xlabel("Learned 1D Bottleneck Feature")
plt.ylabel("Frequency")
plt.title("Distribution of Bottleneck by Class (Train)")
plt.legend()
plt.show()

df_bneck = pd.DataFrame({
    "bottleneck": train_bottleneck_np,
    "label": y_train_np
})

plt.figure()
sns.boxplot(x="label", y="bottleneck", data=df_bneck)
plt.title("Boxplot of Bottleneck Values by Class")
plt.show()


logreg = LogisticRegression()
logreg.fit(train_bottleneck_np.reshape(-1,1), y_train_np)
train_pred = logreg.predict(train_bottleneck_np.reshape(-1,1))
test_pred  = logreg.predict(test_bottleneck_np.reshape(-1,1))

print("Train Accuracy on Bottleneck alone:",
      accuracy_score(y_train_np, train_pred))
print("Test Accuracy on Bottleneck alone:",
      accuracy_score(y_test_np,  test_pred))

# 1) Hidden layer
hidden_weight = model.hidden.weight.detach().cpu().numpy()
hidden_bias   = model.hidden.bias.detach().cpu().numpy()

# 2) Bottleneck layer
bottleneck_weight = model.bottleneck.weight.detach().cpu().numpy()
bottleneck_bias   = model.bottleneck.bias.detach().cpu().numpy()

# 3) Final classifier layer
classifier_weight = model.classifier.weight.detach().cpu().numpy()
classifier_bias   = model.classifier.bias.detach().cpu().numpy()

print("Hidden layer weight shape:", hidden_weight.shape)       # (hidden_dim, input_dim)
print("Hidden layer bias shape:", hidden_bias.shape)           # (hidden_dim,)
print("Bottleneck layer weight shape:", bottleneck_weight.shape)  # (1, hidden_dim)
print("Bottleneck layer bias shape:", bottleneck_bias.shape)      # (1,)
print("Classifier weight shape:", classifier_weight.shape)     # (2, 1)
print("Classifier bias shape:", classifier_bias.shape)         # (2,)

train_bottleneck_np = train_bottleneck.numpy().flatten()
df_train_features = df[feature_cols].iloc[train_idx, :].reset_index(drop=True)


df_bottleneck = pd.DataFrame({
    "z": train_bottleneck_np
}).reset_index(drop=True)

# Merge or just concat side by side if your indexing lines up
df_combined = pd.concat([df_train_features.reset_index(drop=True), df_bottleneck], axis=1)

corr_matrix = df_combined.corr()
print("Correlation of each feature with bottleneck z:")
print(corr_matrix["z"].sort_values(ascending=False))
