import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pysr import PySRRegressor

# ----------------------------------------------------------------
# 1) Load your data
# ----------------------------------------------------------------
# Suppose you already saved these:
#   - X_all.npy: shape (N, D)
#   - z_all.npy: shape (N,) or (N, 1)
X = np.load("X_all.npy")
z = np.load("z_all.npy")

# Ensure z is 1D (PySR expects y array shape of (N,))
if z.ndim == 2 and z.shape[1] == 1:
    z = z.flatten()

print("Shapes:", X.shape, z.shape)

# ----------------------------------------------------------------
# 2) (Optional) Scale the data
# ----------------------------------------------------------------
# This step is optional but often helps if your features are on very different scales.
X_scaler = StandardScaler()
z_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
z_scaled = z_scaler.fit_transform(z.reshape(-1, 1)).flatten()

# ----------------------------------------------------------------
# 3) Define and configure the PySR model
# ----------------------------------------------------------------
model = PySRRegressor(
    # Number of iterations of the search.
    niterations=1000,

    # A list of unary operators to allow in the symbolic expressions:
    unary_operators=[
        "sin", "tan", "exp",
        # "log_abs", "sqrt_abs",
        "relu",  # etc.
    ],

    # A list of binary operators to allow. By default, PySR uses "+", "-", "*", and "/".
    binary_operators=[
        "+", "-", "*", "/",
        "pow",  # exponentiation (x^y)
        # "max", "min",  # can also add if desired
    ],
    constraints={
        "pow": (9, 1)
    },

    # If you want to see partial results in real time:
    progress=True,

    # PySR can multi-thread by default. You may want to limit ncores if needed:
    # ncores=4,

    # Tweak complexity to reduce or allow more complex expressions:
    maxsize=20,

    # The higher this is, the more the algorithm tries to simplify found expressions.
    # This is a good starting point for a small system but can be adjusted:
    populations=100,

    # If you want, you can name your features and target to get nicer prints:

    model_selection="best",  # or "accuracy" if you want best R^2 over complexity
    parsimony = 0.01
)

# ----------------------------------------------------------------
# 4) Fit the model
# ----------------------------------------------------------------
model.fit(X_scaled, z_scaled, variable_names=[f"x{i}" for i in range(X.shape[1])],)

# ----------------------------------------------------------------
# 5) Evaluate the model
# ----------------------------------------------------------------

# Show all discovered equations, sorted by complexity (and also by fitness):
print("\nDiscovered equations (model.equations_):")
print(model.equations_)

# The final best equation:
best_equation = model.get_best()
print("\nBest equation found:")
print(best_equation)

# Evaluate R^2 on training set:
r2_train = model.score(X_scaled, z_scaled)
print(f"\nTraining R^2: {r2_train:.4f}")

# ----------------------------------------------------------------
# 6) Predict on new data
# ----------------------------------------------------------------
# Suppose we have a new batch X_new. We should scale X_new similarly:
# X_new_scaled = X_scaler.transform(X_new)
# z_pred_scaled = model.predict(X_new_scaled)
# Then invert scaling if needed:
# z_pred = z_scaler.inverse_transform(z_pred_scaled.reshape(-1, 1)).flatten()

# ----------------------------------------------------------------
# 7) (Optional) Export / Save your best symbolic model
# ----------------------------------------------------------------
# For example, to get a stand-alone Python function for the best equation:
# model.print()

# Or:
equation_str = model.sympy()
print("Sympy equation:", equation_str)