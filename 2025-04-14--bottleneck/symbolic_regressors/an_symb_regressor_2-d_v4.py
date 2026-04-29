import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Then run symbolic_regression_gplearn(X_scaled, z_rf_pred)


# Load arrays
X_all = np.load("X_all.npy")  # shape: (N, D)
z_all = np.load("z_all.npy")  # shape: (N, bottleneck_dim)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Safe function definitions
def safe_pow(x, y):
    x = np.clip(x, 1e-3, 1e2)
    y = np.clip(y, -3, 3)
    return np.power(x, y)
pow_fun = make_function(function=safe_pow, name='pow', arity=2)


def safe_exp(x):
    x = np.clip(x, -3, 3)
    return np.exp(x)
exp_fun = make_function(function=safe_exp, name='exp', arity=1)


def safe_tanh(x):
    return np.tanh(x)
tanh_fun = make_function(function=safe_tanh, name='tanh', arity=1)


def square(x):
    return np.power(x, 2)
square_fun = make_function(function=square, name='square', arity=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -5, 5)))
sigmoid_fun = make_function(function=sigmoid, name='sigmoid', arity=1)


def abs_fn(x):
    return np.abs(x)
abs_fun = make_function(function=abs_fn, name='abs', arity=1)

# Function to fit symbolic model
def symbolic_regression_gplearn(X, z):
    est = SymbolicRegressor(
        population_size=2000,
        generations=40,
        tournament_size=10,
        stopping_criteria=1e-7,
        function_set=(
            'add', 'sub', 'mul', 'div', 'sqrt', 'log',
            # pow_fun, exp_fun,
            tanh_fun, square_fun, sigmoid_fun, abs_fun
        ),
        metric='rmse',
        p_crossover=0.75,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    est.fit(X, z)
    return est

num_runs = 3
# Loop through each dimension of z
best_models = []
for i in range(z_all.shape[1]):
    print(f"\ Symbolic Regression for z dimension {i}")

    best_r2 = -np.inf
    best_model = None

    for run in range(num_runs):
        model = symbolic_regression_gplearn(X_scaled, z_all[:, i])
        z_pred = model.predict(X_scaled)
        r2 = r2_score(z_all[:, i], z_pred)
        print(f"Run {run + 1} R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    print(f"\n Best R² for z[{i}]: {best_r2:.4f}")
    print("Best Program:\n", best_model._program)
    best_models.append((best_r2, best_model._program))

# Optional: save results
with open("symbolic_models_summary.txt", "w") as f:
    for i, (r2, prog) in enumerate(best_models):
        f.write(f"z[{i}] R²: {r2:.4f}\n")
        f.write(f"Program: {prog}\n\n")
