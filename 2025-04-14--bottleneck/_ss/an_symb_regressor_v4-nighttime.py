import time
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------
# Load Data
# ----------------------
X_all = np.load("X_all.npy")  # shape: (N, D)
z_all = np.load("z_all.npy")  # shape: (N, 1) for 1D target

# Normalize input and target
scaler_X = StandardScaler()
scaler_z = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)
z_scaled = scaler_z.fit_transform(z_all).flatten()  # shape: (N,)


# ----------------------
# Define Safe Functions
# ----------------------
def safe_pow(x, y):
    # Clip bases and exponents to avoid over/underflow
    x = np.clip(x, 1e-3, 1e2)
    y = np.clip(y, -3, 3)
    return np.power(x, y)


pow_fun = make_function(function=safe_pow, name='pow', arity=2)


def safe_exp(x):
    return np.exp(np.clip(x, -3, 3))


exp_fun = make_function(function=safe_exp, name='exp', arity=1)


def safe_tanh(x):
    return np.tanh(x)


tanh_fun = make_function(function=safe_tanh, name='tanh', arity=1)


def square(x):
    return x ** 2


square_fun = make_function(function=square, name='square', arity=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -5, 5)))


sigmoid_fun = make_function(function=sigmoid, name='sigmoid', arity=1)


def abs_fn(x):
    return np.abs(x)


abs_fun = make_function(function=abs_fn, name='abs', arity=1)


def safe_sin(x):
    # Sine is typically safe but let's just clip extreme values
    return np.sin(np.clip(x, -1e5, 1e5))


sin_fun = make_function(function=safe_sin, name='sin', arity=1)


def safe_cos(x):
    return np.cos(np.clip(x, -1e5, 1e5))


cos_fun = make_function(function=safe_cos, name='cos', arity=1)

# ----------------------
# Define Multiple Function Sets
# ----------------------
function_sets = {
    "basic": ('add', 'sub', 'mul', 'div'),
    "with_abs": ('add', 'sub', 'mul', 'div', abs_fun),
    "no_pow_exp": ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                   tanh_fun, square_fun, sigmoid_fun, abs_fun),
    "trig": ('add', 'sub', 'mul', 'div', sin_fun, cos_fun,
             tanh_fun, abs_fun),
    "full_small": ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                   pow_fun, exp_fun)
    # "full_big": ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
    #              pow_fun, exp_fun, tanh_fun, square_fun,
    #              sigmoid_fun, abs_fun, sin_fun, cos_fun)
}

# ----------------------
# Define Parameter Grid to Explore
# ----------------------
# Here we have examples that increase both population size and generations
# to see if it helps find better solutions (R²) at the cost of longer runtime.
param_grid = [
    {
        "population_size": 2000,
        "generations": 20,
        "tournament_size": 10,
        "stopping_criteria": 1e-4,
        "p_crossover": 0.7,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.15,
    },
    {
        "population_size": 3000,
        "generations": 30,
        "tournament_size": 20,
        "stopping_criteria": 1e-6,
        "p_crossover": 0.8,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    },
    {
        "population_size": 4000,
        "generations": 40,
        "tournament_size": 15,
        "stopping_criteria": 1e-7,
        "p_crossover": 0.8,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    },
    {
        "population_size": 5000,
        "generations": 100,
        "tournament_size": 20,
        "stopping_criteria": 1e-7,
        "p_crossover": 0.85,
        "p_subtree_mutation": 0.05,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    },
    {
        "population_size": 5000,
        "generations": 100,
        "tournament_size": 5,
        "stopping_criteria": 1e-7,
        "p_crossover": 0.85,
        "p_subtree_mutation": 0.05,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    },
    {
        "population_size": 7500,
        "generations": 500,
        "tournament_size": 20,
        "stopping_criteria": 1e-7,
        "p_crossover": 0.85,
        "p_subtree_mutation": 0.05,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    },
    {
        "population_size": 7500,
        "generations": 500,
        "tournament_size": 5,
        "stopping_criteria": 1e-7,
        "p_crossover": 0.85,
        "p_subtree_mutation": 0.05,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
    }
]


# ----------------------
# Symbolic Regressor Wrapper
# ----------------------
def symbolic_regression_gplearn(X, z, function_set, params, seed):
    model = SymbolicRegressor(
        population_size=params["population_size"],
        generations=params["generations"],
        tournament_size=params["tournament_size"],
        stopping_criteria=params["stopping_criteria"],
        function_set=function_set,
        metric='rmse',
        p_crossover=params["p_crossover"],
        p_subtree_mutation=params["p_subtree_mutation"],
        p_hoist_mutation=params["p_hoist_mutation"],
        p_point_mutation=params["p_point_mutation"],
        verbose=1,  # set to 1 for more info
        n_jobs=-1,
        random_state=seed
    )
    model.fit(X, z)
    return model


# ----------------------
# Run Experiments
# ----------------------
num_runs_per_setting = 3
all_results = []

# Start the timer
start_time = time.time()

for fn_label, fn_set in function_sets.items():
    for p_idx, p_dict in enumerate(param_grid):
        print(f"\n=== Function Set: {fn_label} | Param Config #{p_idx + 1} ===")

        best_r2 = -np.inf
        best_model = None
        best_seed = None

        for run in range(num_runs_per_setting):
            seed = 42 + run
            print(f"  > Run {run + 1} with seed={seed}")
            model = symbolic_regression_gplearn(
                X_scaled, z_scaled, fn_set, p_dict, seed
            )
            z_pred = model.predict(X_scaled)
            r2 = r2_score(z_scaled, z_pred)
            print(f"    R²: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_seed = seed

        # Keep track of the best for this combination
        all_results.append({
            "function_set": fn_label,
            "param_config_index": p_idx + 1,
            "best_r2": best_r2,
            "best_program": best_model._program,
            "best_seed": best_seed
        })

# End the timer
end_time = time.time()
elapsed = end_time - start_time

# ----------------------
# Save Summary
# ----------------------
filename = "symbolic_model_experiment_results.txt"
with open(filename, "w") as f:
    for r in all_results:
        f.write(f"Function Set: {r['function_set']}\n")
        f.write(f"Param Config #: {r['param_config_index']}\n")
        f.write(f"Best Seed: {r['best_seed']}\n")
        f.write(f"Best R²: {r['best_r2']:.4f}\n")
        f.write(f"Best Program:\n{r['best_program']}\n\n")

# Print elapsed time
print(f"\nAll experiments complete. Summary saved to '{filename}'.")
print(f"Total runtime: {elapsed:.2f} seconds.")
