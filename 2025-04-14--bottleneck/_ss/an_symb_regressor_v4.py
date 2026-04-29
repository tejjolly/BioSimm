# USE ATER AN_BOTTLENECK_V4
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler



# Load the saved arrays
X_all = np.load("X_all.npy")
z_all = np.load("z_all.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

def safe_pow(x, y):
    x = np.clip(x, 1e-3, 1e2)     # avoid zero or negative bases
    y = np.clip(y, -3, 3)         # limit exponent magnitude
    return np.power(x, y)
pow_fun = make_function(function=safe_pow, name='pow', arity=2)

def safe_exp(x):
    x = np.clip(x, -3, 3)  # clip to prevent overflow
    return np.exp(x)
exp_fun = make_function(function=safe_exp, name='exp', arity=1)

import numpy as np
from gplearn.functions import make_function

# --- tanh(x) ---
def safe_tanh(x):
    return np.tanh(x)
tanh_fun = make_function(function=safe_tanh, name='tanh', arity=1)

def square(x):
    return np.power(x, 2)
square_fun = make_function(square, name='square', arity=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -5, 5)))
sigmoid_fun = make_function(sigmoid, name='sigmoid', arity=1)

def abs_fn(x):
    return np.abs(x)
abs_fun = make_function(abs_fn, name='abs', arity=1)



def symbolic_regression_gplearn(X, z):
    """
    Fits a symbolic regression model to the bottleneck z
    returns the best program (string formula).
    """
    # You can define custom functions or use the built-ins
    est = SymbolicRegressor(
        population_size=1000,
        generations=30,
        tournament_size=5,
        stopping_criteria=1e-7,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', pow_fun, exp_fun, tanh_fun,
                      square_fun, sigmoid_fun, abs_fun),
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

# Example usage in your code, after all folds:

# Suppose you have arrays:
# X_all: shape (M, D)
# z_all: shape (M,)

sym_model = symbolic_regression_gplearn(X_all, z_all)
print("\nBest program found:")
print(sym_model._program)

# Evaluate R^2 on the same data (or a hold-out set)
z_sym_pred = sym_model.predict(X_all)
r2_sym = 1 - np.sum((z_all - z_sym_pred)**2)/np.sum((z_all - z_all.mean())**2)
print(f"Symbolic Regressor R^2: {r2_sym:.4f}")

sym_model = symbolic_regression_gplearn(X_all, z_all)
print("Symbolic Program:", sym_model._program)
z_sym = sym_model.predict(X_all)
print("Symbolic R^2:", r2_score(z_all, z_sym))
