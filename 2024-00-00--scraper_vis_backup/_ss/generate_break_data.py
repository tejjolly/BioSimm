#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
OUTCSV = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/"
    "Post_Processing/data/break_test.csv"
)

# -------------------------------------------------
# USER PARAMETERS
# -------------------------------------------------
N_SERIES = 8
N_PER_SERIES = 4

STEN_START = 0.06
STEN_STEP  = 0.01

X_MIN, X_MAX = 0.0, 3
X_BREAK = 1

INTERCEPT = 5
SLOPE_1 = 1.0
SLOPE_2 = 1.1

X_JITTER_FRAC = 0.4
Y_NOISE_STD = 0.1

MIN_DX = 1e-6
MAX_TRIES = 200

rng = np.random.default_rng()

# -------------------------------------------------
# MODEL
# -------------------------------------------------
def piecewise_y(xi):
    if xi <= X_BREAK:
        return INTERCEPT + SLOPE_1 * xi
    yb = INTERCEPT + SLOPE_1 * X_BREAK
    return yb + SLOPE_2 * (xi - X_BREAK)

def generate_monotone_x(n):
    x_base = np.linspace(X_MIN, X_MAX, n)
    dx_nom = (X_MAX - X_MIN) / (n - 1)

    x = np.empty(n)
    x[0] = x_base[0]

    for i in range(1, n):
        for _ in range(MAX_TRIES):
            xi = x_base[i] + rng.normal(0.0, X_JITTER_FRAC * dx_nom)
            xi = np.clip(xi, X_MIN, X_MAX)
            if xi > x[i - 1] + MIN_DX:
                x[i] = xi
                break
        else:
            x[i] = min(X_MAX, x[i - 1] + 0.25 * dx_nom)

    return x

# -------------------------------------------------
# GENERATE DATA
# -------------------------------------------------
rows = []
geom_num = 0

for s in range(N_SERIES):
    sten = STEN_START + s * STEN_STEP
    x = generate_monotone_x(N_PER_SERIES)
    y_clean = np.array([piecewise_y(xi) for xi in x])
    y = y_clean + rng.normal(0.0, Y_NOISE_STD, size=N_PER_SERIES)

    for i in range(N_PER_SERIES):
        rows.append({
            "Condition": "Hyperemic",
            "Geometry Number": geom_num,
            "Location": "LAD",
            "Length": 1,
            "Stenosis Percentage": float(f"{sten:.2f}"),
            "HMR": x[i],
            "P_d/P_a": y[i],
        })

    geom_num += 1

df = pd.DataFrame(rows)

# -------------------------------------------------
# SAVE
# -------------------------------------------------
os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
df.to_csv(OUTCSV, sep=",", index=False)
print(df)
print(f"Saved {len(df)} rows to:")
print(OUTCSV)

# -------------------------------------------------
# PLOT
# -------------------------------------------------
plt.figure(figsize=(9, 5))

for sten, gdf in df.groupby("Stenosis Percentage"):
    gdf = gdf.sort_values("HMR")
    plt.plot(gdf["HMR"], gdf["P_d/P_a"], linewidth=2.2, alpha=0.9)
    plt.scatter(gdf["HMR"], gdf["P_d/P_a"], s=50, label=f"{sten:.2f}")

# true underlying model (for reference)
x_dense = np.linspace(X_MIN, X_MAX, 400)
y_dense = np.array([piecewise_y(xi) for xi in x_dense])
plt.plot(x_dense, y_dense, "k--", linewidth=2.5, label="True model")

plt.axvline(X_BREAK, linestyle=":", linewidth=2.5, color="black", label="Breakpoint")

plt.xlabel("HMR")
plt.ylabel("P_d / P_a (FFR)")
plt.title("Synthetic multi-series breakpoint test data")
plt.legend(title="Stenosis %")
plt.tight_layout()
plt.show()