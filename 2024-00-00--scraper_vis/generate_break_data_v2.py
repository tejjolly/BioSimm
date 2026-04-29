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

X_MIN, X_MAX = 0.0, 8
X_BREAK = 2

INTERCEPT = 0.5
SLOPE_1 = .2
SLOPE_2 = 0.3

# NEW: per-series vertical spacing (e.g., 0.2 => series 0:+0.0, 1:+0.2, 2:+0.4, ...)
INTERCEPT_SPACING = 0.1

X_JITTER_FRAC = 0.2
Y_NOISE_STD = 0.05

MIN_DX = 1e-6
MAX_TRIES = 200

rng = np.random.default_rng()

# -------------------------------------------------
# MODEL
# -------------------------------------------------
def piecewise_y(xi, intercept_series):
    """Piecewise-linear y with a per-series intercept."""
    if xi <= X_BREAK:
        return intercept_series + SLOPE_1 * xi
    yb = intercept_series + SLOPE_1 * X_BREAK
    return yb + SLOPE_2 * (xi - X_BREAK)

def generate_monotone_x(n):
    # --- NEW: choose a per-series x-start and x-end inside the global domain ---
    # start can float upward by up to ~1 nominal step * X_JITTER_FRAC
    dx_global = (X_MAX - X_MIN) / (n - 1)
    start_span = X_JITTER_FRAC * dx_global  # how far above X_MIN the series may start

    x_start = rng.uniform(X_MIN, min(X_MAX - MIN_DX * (n - 1), X_MIN + start_span))

    # end must be above start by at least a small range
    min_range = max(MIN_DX * (n - 1), 0.25 * dx_global)
    x_end = rng.uniform(min(X_MAX, x_start + min_range), X_MAX)

    # base grid lives on [x_start, x_end] now (not [X_MIN, X_MAX])
    x_base = np.linspace(x_start, x_end, n)
    dx_nom = (x_end - x_start) / (n - 1)

    x = np.empty(n)

    for i in range(n):
        for _ in range(MAX_TRIES):
            # jitter EVERY point, including endpoints, around the series-specific base
            xi = x_base[i] + rng.normal(0.0, X_JITTER_FRAC * dx_nom)
            xi = np.clip(xi, X_MIN, X_MAX)

            if i == 0:
                x[i] = xi
                break
            if xi > x[i - 1] + MIN_DX:
                x[i] = xi
                break
        else:
            # fallback: ensure progress
            if i == 0:
                x[i] = np.clip(x_base[0], X_MIN, X_MAX)
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

    # NEW: per-series intercept offset
    intercept_s = INTERCEPT + s * INTERCEPT_SPACING

    x = generate_monotone_x(N_PER_SERIES)
    y_clean = np.array([piecewise_y(xi, intercept_s) for xi in x])
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

# reference "true model" for series 0 (no offset)
x_dense = np.linspace(X_MIN, X_MAX, 400)
y_dense = np.array([piecewise_y(xi, INTERCEPT) for xi in x_dense])
plt.plot(x_dense, y_dense, "k--", linewidth=2.5, label="True model (series 0)")

plt.axvline(X_BREAK, linestyle=":", linewidth=2.5, color="black", label="Breakpoint")

plt.xlabel("HMR")
plt.ylabel("P_d / P_a (FFR)")
plt.title("Synthetic multi-series breakpoint test data")
# plt.legend(title="Stenosis %")
plt.tight_layout()
plt.show()