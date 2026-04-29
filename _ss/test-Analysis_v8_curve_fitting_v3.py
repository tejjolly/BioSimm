 #%%
 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 02:05:59 2025

@author: tejjolly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

# --- NEW imports for color mapping ---
import matplotlib.colors as mcolors
import matplotlib.cm as cm


df = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/summary.csv')

# Filter only hyperemic values
df_hyperemic = df[df['Condition'] == 'Hyperemic'].dropna(subset=['CFR/FFR', 'HMR'])

# Define functional forms for fitting

# Power-law: y = a * x^b
def power_law(x, a, b):
    return a * np.power(x, b)

# Exponential decay: y = a * exp(-b * x)
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# Logarithmic model: y = a * log(x) + b
def log_model(x, a, b):
    return a * np.log(x) + b

# Extract x (HMR) and y (CFR/FFR)
x_data = df_hyperemic['HMR'].values.astype(float)
y_data = df_hyperemic['CFR/FFR'].values.astype(float)

# Also extract stenosis percentage (for color)
# Make sure 'Stenosis Percentage' is numeric
stenosis_data = pd.to_numeric(df_hyperemic['Stenosis Percentage'] * 100, errors='coerce')

# Fit models
power_params, _ = curve_fit(power_law, x_data, y_data, maxfev=5000)
exp_params, _ = curve_fit(exp_decay, x_data, y_data, maxfev=5000)
log_params, _ = curve_fit(log_model, x_data, y_data, maxfev=5000)

# Generate predictions
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_power = power_law(x_fit, *power_params)
y_exp = exp_decay(x_fit, *exp_params)
y_log = log_model(x_fit, *log_params)

# Compute R-squared and RMSE for each model
r2_power = r2_score(y_data, power_law(x_data, *power_params))
rmse_power = np.sqrt(mean_squared_error(y_data, power_law(x_data, *power_params)))

r2_exp = r2_score(y_data, exp_decay(x_data, *exp_params))
rmse_exp = np.sqrt(mean_squared_error(y_data, exp_decay(x_data, *exp_params)))

r2_log = r2_score(y_data, log_model(x_data, *log_params))
rmse_log = np.sqrt(mean_squared_error(y_data, log_model(x_data, *log_params)))

# --- Plot ---
plt.figure(figsize=(6, 4))

# (1) DEFINE COLOR MAPPING FOR STENOSIS
# Here we define 5 intervals between min and max stenosis, 
# but you can adjust as needed (e.g., fixed boundaries).
min_sten = stenosis_data.min()
max_sten = stenosis_data.max()
custom_boundaries = np.linspace(min_sten, max_sten, 6)  # 5 intervals
norm = mcolors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
cmap = cm.get_cmap("RdYlGn_r")  # or any other colormap

# (2) PLOT SCATTER COLORED BY STENOSIS
sc = plt.scatter(
    x_data,
    y_data,
    c=stenosis_data,
    cmap=cmap,
    norm=norm,
    alpha=1,
    edgecolor='k',
    s=20
)

# Add a colorbar
cb = plt.colorbar(sc)
cb.set_label("Stenosis [%]")
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

# Plot each fitted curve
plt.plot(x_fit, y_power, linestyle="--", color='red', alpha = 0.5,
         label=f"Power-Law: y={power_params[0]:.2f}x^{power_params[1]:.2f}")
plt.plot(x_fit, y_exp, linestyle="-.", color='green', alpha = 0.5,
         label=f"Exp: y={exp_params[0]:.2f}e^(-{exp_params[1]:.2f}x)")
plt.plot(x_fit, y_log, linestyle=":", color='orange', alpha = 0.5,
         label=f"Log: y={log_params[0]:.2f}log(x) + {log_params[1]:.2f}")

plt.xlabel("HMR [mmHg/cm/s]")
plt.ylabel("CFR/FFR")
plt.title("Curve Fits for CFR/FFR vs HMR")
plt.legend()

# --- Floating R² & RMSE Annotations ---
def add_annotation(x_fit, y_fit, fraction, text, color):
    """Annotate at a given fraction along the fit curve."""
    idx = int(fraction * len(x_fit))  # Find the index at the fraction position
    plt.annotate(
        text,
        xy=(x_fit[idx], y_fit[idx]),
        xytext=(x_fit[idx] + 0.2, y_fit[idx] + 0.9),
        arrowprops=dict(arrowstyle="-", lw=1.5, color=color),
        fontsize=7,
        color=color,
    )

# Add annotations at 2/5, 3/5, and 4/5 of the way along the x_fit
add_annotation(x_fit, y_power, 2/5, f"R²={r2_power:.3f}\nRMSE={rmse_power:.3f}", "red")
add_annotation(x_fit, y_exp, 3/5, f"R²={r2_exp:.3f}\nRMSE={rmse_exp:.3f}", "green")
add_annotation(x_fit, y_log, 4/5, f"R²={r2_log:.3f}\nRMSE={rmse_log:.3f}", "orange")

plt.tight_layout()
plt.show()

# Print model parameters
print("Power law params:", power_params)
print("Exp decay params:", exp_params)
print("Log model params:", log_params)

# Print a results table
results = {
    "Model": ["Power-Law", "Exponential", "Logarithmic"],
    "R²": [r2_power, r2_exp, r2_log],
    "RMSE": [rmse_power, rmse_exp, rmse_log]
}
results_df = pd.DataFrame(results)
print(results_df)


# Compute predictions for all models on x_data
y_pred_power_data = power_law(x_data, *power_params)
y_pred_exp_data = exp_decay(x_data, *exp_params)
y_pred_log_data = log_model(x_data, *log_params)

# Compute residuals for all models
residuals_power = y_data - y_pred_power_data
residuals_exp = y_data - y_pred_exp_data
residuals_log = y_data - y_pred_log_data

# Plot Residuals vs. Predicted for all models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = ["Power-Law", "Exponential", "Logarithmic"]
residuals_list = [residuals_power, residuals_exp, residuals_log]
predictions_list = [y_pred_power_data, y_pred_exp_data, y_pred_log_data]

for ax, model, residuals, predictions in zip(axes, models, residuals_list, predictions_list):
    ax.scatter(x_data, residuals, alpha=0.7, color='tab:blue', edgecolor='tab:blue')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel("HMR [mmHg/cm/s]")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals for {model} Fit")

plt.tight_layout()
plt.show()

# Plot Residual Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, model, residuals in zip(axes, models, residuals_list):
    ax.hist(residuals, bins=15, color='tab:blue', edgecolor='k', alpha=0.5, density=True)
    
    # Fit normal distribution curve
    mu, std = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(min(residuals), max(residuals), 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, std), 'k--', label="Normal Fit")
    
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution ({model})")
    ax.legend()

plt.tight_layout()
plt.show()

# Sensitivity Analysis for all models
sensitivity_results_power = {}
sensitivity_results_exp = {}
sensitivity_results_log = {}

hmr_variation = np.array([0.9, 1.0, 1.1])  # -10%, baseline, +10%

for hmr_factor in hmr_variation:
    varied_x = x_data * hmr_factor
    sensitivity_results_power[f"HMR {hmr_factor*100:.0f}%"] = power_law(varied_x, *power_params)
    sensitivity_results_exp[f"HMR {hmr_factor*100:.0f}%"] = exp_decay(varied_x, *exp_params)
    sensitivity_results_log[f"HMR {hmr_factor*100:.0f}%"] = log_model(varied_x, *log_params)

# Convert to DataFrames for visualization
sensitivity_df_power = pd.DataFrame(sensitivity_results_power, index=x_data)
sensitivity_df_exp = pd.DataFrame(sensitivity_results_exp, index=x_data)
sensitivity_df_log = pd.DataFrame(sensitivity_results_log, index=x_data)

# Plot sensitivity analysis for all models
plt.figure(figsize=(8, 5))

# Sort x_data for a clearer trend visualization
sorted_indices = np.argsort(x_data)
x_sorted = x_data[sorted_indices]

# Plot Power-Law sensitivity
plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 0], 'r--', label="Power-Law: HMR 90%")
plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 1], 'r-', label="Power-Law: HMR 100%")
plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 2], 'r-.', label="Power-Law: HMR 110%")

plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)
plt.show()

# Plot Exponential sensitivity
plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 0], 'g--', label="Exponential: HMR 90%")
plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 1], 'g-', label="Exponential: HMR 100%")
plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 2], 'g-.', label="Exponential: HMR 110%")

plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)
plt.show()

# Plot Logarithmic sensitivity
plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 0], 'b--', label="Log: HMR 90%")
plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 1], 'b-', label="Log: HMR 100%")
plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 2], 'b-.', label="Log: HMR 110%")

plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)
plt.show()
