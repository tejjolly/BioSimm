#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 02:27:39 2025

@author: tejjolly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

df = pd.read_csv('../data/data.csv')
yvar = 'CFR'
garcia = False
# 1) Filter only hyperemic values with non-null CFR/FFR & HMR
if garcia:
    df_hyperemic = df[(df['Condition'] == 'Hyperemic') &
                      df[yvar].notnull() &
                      df['HMR'].notnull()]
else:
    df_hyperemic = df[(df['Condition'] == 'Hyperemic') &
                      df[yvar].notnull() &
                      df['HMR'].notnull() &
                      (df['source'] == 'mine')
    ]

# 2) Separate your data (source='mine') from Garcia's (source='garcia')
df_hyperemic_mine = df_hyperemic[df_hyperemic['source'] == 'mine']
df_hyperemic_garcia = df_hyperemic[df_hyperemic['source'] == 'garcia']

# -- You can do a single fit to all hyperemic data combined:
x_data = df_hyperemic['HMR'].values
y_data = df_hyperemic[yvar].values

# Define functional forms for fitting
def power_law(x, a, b):
    return a * np.power(x, b)

def exp_decay(x, a, b):
    return a * np.exp(-b * x)

def log_model(x, a, b):
    return a * np.log(x) + b

def linear_model(x, a, b):
    return a * x + b



# Fit models (over entire hyperemic set)
power_params, _ = curve_fit(power_law, x_data, y_data, maxfev=5000)
exp_params, _ = curve_fit(exp_decay, x_data, y_data, maxfev=5000)
log_params, _ = curve_fit(log_model, x_data, y_data, maxfev=5000)


# Predictions for plotting
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_power = power_law(x_fit, *power_params)
y_exp   = exp_decay(x_fit, *exp_params)
y_log   = log_model(x_fit, *log_params)

# Compute R², RMSE for each model
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

r2_power  = r2_score(y_data, power_law(x_data, *power_params))
rmse_power = rmse(y_data, power_law(x_data, *power_params))

r2_exp    = r2_score(y_data, exp_decay(x_data, *exp_params))
rmse_exp   = rmse(y_data, exp_decay(x_data, *exp_params))

r2_log    = r2_score(y_data, log_model(x_data, *log_params))
rmse_log   = rmse(y_data, log_model(x_data, *log_params))

# --- Plot ---
plt.figure(figsize=(7, 4))

# Plot your data in black
plt.scatter(df_hyperemic_mine['HMR'],
            df_hyperemic_mine[yvar],
            alpha=1,
            edgecolor="black",
            color="#5E9096",
            s=20,
            label="Our Data" if garcia else None)

# Plot Garcia's data in gray
if garcia:
    plt.scatter(df_hyperemic_garcia['HMR'], df_hyperemic_garcia[yvar],
                alpha=0.35, edgecolor='gray', color='gray',
                s=20, label="External Data")

# Plot each fitted curve
plt.plot(x_fit, y_power, linestyle="--", color='red',
         label=f"Power-Law: y={power_params[0]:.2f}x^{power_params[1]:.2f}")
plt.plot(x_fit, y_exp, linestyle="-.", color='green',
         label=f"Exp: y={exp_params[0]:.2f}e^(-{exp_params[1]:.2f}x)")
plt.plot(x_fit, y_log, linestyle=":", color='orange',
         label=f"Log: y={log_params[0]:.2f}log(x) + {log_params[1]:.2f}")

plt.xlabel("HMR")
plt.ylabel(yvar)
plt.title(f"{yvar} vs HMR")
plt.legend()

def add_annotation(xvals, yvals, fraction, text, color):
    """Annotate at a given fraction along the fit curve."""
    idx = int(fraction * len(xvals))
    plt.annotate(
        text,
        xy=(xvals[idx], yvals[idx]),
        xytext=(xvals[idx] + 0.2, yvals[idx] + 1),
        arrowprops=dict(arrowstyle="-", lw=1.5, color=color),
        fontsize=7,
        color=color,
    )

# Add R² and RMSE annotations along each curve
add_annotation(x_fit, y_power, 2/5, f"R²={r2_power:.3f}\nRMSE={rmse_power:.3f}", "red")
add_annotation(x_fit, y_exp,   3/5, f"R²={r2_exp:.3f}\nRMSE={rmse_exp:.3f}", "green")
add_annotation(x_fit, y_log,   4/5, f"R²={r2_log:.3f}\nRMSE={rmse_log:.3f}", "orange")

plt.tight_layout()
plt.show()

print("Power law params:", power_params)
print("Exp decay params:", exp_params)
print("Log model params:", log_params)

# Results Table
results = {
    "Model": ["Power-Law", "Exponential", "Logarithmic"],
    "R²": [r2_power, r2_exp, r2_log],
    "RMSE": [rmse_power, rmse_exp, rmse_log]
}
results_df = pd.DataFrame(results)
print(results_df)

# # --- Residual Plots (Same approach, just repeated) ---
# y_pred_power_data = power_law(x_data, *power_params)
# y_pred_exp_data   = exp_decay(x_data, *exp_params)
# y_pred_log_data   = log_model(x_data, *log_params)
#
# residuals_power = y_data - y_pred_power_data
# residuals_exp   = y_data - y_pred_exp_data
# residuals_log   = y_data - y_pred_log_data
#
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# models         = ["Power-Law", "Exponential", "Logarithmic"]
# residuals_list = [residuals_power, residuals_exp, residuals_log]
# predictions    = [y_pred_power_data, y_pred_exp_data, y_pred_log_data]
#
# for ax, model, residuals, ypred in zip(axes, models, residuals_list, predictions):
#     # Plot residuals vs. HMR
#     ax.scatter(x_data, residuals, alpha=0.7, color='tab:blue', edgecolor='tab:blue')
#     ax.axhline(0, color='k', linestyle='--')
#     ax.set_xlabel("HMR")
#     ax.set_ylabel("Residuals")
#     ax.set_title(f"Residuals for {model} Fit")
#
# plt.tight_layout()
# plt.show()
#
# # --- Residual Distributions ---
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# for ax, model, residuals in zip(axes, models, residuals_list):
#     ax.hist(residuals, bins=15, color='tab:blue', edgecolor='k', alpha=0.5, density=True)
#     mu, std = np.mean(residuals), np.std(residuals)
#     x_norm = np.linspace(min(residuals), max(residuals), 100)
#     ax.plot(x_norm, stats.norm.pdf(x_norm, mu, std), 'k--', label="Normal Fit")
#     ax.set_xlabel("Residuals")
#     ax.set_ylabel("Density")
#     ax.set_title(f"Residual Distribution ({model})")
#     ax.legend()
#
# plt.tight_layout()
# plt.show()
#
# # --- Sensitivity Analysis (±10% in HMR) ---
# sensitivity_results_power = {}
# sensitivity_results_exp   = {}
# sensitivity_results_log   = {}
#
# hmr_variation = np.array([0.9, 1.0, 1.1])
#
# for factor in hmr_variation:
#     varied_x = x_data * factor
#     sensitivity_results_power[f"HMR {factor*100:.0f}%"] = power_law(varied_x, *power_params)
#     sensitivity_results_exp[f"HMR {factor*100:.0f}%"]   = exp_decay(varied_x, *exp_params)
#     sensitivity_results_log[f"HMR {factor*100:.0f}%"]   = log_model(varied_x, *log_params)
#
# sensitivity_df_power = pd.DataFrame(sensitivity_results_power, index=x_data)
# sensitivity_df_exp   = pd.DataFrame(sensitivity_results_exp,   index=x_data)
# sensitivity_df_log   = pd.DataFrame(sensitivity_results_log,   index=x_data)
#
# # Sort x_data for a cleaner plot
# sorted_indices = np.argsort(x_data)
# x_sorted = x_data[sorted_indices]
#
# # Plot Power-Law Sensitivity
# plt.figure(figsize=(8, 5))
# plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 0], 'r--', label="Power-Law: HMR 90%")
# plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 1], 'r-',  label="Power-Law: HMR 100%")
# plt.plot(x_sorted, sensitivity_df_power.iloc[sorted_indices, 2], 'r-.', label="Power-Law: HMR 110%")
# plt.xlabel("HMR")
# plt.ylabel("CFR/FFR")
# plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%) - Power-Law")
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(True)
# plt.show()
#
# # Plot Exponential Sensitivity
# plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 0], 'g--', label="Exponential: HMR 90%")
# plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 1], 'g-',  label="Exponential: HMR 100%")
# plt.plot(x_sorted, sensitivity_df_exp.iloc[sorted_indices, 2], 'g-.', label="Exponential: HMR 110%")
# plt.xlabel("HMR")
# plt.ylabel("CFR/FFR")
# plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%) - Exponential")
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(True)
# plt.show()
#
# # Plot Logarithmic Sensitivity
# plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 0], 'b--', label="Log: HMR 90%")
# plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 1], 'b-',  label="Log: HMR 100%")
# plt.plot(x_sorted, sensitivity_df_log.iloc[sorted_indices, 2], 'b-.', label="Log: HMR 110%")
# plt.xlabel("HMR")
# plt.ylabel("CFR/FFR")
# plt.title("Sensitivity Analysis: CFR/FFR vs HMR (±10%) - Logarithmic")
# plt.legend(loc="upper right", fontsize=8)
# plt.grid(True)
# plt.show()
