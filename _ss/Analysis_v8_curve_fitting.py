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


df = pd.read_csv('../summary.csv')

# Filter only hyperemic values
df_hyperemic = df[df['Condition'] == 'Hyperemic'].dropna(subset=['CFR/FFR', 'HMR'])

# Plot CFR/FFR vs HMR
plt.figure(figsize=(6,4))
plt.scatter(df_hyperemic['HMR'], df_hyperemic['CFR/FFR'], alpha=0.7, edgecolor='k')
plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("CFR/FFR vs HMR (Hyperemic)")
plt.show()



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
x_data = df_hyperemic['HMR'].values
y_data = df_hyperemic['CFR/FFR'].values

# Fit models
power_params, _ = curve_fit(power_law, x_data, y_data, maxfev=5000)
exp_params, _ = curve_fit(exp_decay, x_data, y_data, maxfev=5000)
log_params, _ = curve_fit(log_model, x_data, y_data, maxfev=5000)

# Generate predictions
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_power = power_law(x_fit, *power_params)
y_exp = exp_decay(x_fit, *exp_params)
y_log = log_model(x_fit, *log_params)

# Plot results
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, alpha=0.2, label="Data", edgecolor='k')

# Add fits
plt.plot(x_fit, y_power, label=f"Power-Law: y={power_params[0]:.2f}x^{power_params[1]:.2f}", linestyle="--")
plt.plot(x_fit, y_exp, label=f"Exp: y={exp_params[0]:.2f}e^(-{exp_params[1]:.2f}x)", linestyle="-.")
plt.plot(x_fit, y_log, label=f"Log: y={log_params[0]:.2f}log(x) + {log_params[1]:.2f}", linestyle=":")

plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("Model Fits for CFR/FFR vs HMR")
plt.legend()
plt.show()

# Print model parameters
power_params, exp_params, log_params

# Compute R-squared and RMSE for each model
r2_power = r2_score(y_data, power_law(x_data, *power_params))
rmse_power = np.sqrt(mean_squared_error(y_data, power_law(x_data, *power_params)))

r2_exp = r2_score(y_data, exp_decay(x_data, *exp_params))
rmse_exp = np.sqrt(mean_squared_error(y_data, exp_decay(x_data, *exp_params)))

r2_log = r2_score(y_data, log_model(x_data, *log_params))
rmse_log = np.sqrt(mean_squared_error(y_data, log_model(x_data, *log_params)))

# Store results in a dictionary
results = {
    "Model": ["Power-Law", "Exponential", "Logarithmic"],
    "R²": [r2_power, r2_exp, r2_log],
    "RMSE": [rmse_power, rmse_exp, rmse_log]
}

# Convert to DataFrame for better visualization
results_df = pd.DataFrame(results)

print(results_df)

# Display the results
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Model Fit Comparison", dataframe=results_df)

