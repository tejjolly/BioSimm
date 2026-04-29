#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 02:27:39 2025

@author: ...
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 1) LOAD & PREPARE DATA
# ---------------------------------------------------------------------------
df = pd.read_csv('./summary2.csv')

# Filter hyperemic rows that have HMR and CFR/FFR
df_hyperemic = df[(df['Condition'] == 'Hyperemic') & 
                  df['CFR/FFR'].notnull() & 
                  df['HMR'].notnull()]

# Separate your data vs. García's
df_hyperemic_mine = df_hyperemic[df_hyperemic['source'] == 'mine']
df_hyperemic_garcia = df_hyperemic[df_hyperemic['source'] == 'garcia']

# If you plan to do k-means on P_Loss_Coeff, ensure it's notnull:
df_hyperemic_km = df_hyperemic.dropna(subset=['P_Loss_Coeff']).copy()

# ---------------------------------------------------------------------------
# 2) DEFINE MODELS & FIT
#    (same as your original code)
# ---------------------------------------------------------------------------
# x_data = HMR, y_data = CFR/FFR
x_data = df_hyperemic['HMR'].values
y_data = df_hyperemic['CFR/FFR'].values

def power_law(x, a, b):
    return a * np.power(x, b)

def exp_decay(x, a, b):
    return a * np.exp(-b * x)

def log_model(x, a, b):
    return a * np.log(x) + b

# Fit each model across the entire hyperemic dataset
power_params, _ = curve_fit(power_law, x_data, y_data, maxfev=5000)
exp_params, _   = curve_fit(exp_decay, x_data, y_data, maxfev=5000)
log_params, _   = curve_fit(log_model, x_data, y_data, maxfev=5000)

# Generate curves for plotting
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_power = power_law(x_fit, *power_params)
y_exp   = exp_decay(x_fit, *exp_params)
y_log   = log_model(x_fit, *log_params)

# Evaluate R², RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

r2_power   = r2_score(y_data, power_law(x_data, *power_params))
rmse_power = rmse(y_data, power_law(x_data, *power_params))

r2_exp   = r2_score(y_data, exp_decay(x_data, *exp_params))
rmse_exp = rmse(y_data, exp_decay(x_data, *exp_params))

r2_log   = r2_score(y_data, log_model(x_data, *log_params))
rmse_log = rmse(y_data, log_model(x_data, *log_params))

# ---------------------------------------------------------------------------
# 3) SCATTER PLOT #1: BLACK vs GRAY, plus fitted curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4))

# Plot your data in black
plt.scatter(
    df_hyperemic_mine['HMR'],
    df_hyperemic_mine['CFR/FFR'],
    alpha=1, edgecolor='black', color='black',
    s=20, label="Our Data"
)

# Plot García's data in gray
plt.scatter(
    df_hyperemic_garcia['HMR'],
    df_hyperemic_garcia['CFR/FFR'],
    alpha=0.35, edgecolor='gray', color='gray',
    s=20, label="External Data"
)

# Overplot each fitted curve
plt.plot(x_fit, y_power, linestyle="--", color='red',
         label=f"Power-Law: y={power_params[0]:.2f}x^{power_params[1]:.2f}")
plt.plot(x_fit, y_exp, linestyle="-.", color='green',
         label=f"Exp: y={exp_params[0]:.2f} e^(-{exp_params[1]:.2f}x)")
plt.plot(x_fit, y_log, linestyle=":", color='orange',
         label=f"Log: y={log_params[0]:.2f} log(x) + {log_params[1]:.2f}")

plt.xlabel("HMR")
plt.ylabel("CFR/FFR")
plt.title("Model Fits for CFR/FFR vs HMR (Black=Ours, Gray=Garcia)")
plt.legend(loc='best')

# Annotate R² and RMSE (optional)
def add_annotation(xvals, yvals, fraction, text, color):
    idx = int(fraction * len(xvals))
    plt.annotate(
        text,
        xy=(xvals[idx], yvals[idx]),
        xytext=(xvals[idx] + 0.2, yvals[idx] + 0.9),
        arrowprops=dict(arrowstyle="-", lw=1.5, color=color),
        fontsize=7,
        color=color,
    )
add_annotation(x_fit, y_power, 2/5, f"R²={r2_power:.3f}\nRMSE={rmse_power:.3f}", "red")
add_annotation(x_fit, y_exp,   3/5, f"R²={r2_exp:.3f}\nRMSE={rmse_exp:.3f}",   "green")
add_annotation(x_fit, y_log,   4/5, f"R²={r2_log:.3f}\nRMSE={rmse_log:.3f}",   "orange")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 4) K-MEANS on P_Loss_Coeff (1D) & SCATTER PLOT #2: Color by cluster
# ---------------------------------------------------------------------------
# (If you want a multi-dimensional cluster, adapt the line below.)
df_km = df_hyperemic_km  # rows that have non-null P_Loss_Coeff
X_km1 = df_km[['P_Loss_Coeff']].values  # 1D (n_samples x 1)
X_km2 = df_km[['P_Loss_Coeff','BMR/HMR']].values  # 1D (n_samples x 1)

X_kms = [X_km1, X_km2]

i = 0
for X_km in X_kms:

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_km['cluster'] = kmeans.fit_predict(X_km)
    
    print("K-Means cluster label counts:")
    print(df_km['cluster'].value_counts())
    
    # Plot the same HMR vs CFR/FFR, but color by cluster
    plt.figure(figsize=(7, 4))
    
    # We'll define some cluster colors
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for c_label in sorted(df_km['cluster'].unique()):
        subdf = df_km[df_km['cluster'] == c_label]
        plt.scatter(
            subdf['HMR'],
            subdf['CFR/FFR'],
            color=cluster_colors[c_label],
            alpha=0.6,
            edgecolor='k',
            s=40,
            label=f"Cluster {c_label}"
        )
    
    # Overplot the same fitted curves for reference
    plt.plot(x_fit, y_power, linestyle="--", color='gray', lw=1)
    plt.plot(x_fit, y_exp,   linestyle="-.", color='gray', lw=1)
    plt.plot(x_fit, y_log,   linestyle=":",  color='gray', lw=1)
    
    plt.xlabel("HMR")
    plt.ylabel("CFR/FFR")
    if i == 0:
        plt.title("K-Means Clusters (1-D on P_Loss_Coeff)")
    elif i == 1:
        plt.title("K-Means Clusters (2-D on P_Loss_Coeff and BMR/HMR)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    i+=1
