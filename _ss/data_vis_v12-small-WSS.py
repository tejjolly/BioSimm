#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 02:48:17 2025

@author: tejjolly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 02:26:37 2025

@author: YourName
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# -----------------------------
# Global matplotlib settings
# -----------------------------
plt.rcParams.update({
    'font.size': 12,            # Increase base font size
    'axes.labelsize': 14,       # Axis label font size
    'axes.titlesize': 14,       # Title font size
    'xtick.labelsize': 12,      # X-tick label font size
    'ytick.labelsize': 12,      # Y-tick label font size
    'legend.fontsize': 8,       # Legend font size
    'figure.dpi': 600           # Higher DPI for clearer text in smaller figure
})

# ------------------------------------------------------------------------
# 1) READ DATA
# ------------------------------------------------------------------------
summary_file = '/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary.csv'
df = pd.read_csv(summary_file)

# Convert necessary columns to numeric, coercing errors to NaN
cols_to_num = ['CFR/FFR', 'WSS_AREA_BIFUR', 'HMR']
for col in cols_to_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------------------------------------------------
# 2) SIMPLE PLOTTING FUNCTION
# ------------------------------------------------------------------------
def make_smart_scatter(
    data, x_col, y_col, color_col,
    x_label, y_label, title,
    cmap_name='RdYlGn_r',
    custom_boundaries=None,
    color_label='',
    add_threshold=None,
    alpha_scatter=0.7,
    s_scatter=60
):
    """
    Create a 2D scatter plot with color-coded data based on a numeric column.
    """
    from matplotlib.ticker import FormatStrFormatter  # Import for formatting

    plt.figure(figsize=(6, 4))

    # Filter rows missing x, y, or color
    df_plot = data.copy()
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]

    if df_plot.empty:
        print("No valid data points to plot (missing values or columns?).")
        return

    # Define color boundaries if none provided
    if custom_boundaries is None:
        cmin = df_plot[color_col].min()
        cmax = df_plot[color_col].max()
        custom_boundaries = np.linspace(cmin, cmax, 6)

    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = cm.get_cmap(cmap_name)

    sc = plt.scatter(
        df_plot[x_col],
        df_plot[y_col],
        c=df_plot[color_col],
        cmap=cmap,
        norm=norm,
        edgecolor='k',
        alpha=alpha_scatter,
        s=s_scatter
    )

    # Add colorbar
    cbar = plt.colorbar(sc, ticks=custom_boundaries)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # Set color bar to 1 decimal place
    if color_label:
        cbar.set_label(color_label, fontsize=12)

    # Optional threshold lines
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')
            value = tdict.get('value', 0.0)
            style = tdict.get('style', '--')
            color_ = tdict.get('color', 'gray')
            width_ = tdict.get('width', 1.0)
            if axis_type == 'y':
                plt.axhline(y=value, color=color_, linestyle=style, linewidth=width_)
            else:
                plt.axvline(x=value, color=color_, linestyle=style, linewidth=width_)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# ------------------------------------------------------------------------
# 1) CREATE THE PLOT: CFR/FFR vs. WSS_AREA_BIFUR, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_AREA_BIFUR', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR/FFR',
    color_col='HMR',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS Area Bifur, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_AREA_BIFUR (colored by HMR).")

# ------------------------------------------------------------------------
# 2) CREATE THE PLOT: CFR/FFR vs. WSS_AREA_BIFUR, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_AREA_BIFUR', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR/FFR',
    color_col='BMR/HMR',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS Area Bifur, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_AREA_BIFUR (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 3) CREATE THE PLOT: CFR/FFR vs. WSS_AREA_BIFUR, colored by Stenosis
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_AREA_BIFUR', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR/FFR',
    color_col='Stenosis Percentage',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS Area Bifur, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_AREA_BIFUR (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 4) CREATE THE PLOT: CFR vs. WSS_AREA_BIFUR, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_AREA_BIFUR', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR',
    color_col='HMR',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR',
    title='CFR vs. WSS Area Bifur, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_AREA_BIFUR (colored by HMR).")

# ------------------------------------------------------------------------
# 5) CREATE THE PLOT: CFR vs. WSS_AREA_BIFUR, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_AREA_BIFUR', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR',
    color_col='BMR/HMR',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR',
    title='CFR vs. WSS Area Bifur, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_AREA_BIFUR (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 6) CREATE THE PLOT: CFR vs. WSS_AREA_BIFUR, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_AREA_BIFUR', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='CFR',
    color_col='Stenosis Percentage',
    x_label='WSS_AREA_BIFUR',
    y_label='CFR',
    title='CFR vs. WSS Area Bifur, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_AREA_BIFUR (colored by Stenosis Percentage).")


# ------------------------------------------------------------------------
# 7) CREATE THE PLOT: HMR vs. WSS_AREA_BIFUR, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_AREA_BIFUR', 'CFR']].dropna()

# Create 5 bins => 6 boundaries for CFR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['CFR'].min()
    CFR_max = df_plot_me['CFR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='HMR',
    color_col='CFR',
    x_label='WSS_AREA_BIFUR',
    y_label='HMR',
    title='HMR vs. WSS Area Bifur, Colored by CFR',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='CFR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_AREA_BIFUR (colored by CFR).")

# ------------------------------------------------------------------------
# 8) CREATE THE PLOT: HMR vs. WSS_AREA_BIFUR, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_AREA_BIFUR', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['BMR/HMR'].min()
    CFR_max = df_plot_me['BMR/HMR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='HMR',
    color_col='BMR/HMR',
    x_label='WSS_AREA_BIFUR',
    y_label='HMR',
    title='HMR vs. WSS Area Bifur, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=CFR_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_AREA_BIFUR (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 9) CREATE THE PLOT: HMR vs. WSS_AREA_BIFUR, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_AREA_BIFUR', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['Stenosis Percentage'].min()
    CFR_max = df_plot_me['Stenosis Percentage'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_AREA_BIFUR',
    y_col='HMR',
    color_col='Stenosis Percentage',
    x_label='WSS_AREA_BIFUR',
    y_label='HMR',
    title='HMR vs. WSS Area Bifur, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_AREA_BIFUR (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 10) CREATE THE PLOT: CFR/FFR vs. WSS_LE, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_LE', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR/FFR',
    color_col='HMR',
    x_label='WSS_LE',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Plaque LE, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_LE (colored by HMR).")

# ------------------------------------------------------------------------
# 11) CREATE THE PLOT: CFR/FFR vs. WSS_LE, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_LE', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR/FFR',
    color_col='BMR/HMR',
    x_label='WSS_LE',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Plaque LE, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_LE (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 12) CREATE THE PLOT: CFR/FFR vs. WSS_LE, colored by Stenosis
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_LE', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR/FFR',
    color_col='Stenosis Percentage',
    x_label='WSS_LE',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Plaque LE, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_LE (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 13) CREATE THE PLOT: CFR vs. WSS_LE, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_LE', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR',
    color_col='HMR',
    x_label='WSS_LE',
    y_label='CFR',
    title='CFR vs. WSS @ Plaque LE, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_LE (colored by HMR).")

# ------------------------------------------------------------------------
# 14) CREATE THE PLOT: CFR vs. WSS_LE, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_LE', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR',
    color_col='BMR/HMR',
    x_label='WSS_LE',
    y_label='CFR',
    title='CFR vs. WSS @ Plaque LE, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_LE (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 15) CREATE THE PLOT: CFR vs. WSS_LE, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_LE', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='CFR',
    color_col='Stenosis Percentage',
    x_label='WSS_LE',
    y_label='CFR',
    title='CFR vs. WSS @ Plaque LE, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_LE (colored by Stenosis Percentage).")


# ------------------------------------------------------------------------
# 16) CREATE THE PLOT: HMR vs. WSS_LE, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_LE', 'CFR']].dropna()

# Create 5 bins => 6 boundaries for CFR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['CFR'].min()
    CFR_max = df_plot_me['CFR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='HMR',
    color_col='CFR',
    x_label='WSS_LE',
    y_label='HMR',
    title='HMR vs. WSS @ Plaque LE, Colored by CFR',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='CFR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_LE (colored by CFR).")

# ------------------------------------------------------------------------
# 17) CREATE THE PLOT: HMR vs. WSS_LE, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_LE', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['BMR/HMR'].min()
    CFR_max = df_plot_me['BMR/HMR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='HMR',
    color_col='BMR/HMR',
    x_label='WSS_LE',
    y_label='HMR',
    title='HMR vs. WSS @ Plaque LE, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=CFR_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_LE (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 18) CREATE THE PLOT: HMR vs. WSS_LE, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_LE', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['Stenosis Percentage'].min()
    CFR_max = df_plot_me['Stenosis Percentage'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_LE',
    y_col='HMR',
    color_col='Stenosis Percentage',
    x_label='WSS_LE',
    y_label='HMR',
    title='HMR vs. WSS @ Plaque LE, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_LE (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 19) CREATE THE PLOT: CFR/FFR vs. WSS_BIF, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_BIF', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR/FFR',
    color_col='HMR',
    x_label='WSS_BIF',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Bifurcation, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_BIF (colored by HMR).")

# ------------------------------------------------------------------------
# 20) CREATE THE PLOT: CFR/FFR vs. WSS_BIF, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_BIF', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR/FFR',
    color_col='BMR/HMR',
    x_label='WSS_BIF',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Bifurcation, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_BIF (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 21) CREATE THE PLOT: CFR/FFR vs. WSS_BIF, colored by Stenosis
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'WSS_BIF', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR/FFR',
    color_col='Stenosis Percentage',
    x_label='WSS_BIF',
    y_label='CFR/FFR',
    title='CFR/FFR vs. WSS @ Bifurcation, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. WSS_BIF (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 22) CREATE THE PLOT: CFR vs. WSS_BIF, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_BIF', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR',
    color_col='HMR',
    x_label='WSS_BIF',
    y_label='CFR',
    title='CFR vs. WSS @ Bifurcation, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_BIF (colored by HMR).")

# ------------------------------------------------------------------------
# 23) CREATE THE PLOT: CFR vs. WSS_BIF, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_BIF', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['BMR/HMR'].min()
    hmr_max = df_plot_me['BMR/HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR',
    color_col='BMR/HMR',
    x_label='WSS_BIF',
    y_label='CFR',
    title='CFR vs. WSS @ Bifurcation, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=hmr_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_BIF (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 24) CREATE THE PLOT: CFR vs. WSS_BIF, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR', 'WSS_BIF', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['Stenosis Percentage'].min()
    hmr_max = df_plot_me['Stenosis Percentage'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='CFR',
    color_col='Stenosis Percentage',
    x_label='WSS_BIF',
    y_label='CFR',
    title='CFR vs. WSS @ Bifurcation, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR vs. WSS_BIF (colored by Stenosis Percentage).")


# ------------------------------------------------------------------------
# 25) CREATE THE PLOT: HMR vs. WSS_BIF, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_BIF', 'CFR']].dropna()

# Create 5 bins => 6 boundaries for CFR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['CFR'].min()
    CFR_max = df_plot_me['CFR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='HMR',
    color_col='CFR',
    x_label='WSS_BIF',
    y_label='HMR',
    title='HMR vs. WSS @ Bifurcation, Colored by CFR',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='CFR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_BIF (colored by CFR).")

# ------------------------------------------------------------------------
# 26) CREATE THE PLOT: HMR vs. WSS_BIF, colored by BMR/HMR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_BIF', 'BMR/HMR']].dropna()

# Create 5 bins => 6 boundaries for BMR/HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['BMR/HMR'].min()
    CFR_max = df_plot_me['BMR/HMR'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='HMR',
    color_col='BMR/HMR',
    x_label='WSS_BIF',
    y_label='HMR',
    title='HMR vs. WSS @ Bifurcation, Colored by BMR/HMR',
    cmap_name='RdYlGn',
    custom_boundaries=CFR_boundaries,
    color_label='BMR/HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_BIF (colored by BMR/HMR).")

# ------------------------------------------------------------------------
# 27) CREATE THE PLOT: HMR vs. WSS_BIF, colored by CFR
# ------------------------------------------------------------------------
df_plot_me = df[['HMR', 'WSS_BIF', 'Stenosis Percentage']].dropna()

# Create 5 bins => 6 boundaries for Stenosis Percentage
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    CFR_min = df_plot_me['Stenosis Percentage'].min()
    CFR_max = df_plot_me['Stenosis Percentage'].max()
    CFR_boundaries = np.linspace(CFR_min, CFR_max, 6)
else:
    CFR_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='WSS_BIF',
    y_col='HMR',
    color_col='Stenosis Percentage',
    x_label='WSS_BIF',
    y_label='HMR',
    title='HMR vs. WSS @ Bifurcation, Colored by Stenosis',
    cmap_name='RdYlGn_r',
    custom_boundaries=CFR_boundaries,
    color_label='Stenosis Percentage',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: HMR vs. WSS_BIF (colored by Stenosis Percentage).")

# ------------------------------------------------------------------------
# 28) CREATE THE PLOT: CFR/FFR vs. WSS_AREA_BIFUR, colored by HMR
# ------------------------------------------------------------------------
df_plot_me = df[['CFR/FFR', 'Average Flow', 'HMR']].dropna()

# Create 5 bins => 6 boundaries for HMR
# You can customize the min/max if you already know a typical range
if not df_plot_me.empty:
    hmr_min = df_plot_me['HMR'].min()
    hmr_max = df_plot_me['HMR'].max()
    hmr_boundaries = np.linspace(hmr_min, hmr_max, 6)
else:
    hmr_boundaries = [0, 1, 2, 3, 4, 5]  # Fallback if no data

# Plot
make_smart_scatter(
    data=df,
    x_col='Average Flow',
    y_col='CFR/FFR',
    color_col='HMR',
    x_label='Flow',
    y_label='CFR/FFR',
    title='CFR/FFR vs. Flow, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=hmr_boundaries,
    color_label='HMR',
    add_threshold=[{'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 1.0}]
)

print("Done plotting: CFR/FFR vs. Flow (colored by HMR).")