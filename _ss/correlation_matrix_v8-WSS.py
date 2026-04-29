#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:43:19 2025

@author: tejjolly
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from scipy.stats import pearsonr
import matplotlib.colors as mcolors

pairgrid_switch = False

# Load the data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary.csv')

n_bins = 5
cmap = plt.get_cmap('Reds', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Keep 'Condition' in string form (not numeric codes)
# We'll use it to split P_d/P_a into iFR or FFR
# e.g. Condition: "Non-hyperemic" or "Hyperemic"
df_full['Condition'] = df_full['Condition'].astype(str)

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
# Rows that aren't of that type get NaN in that column
df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic',
                          df_full['P_d/P_a'], np.nan)
df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic',
                          df_full['P_d/P_a'], np.nan)

# Optional: drop the original 'P_d/P_a' column, as it's now split
df_full.drop(columns=['P_d/P_a'], inplace=True)

# If you still don't need 'Condition' or 'Geometry Number' for the correlation/p-value,
# you can drop them. (If you do want them, skip dropping or adjust as needed.)
df_full.drop(columns=['Condition', 'Geometry Number'], inplace=True)

# Now rename other columns as before
df_full.rename(columns={
    'Stenosis Percentage': 'Stenosis',
    'Average Flow': 'Flow',
    'Rtotal_cor Value': 'R. mult.',
    'WSS_Area_Bifur': 'WSS_Bif_Area',

}, inplace=True)

    
columns_of_interest = [
    # WSS variables in the order you specified:
    'WSS_LMB',
    'WSS', 
    'WSS_TE', 
    'WSS_LE', 
    'WSS_Bif', 
    'WSS_Avg_Area',
    'WSS_TE_Area', 
    'WSS_LE_Area', 
    'WSS_Bif_Area',
    'Flow',
    ## target variables
    'HMR', 
    'BMR/HMR', 
    'iFR', 
    'FFR', 
    'CFR', 
    'CFR/FFR',
    'HSR'
]

# Filter out columns that might not exist in df_full
existing_cols = [col for col in columns_of_interest if col in df_full.columns]

# Subset the DataFrame to just these columns
df_subset = df_full[existing_cols]

# Convert to numeric if needed (coerce any weird strings to NaN)
df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

df_subset["WSS_LMB"] = df_subset["WSS_LMB"].replace(0, np.nan)
df_subset = df_subset.dropna(subset=["WSS_LMB"])

for col in ["WSS", "WSS_TE", "WSS_LE", "WSS_Bif"]:
    df_subset[col] = df_subset[col] / df_subset["WSS_LMB"]

vars_to_plot = [
    # 'WSS_LMB',
    'WSS_Bif', 
    'WSS_LE', 
    'WSS_TE', 
    'WSS', 
    'WSS_Bif_Area',
    'WSS_LE_Area', 
    'WSS_TE_Area', 
    'WSS_Avg_Area',
    'Flow',
    # target variables
    'HMR', 
    'BMR/HMR', 
    'iFR', 
    'FFR', 
    'CFR', 
    'CFR/FFR'
    ]

# 2) Drop rows where HSR is NaN
df_plot = df_subset.dropna(subset=["HSR"])

if pairgrid_switch:
    # 3) Build the PairGrid
    g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)    
    
    # 4) Helper for scatter with color
    def scatter_cmap(x, y, c=None, cmap=None, norm=None, scale = 1, **kwargs, ):
        # Remove seaborn’s default color
        if "color" in kwargs:
            kwargs.pop("color")
            
        plt.rcParams.update({
        "font.size": 12 * scale,  # Adjust global font size
        "axes.labelsize": 14, #* scale,  # Axis labels
        "xtick.labelsize": 12, #* scale,  # X-axis tick labels
        "ytick.labelsize": 12 #* scale,  # Y-axis tick labels
        })
        
        plt.scatter(x, y, c=c, cmap=cmap, norm=norm, **kwargs)
    
    # --- Discrete bins setup for HSR ---
    # Find min/max of HSR, create 5 bins => 6 boundaries
    min_val = df_plot["HSR"].min()
    max_val = df_plot["HSR"].max()
    boundaries = np.linspace(min_val, max_val, 6)  # 6 edges -> 5 bins
    
    # We can use any built-in colormap; 'RdYlGn' with 5 discrete levels:
    cmap = plt.get_cmap("RdYlGn_r", 5)
    
    # Create a BoundaryNorm so each bin is mapped to one color
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    
    scale_plot = 1.9
    
    # 5) Map your custom scatter
    g.map_lower(scatter_cmap, c=df_plot["HSR"], cmap=cmap, norm=norm, edgecolor="k", scale = scale_plot)
    g.map_diag(sns.histplot, fill=True)
    
    # 6) Add the discrete colorbar
    fig = g.fig
    cax = fig.add_axes([.975, 0.3, 0.02, 0.4])  # x, y, width, height (in figure fraction)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy array for older mpl versions
    cbar = fig.colorbar(sm, cax=cax)
        
    # Optionally set bin-edge ticks explicitly:
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])  # or custom labels
    
    scale_cbar = 4
    
    cbar.ax.tick_params(labelsize=12 * scale_cbar)  # Change colorbar tick font size
    cbar.set_label("HSR [mmHg/cm/s]", fontsize=14 * scale_cbar)  # Change colorbar label font size

    
    plt.show()

"""INDIVIDUAL PLOT CREATOR"""""""""""""""""""""""""""""""""""
def discrete_color_scatter(
    df,
    x_col,
    y_col,
    cbar="HSR",
    bins=5,
    cmap_name="RdYlGn_r",
    cbar_label="HSR [mmHg/cm/s]",
    marker="o",
    edgecolor="k",
    alpha=0.8,
    figsize=(5,4),
    text_size=9,
    cbar_text_size=9,
    h_line=0,
    v_line=0
):
    """
    Make a scatter plot of df[x_col] vs df[y_col], colored by df[color_col],
    using a discrete colormap with `bins` bins. A colorbar is added on the right.

    Parameters
    ----------
    df : pandas.DataFrame
        Data source.
    x_col : str
        Name of column for x-axis.
    y_col : str
        Name of column for y-axis.
    cbar : str, optional
        Name of the numeric column to map to discrete color bins, by default "HSR".
    bins : int, optional
        Number of color bins, by default 5.
    cmap_name : str, optional
        Name of the colormap, by default "RdYlGn_r".
    cbar_label : str, optional
        Label for the colorbar, by default "HSR [mmHg/cm/s]".
    marker : str, optional
        Marker style for scatter, by default "o".
    edgecolor : str, optional
        Edge color for each scatter point, by default "k" (black).
    alpha : float, optional
        Marker transparency, by default 0.8.
    figsize : tuple, optional
        Size of the figure (width, height) in inches, by default (5,4).
    text_size : int, optional
        Font size for axis labels and title, by default 12.
    cbar_text_size : int, optional
        Font size for colorbar labels and ticks, by default 12.
    h_line : float, optional
        If nonzero, plots a horizontal dotted line at the specified value, by default 0.
    v_line : float, optional
        If nonzero, plots a vertical dotted line at the specified value, by default 0.
    
    Returns
    -------
    None
        Displays the plot.
    """
    # Drop rows without valid x, y, or color
    df_plot = df.dropna(subset=[x_col, y_col, cbar]).copy()
    
    if df_plot.empty:
        print(f"No valid data to plot for {x_col}, {y_col}, {cbar}.")
        return

    # Figure creation
    fig, ax = plt.subplots(figsize=figsize)

    # Determine min/max of the color variable to set bin boundaries
    cmin = df_plot[cbar].min()
    cmax = df_plot[cbar].max()
    boundaries = np.linspace(cmin, cmax, bins + 1)  # e.g. 6 boundaries => 5 bins

    # Create colormap with discrete bins
    cmap = plt.get_cmap(cmap_name, bins) 
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    # Scatter plot using discrete colormap
    sc = ax.scatter(
        df_plot[x_col],
        df_plot[y_col],
        c=df_plot[cbar],
        cmap=cmap,
        norm=norm,
        marker=marker,
        edgecolors=edgecolor,
        alpha=alpha
    )

    # Set axis labels and title with adjustable font size
    ax.set_xlabel(x_col, fontsize=text_size)
    ax.set_ylabel(y_col, fontsize=text_size)
    # ax.set_title(f"{y_col} vs {x_col}", fontsize=text_size)
    
    # Add optional horizontal and vertical lines
    if h_line:
        ax.axhline(h_line, color='gray', linestyle='dotted', linewidth=1)
    if v_line:
        ax.axvline(v_line, color='gray', linestyle='dotted', linewidth=1)

    # Add discrete colorbar
    cbar = fig.colorbar(sc, ax=ax, spacing="proportional")
    cbar.set_label(cbar_label, fontsize=cbar_text_size)
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])
    
    # Adjust colorbar font size
    cbar.ax.tick_params(labelsize=cbar_text_size)
    
    plt.tight_layout()
    plt.show()


# # Create plots
ys = ['CFR','CFR/FFR']
cols = ['HSR','HMR', 'BMR/HMR']
for y in ys:
    for col in cols:
        
        discrete_color_scatter(df_plot,
                               x_col="WSS_Bif_Area",
                               y_col=y, 
                               cbar = col,
                               cbar_label = f"{col}{' [mmHg/cm/s]' if col == 'HSR' or col == 'HMR' else ''}",
                               h_line=2, 
                               figsize=(6,4))
        # discrete_color_scatter(df_plot, x_col="WSS", y_col="WSS_TE", color_col="HSR", bins=5)
