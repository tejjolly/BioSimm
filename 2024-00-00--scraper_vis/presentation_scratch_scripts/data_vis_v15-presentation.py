#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: tejjolly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.linear_model import LinearRegression

# -----------------------------
# Global matplotlib settings (fonts, ticks, etc.)
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

# -----------------------------
# 1) READ DATA
# -----------------------------
# summary_file = '/Users/tejjolly/Documents/BioSimm/Simulations/summary-HMR2.csv'     # Just HMR Runs
summary_file = '../../data/data.csv'          # All runs
df = pd.read_csv(summary_file)

# Convert columns to numeric, coercing errors to NaN
cols_to_num = [
    'CFR', 'P_d/P_a','BMR/HMR', 'R_total',
    'Stenosis Percentage', 'Length', 'HMR', 'HSR', 'P_Loss_Coeff'
]
for col in cols_to_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# OPTIONAL: If you want to automatically define BMR = (BMR/HMR) * HMR
if 'BMR' not in df.columns:
    df['BMR'] = df['BMR/HMR'] * df['HMR']

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
df['iFR'] = np.where(df['Condition'] == 'Non-hyperemic',
                          df['P_d/P_a'], np.nan)
df['FFR'] = np.where(df['Condition'] == 'Hyperemic',
                          df['P_d/P_a'], np.nan)

df = df[df['R_total']==0.24]
# df = df[df['BMR/HMR']>1.5]

df.to_csv('data_rtotal.csv',index=False)

# Some CSVs have a 'Condition' column (Hyperemic, Baseline, etc.). 
# If not, you can skip or adapt.

# -----------------------------
# 2) HELPER: ROUND STENOSIS
# -----------------------------
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """
    Round `val` to `decimal_places` and group it if within ±tolerance of that rounded value.
    Returns NaN if out of range, so it won't be grouped.
    """
    rounded_val = round(val, decimal_places)
    if abs(val - rounded_val) <= tolerance:
        return rounded_val
    else:
        return np.nan

# Create a new column 'Stenosis Group' based on the above logic
df['Stenosis Group'] = df['Stenosis Percentage'].apply(stenosis_group)

# -----------------------------
# 3) BUILD DICTIONARIES FOR MARKERS & LINE STYLES
# -----------------------------
# We'll assign each unique Stenosis Group a marker, and each unique Length a line style.

# Collect all unique Stenosis Group values (excluding NaN)
unique_sten_vals = sorted(df['Stenosis Group'].dropna().unique())
# A list of possible marker symbols to cycle through
# possible_markers = ['o', 's', '^', 'D', 'X', 'P', 'v', '>']
possible_markers = ['o']  # UNCOMMENT FOR SINGLE MARKER STYLE

markers_by_sten = {}
for i, st_val in enumerate(unique_sten_vals):
    markers_by_sten[st_val] = possible_markers[i % len(possible_markers)]

# Collect all unique Length values
unique_length_vals = sorted(df['Length'].dropna().unique())
# A list of possible line styles to cycle through
# possible_linestyles = ['solid','dotted','dashed','dashdot',(0,(5,5)),(0,(3,5,1,5))]
possible_linestyles = ['solid']  # UNCOMMENT FOR SINGLE LINESTYLE

linestyle_by_length = {}
for i, ln_val in enumerate(unique_length_vals):
    linestyle_by_length[ln_val] = possible_linestyles[i % len(possible_linestyles)]


def make_smart_scatter(
    data, x_col, y_col, color_col, 
    x_label, y_label, title,
    cmap_name='RdYlGn', 
    custom_boundaries=None,
    color_label='',
    add_threshold=None,    # list of dicts: e.g. [ {'axis':'x','value':0.8}, {'axis':'y','value':2.0} ]
    alpha_scatter=0.7,
    s_scatter=60,
    connect_stenosis_groups = False
):
    """
    data: DataFrame to plot
    x_col, y_col, color_col: names of columns for x, y, color
    x_label, y_label, title: strings for labeling
    cmap_name: name of matplotlib colormap
    custom_boundaries: optional array of color boundaries (e.g. np.linspace(min, max, 6))
    color_label: label for colorbar
    add_threshold: optional list of threshold lines, each item is:
                   {'axis': 'x' or 'y', 'value': val, 'style': '--', 'color': 'gray', 'width': 0.8}
    """
    plt.figure(figsize=(6, 4))
    
    # Filter out rows missing x, y, or color
    df_plot = data.copy()
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]
    
    # If needed, define boundaries from data min/max
    if custom_boundaries is None:
        cmin = df_plot[color_col].min()
        cmax = df_plot[color_col].max()
        # e.g. 5 intervals from min to max
        custom_boundaries = np.linspace(cmin, cmax, 6)
    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = plt.get_cmap(cmap_name)
    
    # For building the colorbar, we need a reference scatter
    first_scatter = None
    
    # Group by (Stenosis Group, Length)
    groups = df_plot.groupby(['Stenosis Group', 'Length'])
    
    for (sten_val, length_val), gdf in groups:
        # If either is NaN (some row might have no grouping info), skip
        if pd.isna(sten_val) or pd.isna(length_val):
            continue
        
        # Decide marker from Stenosis
        marker_style = markers_by_sten.get(sten_val, 'o')
        # Decide line style from Length
        ls = linestyle_by_length.get(length_val, 'solid')
        
        # Plot the scatter
        sc = plt.scatter(
            gdf[x_col],
            gdf[y_col],
            c=gdf[color_col],
            cmap=cmap,
            norm=norm,
            edgecolor='k',
            alpha=alpha_scatter,
            s=s_scatter,
            marker=marker_style,
            label=f"S={sten_val}, L={length_val}"
        )
        # Keep reference for colorbar
        if first_scatter is None:
            first_scatter = sc
        
        # connect_stenosis_groups = False  # Connecting line between constant
        #                                     # stenosis groups (varying HMR) on/off
        
        if connect_stenosis_groups:
            # If there's more than 1 point, connect them
            if len(gdf) > 1:
                gdf_sorted = gdf.sort_values(by=x_col)  # sort by x for a sensible line
                plt.plot(
                    gdf_sorted[x_col], 
                    gdf_sorted[y_col],
                    color='gray',
                    linestyle=ls,
                    alpha=0.5
                )
    
    # Add colorbar
    if first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label, fontsize=12)
    
    # Add threshold lines if desired
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')    # 'x' or 'y'
            value = tdict.get('value', 0.0)
            style = tdict.get('style', '--')
            c = tdict.get('color', 'gray')
            w = tdict.get('width', 0.8)
            
            if axis_type == 'y':
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([0.40343411323750916, 1.017823131798938])
    plt.ylim([1.0712680947042625, 3.910994942707241])
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    # plt.legend(loc='best')
    plt.show()

# ---------------------------------------------------------------------------------------
# 2) Plots
# ---------------------------------------------------------------------------------------

# PLOT 1) CFR vs FFR (P_d/P_a), colored by BMR/HMR
df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
df_filtered_cfr = df[(df['CFR'].notna()) & (df['P_d/P_a'].notna())]
nbins = 5
boundaries_cfr = np.linspace(1, 6, nbins+1)
make_smart_scatter(
    data=df_filtered_cfr,
    x_col='P_d/P_a', y_col='CFR', color_col='HMR',
    x_label='FFR', y_label='CFR',
    title='CFR vs FFR, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=boundaries_cfr,
    color_label='Microvascular Resistance [mmHg/cm/s]',
    add_threshold=[
        {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
        {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
    ]
)

# # PLOT 2) HMR vs HSR, colored by R_mult
# df_filtered_hmr = df[
#     (df['Condition'] == 'Hyperemic') &
#     df['HMR'].notna() &
#     df['HSR'].notna() &
#     df['P_d/P_a'].notna()
# ]
# boundaries = np.linspace(0.06, 0.81, 5)
# make_smart_scatter(
#     data=df_filtered_hmr,
#     x_col='HSR', y_col='HMR', color_col='R_total',
#     x_label='HSR [mmHg/cm/s]', y_label='HMR [mmHg/cm/s]',
#     title='HSR vs HMR, Colored by R. Multiplier',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries,
#     color_label='R. Multiplier'
# )

# # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
# df_filtered_third = df[
#     (df['Condition'] == 'Hyperemic') &
#     df['HMR'].notna() &
#     df['HSR'].notna() &
#     df['P_d/P_a'].notna()
# ]
# boundaries_third = np.linspace(0.2, 1.2, 6)
# make_smart_scatter(
#     data=df_filtered_third,
#     x_col='HMR', y_col='P_d/P_a', color_col='HSR',
#     x_label='HMR [mmHg/cm/s]', y_label='FFR',
#     title='FFR vs HMR, Colored by HSR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_third,
#     color_label='HSR [mmHg/cm/s]',
#     add_threshold=[{'axis': 'y', 'value': 0.8}]
# )
#
# # PLOT 4) FFR (P_d/P_a) vs HSR, colored by HMR
# df_filtered_fourth = df_filtered_third  # same filter
# boundaries_fourth = np.linspace(1, 6, 6)
# make_smart_scatter(
#     data=df_filtered_fourth,
#     x_col='HSR', y_col='P_d/P_a', color_col='HMR',
#     x_label='HSR [mmHg/cm/s]', y_label='FFR',
#     title='FFR vs HSR, Colored by HMR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_fourth,
#     color_label='HMR [mmHg/cm/s]',
#     add_threshold=[{'axis': 'y', 'value': 0.8}]
# )
#
# # PLOT 5) CFR vs HMR, colored by HSR
# df_filtered_cfr_hmr = df[
#     df['CFR'].notna() &
#     df['HMR'].notna() &
#     df['HSR'].notna()
# ]
# boundaries_cfr_hmr = np.linspace(0.1, 1.1, 6)
# make_smart_scatter(
#     data=df_filtered_cfr_hmr,
#     x_col='HMR', y_col='CFR', color_col='HSR',
#     x_label='HMR [mmHg/cm/s]', y_label='CFR',
#     title='CFR vs HMR, Colored by HSR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_cfr_hmr,
#     color_label='HSR [mmHg/cm/s]',
#     add_threshold=[{'axis': 'y', 'value': 2.0}]
# )
#
# # PLOT 6) HMR/(HMR+HSR) vs HMR, colored by HSR
# df_filtered_cfr_hmr = df[
#     df['CFR'].notna() &
#     df['HMR'].notna() &
#     df['HSR'].notna()
# ]
# boundaries_cfr_hmr = np.linspace(0.1, 1.1, 6)
# make_smart_scatter(
#     data=df_filtered_cfr_hmr,
#     x_col='FFR', y_col='ash', color_col='HSR',
#     x_label='FFR', y_label='ash',
#     title='CFR vs HMR, Colored by HSR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_cfr_hmr,
#     color_label='HSR [mmHg/cm/s]',
#     add_threshold=[{'axis': 'x', 'value': 2.5}]
# )
