#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: your_name
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
    'font.size': 12,  # Base font size
    'axes.labelsize': 14,  # Axis label font size
    'axes.titlesize': 14,  # Title font size
    'xtick.labelsize': 12,  # X-tick label font size
    'ytick.labelsize': 12,  # Y-tick label font size
    'legend.fontsize': 8,  # Legend font size
    'figure.dpi': 600      # Higher DPI for crisp text
})

# --------------------------------------------------
# 1) READ DATA
# --------------------------------------------------
summary_file = '../../data/data.csv'
df = pd.read_csv(summary_file)

# Convert columns to numeric, coercing errors to NaN
cols_to_num = [
    'CFR', 'P_d/P_a', 'BMR/HMR', 'R_total',
    'Stenosis Percentage', 'Length', 'HMR', 'HSR', 'R_micro'
]
for col in cols_to_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Optional: define BMR if your CSV has BMR/HMR but no explicit BMR column
if 'BMR' not in df.columns and 'BMR/HMR' in df.columns and 'HMR' in df.columns:
    df['BMR'] = df['BMR/HMR'] * df['HMR']

# If you want only Hyperemic, you can uncomment:
df = df[df['Condition'] == 'Hyperemic'].copy()
# df = df[df['source'] == 'mine']

# --------------------------------------------------
# 2) EXCLUDE R_micro=0 EXCEPT FOR THE SPECIAL ROW
# --------------------------------------------------
# Identify that one special row: hyperemic, R_total≈0.24, R_micro=0
special_mask = (
    (df['Condition'] == 'Hyperemic') &
    (df['R_micro'] == 0) &
    df['R_total'].notna() &
    np.isclose(df['R_total'], 0.24)
)
df_special = df[special_mask].copy()

# Remove all rows that have R_micro=0 and are NOT the special row
# df = df[~((df['R_micro'] == 0) & ~special_mask)].copy()

# Re-append the special row (so it's in the data if it exists)
# This way, any geometry that has only 0 or includes other values
# can still keep that single row if it matches the special condition.
df = pd.concat([df, df_special], ignore_index=True)

# Round stenosis for tidiness
df['Stenosis Percentage'] = df['Stenosis Percentage'].round(2)

# Now we want only geometries that have >1 distinct R_micro
df['rmicro_unique_count'] = df.groupby(['Stenosis Percentage', 'Length'])['R_micro'].transform('nunique')
df = df[df['rmicro_unique_count'] > 1].copy()

# # # Now we want only geometries that have >1 distinct R_total
# df['rtotal_unique_count'] = df.groupby(['Stenosis Percentage', 'Length'])['R_total'].transform('nunique')
# df = df[df['rtotal_unique_count'] > 1].copy()
# --------------------------------------------------
# 3) HELPER: ROUND STENOSIS (Example usage)
# --------------------------------------------------
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """
    Round `val` to `decimal_places` and group it if within ±tolerance.
    Returns NaN if out of range, so it won't be grouped.
    """
    rounded_val = round(val, decimal_places)
    if abs(val - rounded_val) <= tolerance:
        return rounded_val
    else:
        return np.nan

# Create a new column 'Stenosis Group'
df['Stenosis Group'] = df['Stenosis Percentage'].apply(stenosis_group)

# --------------------------------------------------
# 4) DICTIONARIES FOR MARKERS & LINE STYLES
# --------------------------------------------------
unique_sten_vals = sorted(df['Stenosis Group'].dropna().unique())
possible_markers = ['o', 's', '^', 'D', 'X', 'P', 'v', '>']
# possible_markers = ['o']#, 's', '^', 'D', 'X', 'P', 'v', '>']


markers_by_sten = {}
for i, st_val in enumerate(unique_sten_vals):
    markers_by_sten[st_val] = possible_markers[i % len(possible_markers)]

unique_length_vals = sorted(df['Length'].dropna().unique())
print(f'stens: {unique_sten_vals}')
print(f'lengths: {unique_length_vals}')
possible_linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 5, 1, 5))]
# possible_linestyles = ['solid']#, 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 5, 1, 5))]


linestyle_by_length = {}
for i, ln_val in enumerate(unique_length_vals):
    linestyle_by_length[ln_val] = possible_linestyles[i % len(possible_linestyles)]

series_final = (df[['Stenosis Group', 'Length']]
                .dropna()                      # toss rows where stenosis_group is NaN
                .drop_duplicates()
                .sort_values(['Stenosis Group', 'Length']))

print("\n=== SERIES THAT WILL PLOT ===")
print(series_final.to_string(index=False))
print(f"\nTotal plotted series = {len(series_final)}")

# --------------------------------------------------
# 5) MAKE SMART SCATTER (MODIFIED TO CONNECT SERIES)
# --------------------------------------------------
def make_smart_scatter(
        data, x_col, y_col, color_col,
        x_label, y_label, title,
        cmap_name='RdYlGn',
        custom_boundaries=None,
        color_label='',
        add_threshold=None,
        alpha_scatter=0.7,
        s_scatter=60,
        connect_stenosis_groups=True
):
    """
    Similar to your original function, but we add an option to
    connect points within each geometry series with a line.
    """
    plt.figure(figsize=(6, 4))

    # Filter out rows missing x, y, or color
    df_plot = data.copy()
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]

    # If needed, define boundaries from data min/max
    if custom_boundaries is None:
        cmin = df_plot[color_col].min()
        cmax = df_plot[color_col].max()
        custom_boundaries = np.linspace(cmin, cmax, 6)
    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = plt.get_cmap(cmap_name)

    first_scatter = None

    # Group by geometry
    groups = df_plot.groupby(['Stenosis Group', 'Length'])

    for (sten_val, length_val), gdf in groups:
        if pd.isna(sten_val) or pd.isna(length_val):
            continue

        # Marker & line style
        marker_style = markers_by_sten.get(sten_val, 'o')
        ls = linestyle_by_length.get(length_val, 'solid')

        # Plot scatter
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
            # label=f"S={sten_val}, L={length_val}"
            # label = f"Sten.={sten_val}"

        )
        plt.scatter(
            [], [],  # no data
            c='k',  # black face color
            marker=marker_style,
            s=s_scatter,
            label=f"Stenosis: {(round(sten_val,1)*100):.0f}%"
        )
        if first_scatter is None:
            first_scatter = sc

        # Connect points by x_col if there's more than 1 row
        if connect_stenosis_groups and len(gdf) > 1:
            gdf_sorted = gdf.sort_values(by=x_col)
            plt.plot(
                gdf_sorted[x_col],
                gdf_sorted[y_col],
                color='gray',
                linestyle=ls,
                alpha=0.6
            )

    # Add colorbar if we have at least one scatter
    if first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label, fontsize=12)

    # Add threshold lines if desired
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')
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
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    # plt.legend(loc='best')

    # Print axis limits
    # print("Auto xlim:", plt.gca().get_xlim())
    # print("Auto ylim:", plt.gca().get_ylim())

    # plt.xlim([0.40343411323750916, 1.017823131798938])
    # plt.ylim([1.0712680947042625, 3.910994942707241])

    plt.show()

""""""""""""""""""""""""""""""""""""""

# ###############################################################################
# PLOT 1) CFR vs FFR (P_d/P_a), colored by BMR/HMR
###############################################################################
# Filter
df_filtered_cfr = df[(df['CFR'].notna()) & (df['P_d/P_a'].notna())]
# Some data might not have BMR/HMR at all. It's okay; they just won't show up as colored.
# Let’s define boundaries for BMR/HMR ~ your example: np.linspace(1, 3.5, 6)
boundaries_cfr = np.linspace(1, 3.5, 6)

make_smart_scatter(
    data=df_filtered_cfr,
    x_col='P_d/P_a', y_col='CFR', color_col='HMR',
    x_label='FFR', y_label='CFR',
    title='CFR vs FFR, Colored by HMR',
    cmap_name='RdYlGn_r',
    custom_boundaries=boundaries_cfr,
    color_label='HMR [mmHg/cm/s]',
    add_threshold=[
        {'axis':'y', 'value':2.0, 'style':'--', 'color':'gray', 'width':0.8},
        {'axis':'x', 'value':0.8, 'style':'--', 'color':'gray', 'width':0.8}
    ],
    connect_stenosis_groups=True

)

# ###############################################################################
# # PLOT 2) HSR vs HMR, colored by FFR (P_d/P_a)
# ###############################################################################
# # Usually you only plotted hyperemic data for HMR/HSR
# df_filtered_hmr = df[
#     (df['Condition'] == 'Hyperemic') &
#     df['HMR'].notna() &
#     df['HSR'].notna() &
#     df['P_d/P_a'].notna()
# ]
# # Example boundaries: from your code we had np.linspace(0.65, 0.9, 6)
# boundaries_hmr = np.linspace(0.65, 0.9, 6)
#
# make_smart_scatter(
#     data=df_filtered_hmr,
#     x_col='HMR', y_col='HSR', color_col='P_d/P_a',
#     x_label='HMR [mmHg/cm/s]', y_label='HSR [mmHg/cm/s]',
#     title='HSR vs HMR, Colored by FFR',
#     cmap_name='RdYlGn',
#     custom_boundaries=boundaries_hmr,
#     color_label='FFR',
#     connect_stenosis_groups=True
#
#     # no threshold lines here
# )

# ###############################################################################
# # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
# ###############################################################################
# df_filtered_third = df[
#     (df['Condition'] == 'Hyperemic') &
#     df['HMR'].notna() &
#     df['HSR'].notna() &
#     df['P_d/P_a'].notna() &
#     (df['R_micro'] == 0)
# ]
#
# # Boundaries from data (or specify custom). Let's do auto:
# hsr_min = df_filtered_third['HSR'].min()
# hsr_max = df_filtered_third['HSR'].max()
# boundaries_third = np.linspace(.1, 1.1, 6)
#
# make_smart_scatter(
#     data=df_filtered_third,
#     x_col='HMR', y_col='P_d/P_a', color_col='HSR',
#     x_label='HMR [mmHg/cm/s]', y_label='FFR',
#     title='FFR vs HMR, Colored by HSR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_third,
#     color_label='HSR [mmHg/cm/s]',
#     add_threshold=[{'axis':'y','value':0.8}],
#     connect_stenosis_groups=True
#
# )

# ###############################################################################
# # PLOT 4) FFR (P_d/P_a) vs HSR, colored by HMR
# ###############################################################################
# df_filtered_fourth = df_filtered_third  # same filter
#
# hmr_min = df_filtered_fourth['HMR'].min()
# hmr_max = df_filtered_fourth['HMR'].max()
# boundaries_fourth = np.linspace(1, 6, 6)
#
# make_smart_scatter(
#     data=df_filtered_fourth,
#     x_col='HSR', y_col='P_d/P_a', color_col='HMR',
#     x_label='HSR [mmHg/cm/s]', y_label='FFR',
#     title='FFR vs HSR, Colored by HMR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_fourth,
#     color_label='HMR [mmHg/cm/s]',
#     add_threshold=[{'axis':'y','value':0.8}],
#     connect_stenosis_groups=True
#
# )
#
# ###############################################################################
# # PLOT 5) CFR vs HMR, colored by HSR
# ###############################################################################
# df_filtered_cfr_hmr = df[
#     df['CFR'].notna() &
#     df['HMR'].notna() &
#     df['HSR'].notna()
# ]
#
# hsr_min_ch = df_filtered_cfr_hmr['HSR'].min()
# hsr_max_ch = df_filtered_cfr_hmr['HSR'].max()
# boundaries_cfr_hmr = np.linspace(.1, 1.1, 6)
#
# make_smart_scatter(
#     data=df_filtered_cfr_hmr,
#     x_col='HMR', y_col='CFR', color_col='HSR',
#     x_label='HMR [mmHg/cm/s]', y_label='CFR',
#     title='CFR vs HMR, Colored by HSR',
#     cmap_name='RdYlGn_r',
#     custom_boundaries=boundaries_cfr_hmr,
#     color_label='HSR [mmHg/cm/s]',
#     add_threshold=[{'axis':'y','value':2.0}],
#     connect_stenosis_groups=True
#
# )

