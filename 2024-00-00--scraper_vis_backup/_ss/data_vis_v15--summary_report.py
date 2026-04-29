#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: tejjolly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.patheffects as pe


log_switch = False

plot1   = False # CFR vs FFR, col BMR/HMR
plot1_5 = False  # CFR vs FFR, col HMR
plot2   = False # HMR vs HSR, col FFR
plot3   = True # FFR vs HMR, col HSR
plot35  = False # FFR vs BMR/HMR, col HSR
plot36  = False # FFR vs BMR/HMR, col HSR
plot4   = False # FFR vs HSR, col HMR
plot45  = False # FFR vs HSR, col BMR/HMR
plot5   = True # CFR vs HMR, col HSR
plot55  = False # CFR vs BMR/HMR, col HSR
plot56  = False # CFR vs HSR, col HMR
plot57  = False # CFR vs HSR, col BMR/HMR
plot6   = False # Janky plot
plot7   = False # 7-10 are 2-5 w/ P_Loss_Coeff instead of HSR
plot8   = False
plot9   = False
plot10  = False
plot11  = False  # Area Low LE WSS
plot12  = False  # Area Low TE WSS
plot13  = False  # Area Low Bif WSS
plot14  = False  # Area High LE WSS
plot15  = False  # Area High TE WSS
plot16  = False  # Area High Bif WSS
plot17  = False # FFR vs. v_distal, colored by HSR

# -----------------------------
# Global matplotlib settings (fonts, ticks, etc.)
# -----------------------------
original_settings = False
### Originals:
if original_settings:
    plt.rcParams.update({
        'font.size': 12,            # Increase base font size
        'axes.labelsize': 14,       # Axis label font size
        'axes.titlesize': 14,       # Title font size
        'xtick.labelsize': 12,      # X-tick label font size
        'ytick.labelsize': 12,      # Y-tick label font size
        'legend.fontsize': 8,       # Legend font size
        'figure.dpi': 600           # Higher DPI for clearer text in smaller figure
    })

else:
    plt.rcParams.update({
        'font.size': 20,            # Increase base font size
        'axes.labelsize': 18,       # Axis label font size
        'axes.titlesize': 18,       # Title font size
        'xtick.labelsize': 18,      # X-tick label font size
        'ytick.labelsize': 18,      # Y-tick label font size
        'legend.fontsize': 18,       # Legend font size
        'figure.dpi': 600           # Higher DPI for clearer text in smaller figure
    })

# -----------------------------
# 1) READ DATA
# -----------------------------
data_file = '../../data/data.csv'
df = pd.read_csv(data_file)

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

def make_smart_scatter(
    data, x_col, y_col, color_col,
    x_label, y_label, title,
    cmap_name='RdYlGn',
    custom_boundaries=None,
    color_label='',
    add_threshold=None,
    alpha_scatter=0.7,
    s_scatter=60,
    connect_stenosis_groups=False,
    show_singletons=True,
    savefig=False,
    dpi=600,
    dir = 'images',
    labels = False
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
    # if original_settings:
    # plt.figure(figsize=(6, 4))
    # else:
    plt.figure(figsize=(7, 4))

    # Filter out rows missing x, y, or color
    df_plot = data.copy()
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]

    # Copy + drop missing
    df_plot = data.copy()
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]

    # Filter out singleton groups if needed
    if not show_singletons:
        counts = df_plot.groupby(['Stenosis Group', 'Length']).size()
        valid_groups = counts[counts > 1].index
        df_plot = df_plot.set_index(['Stenosis Group', 'Length']).loc[valid_groups].reset_index()

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

    series_to_plot = (
        df_plot[['Stenosis Group', 'Length']]  # keep the two grouping columns
        .dropna()  # throw out rows missing either value
        .drop_duplicates()  # unique combos only
        .sort_values(['Stenosis Group', 'Length'])
    )

    ## START CHECKER
    print("\n UNIQUE SERIES THAT WILL PLOT  "
          f"(total = {len(series_to_plot)})")
    print(series_to_plot.to_string(index=False))
    ## END CHECKER

    # Group by (Stenosis Group, Length)
    groups = df_plot.groupby(['Stenosis Group', 'Length'])

    unique_sten_vals = sorted(df_plot['Stenosis Group'].dropna().unique())
    print(f'unique_sten_groups: {unique_sten_vals}')
    possible_markers = ['o', 's', '^', 'D', 'X', 'P', 'v', '>']   # or extend
    # possible_markers = ['o']  # UNCOMMENT FOR SINGLE MARKER STYLE

    markers_by_sten = {
        s: possible_markers[i % len(possible_markers)]
        for i, s in enumerate(unique_sten_vals)
    }

    unique_length_vals = sorted(df_plot['Length'].dropna().unique())
    possible_linestyles = ['solid','dotted','dashed','dashdot',
                           (0,(5,5)),(0,(3,5,1,5))]
    # possible_linestyles = ['solid']  # UNCOMMENT FOR SINGLE LINESTYLE

    linestyle_by_length = {
        l: possible_linestyles[i % len(possible_linestyles)]
        for i, l in enumerate(unique_length_vals)
    }

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

        # --- NEW: add Geometry-Number labels ------------------------
        if labels:
            for xi, yi, lab in zip(gdf[x_col], gdf[y_col], gdf['Geometry Number']):
                plt.text(xi, yi, str(lab),
                         fontsize=8, ha='right', va='top',
                         path_effects=[pe.withStroke(linewidth=1.5,
                                                     foreground='white')])

        # Keep reference for colorbar
        if first_scatter is None:
            first_scatter = sc

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
            if original_settings:
                cbar.set_label(color_label, fontsize=12)
            else:
                cbar.set_label(color_label)
    
    # Add threshold lines if desired
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')    # 'x' or 'y'
            value = tdict.get('value', 0.0)
            style = tdict.get('style', '--')
            c = tdict.get('color', 'gray')
            w = tdict.get('width', 2.5)
            
            if axis_type == 'y':
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    # plt.legend(loc='best')

    if savefig:
        if savefig is True:
            if x_col == 'P_d/P_a':
                fname = f"{y_col}_vs_FFR_col_{color_col}"
            elif y_col == 'P_d/P_a':
                fname = f"FFR_vs_{x_col}_col_{color_col}"
            else:
                fname = f"{y_col}_vs_{x_col}_col_{color_col}"
        else:
            fname = str(savefig)

        plt.savefig(
            f'{dir}/{fname}.png',
            dpi=dpi,
            transparent=True,
            bbox_inches='tight'
        )
        plt.savefig(
            f'{dir}/{fname}.svg',
            transparent=True,
            bbox_inches='tight'
        )
        print(f"saved → {savefig}")


    plt.show()
    plt.close()

    n_samples = len(df_plot)
    print(f"{title or f'{y_col} vs {x_col}'}: {n_samples} points")
    if n_samples == 0:
        print("Nothing to plot – aborting.")
        return


if plot1:
    # PLOT 1) CFR vs FFR (P_d/P_a), colored by BMR/HMR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['CFR'].notna()) &
        (df['P_d/P_a'].notna()) &
        (df['source'] == 'mine')]
    boundaries_cfr = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='P_d/P_a', y_col='CFR', color_col='BMR/HMR',
        x_label='FFR', y_label='CFR',
        title='CFR vs FFR, Colored by BMR/HMR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_cfr,
        color_label='BMR/HMR',
        add_threshold=[
            {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
            {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=True,
        show_singletons=True
    )

# PLOT 1.5.....) CFR vs FFR (P_d/P_a), colored by HMR
if plot1_5:
    df_filtered_cfr = df[(df['CFR'].notna()) &
                         (df['P_d/P_a'].notna()) &
                         (df['source'] == 'mine') &
                         (df['Location'] == 'LAD') &
                         (df['R_micro'] == 0) #&
                         # (df['Geometry Number'] < 100)
                         ]
    boundaries_cfr = np.linspace(1, 7, 5)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='P_d/P_a', y_col='CFR', color_col='HMR',
        x_label='FFR', y_label='CFR',
        title='',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[
            {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray'},
            {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray'}
        ],
        connect_stenosis_groups=True,
        show_singletons=True,
        savefig=True,
        labels = False
    )

if plot2:
# PLOT 2) HMR vs HSR, colored by R_mult
    df_filtered_hmr = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
    ]
    boundaries = np.linspace(0.5, 1, 5)
    make_smart_scatter(
        data=df_filtered_hmr,
        x_col='HSR', y_col='HMR', color_col='P_d/P_a',
        x_label='HSR [mmHg/cm/s]', y_label='HMR [mmHg/cm/s]',
        title='HSR vs HMR, Colored by FFR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries,
        color_label='FFR'
    )

if plot3:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0) &
        ~(df['Stenosis Group'].round(2) == 0.48) &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HMR', y_col='P_d/P_a', color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        savefig=True
    )

if plot35:
    # PLOT 3.5) FFR (P_d/P_a) vs BMR/HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries_third = np.linspace(0.2, 1.2, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='BMR/HMR', y_col='P_d/P_a', color_col='HSR',
        x_label='BMR/HMR', y_label='FFR',
        title='FFR vs BMR/HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot36:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['v_distal'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0, 50, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HMR', y_col='P_d/P_a', color_col='v_distal',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by Distal Velocity',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_third,
        color_label='Distal velocity [cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot4:
    # PLOT 4) FFR (P_d/P_a) vs HSR, colored by HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries_fourth = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HSR', y_col='P_d/P_a', color_col='HMR',
        x_label='HSR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HSR, Colored by HMR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_fourth,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot45:
    # PLOT 4.5) FFR (P_d/P_a) vs HSR, colored by BMR/HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries_fourth = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HSR', y_col='P_d/P_a', color_col='BMR/HMR',
        x_label='HSR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HSR, Colored by BMR/HMR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_fourth,
        color_label='BMR/HMR',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot5:
    # PLOT 5) CFR vs HMR, colored by HSR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['HSR'].notna() &
        (df['R_micro'] == 0) &
        ~(df['Stenosis Group'].round(2) == 0.48) &
        (df['Location'] == 'LAD')
    ]
    boundaries_cfr_hmr = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HMR', y_col='CFR', color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
        connect_stenosis_groups=True,
        show_singletons=False,
        savefig=True
    )

if plot55:
    # PLOT 5.5) CFR vs BMR/HMR, colored by HSR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['BMR/HMR'].notna() &
        df['HSR'].notna()
    ]
    boundaries_cfr_hmr = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='BMR/HMR', y_col='CFR', color_col='HSR',
        x_label='BMR/HMR', y_label='CFR',
        title='CFR vs BMR/HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}]
    )

if plot56:
    # PLOT 5.6) CFR vs HSR, colored by HMR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['HSR'].notna()
    ]
    boundaries_cfr_hmr = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HSR', y_col='CFR', color_col='HMR',
        x_label='HSR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HSR, Colored by HMR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}]
    )

if plot57:
    # PLOT 5.6) CFR vs HSR, colored by BMR/HMR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['BMR/HMR'].notna() &
        df['HSR'].notna()
    ]
    boundaries_cfr_hmr = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HSR', y_col='CFR', color_col='BMR/HMR',
        x_label='HSR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HSR, Colored by BMR/HMR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='BMR/HMR',
        add_threshold=[{'axis': 'y', 'value': 2.0}]
    )

if plot6:
    # PLOT 6) HMR/(HMR+HSR) vs HMR, colored by HSR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['HSR'].notna()
    ]
    boundaries_cfr_hmr = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='FFR', y_col='ash', color_col='HSR',
        x_label='FFR', y_label='ash',
        title='CFR vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'x', 'value': 2.5}]
    )

if plot7:
# PLOT 2) HMR vs P_Loss_Coeff, colored by FFR
    df_filtered_hmr = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna()
    ]
    boundaries = np.linspace(0.5, 1, 5)
    make_smart_scatter(
        data=df_filtered_hmr,
        x_col='P_Loss_Coeff', y_col='HMR', color_col='P_d/P_a',
        x_label='$log_{10}(ζ_{L})$', y_label='HMR [mmHg/cm/s]',
        title='$log_{10}(ζ_{L})$ vs HMR, Colored by FFR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries,
        color_label='FFR'
    )

if plot8:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by P_Loss_Coeff
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna()
    ]
    boundaries_third = np.linspace(0.5, 2.5, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HMR', y_col='P_d/P_a', color_col='P_Loss_Coeff',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by $log_{10}(ζ_{L})$',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='$log_{10}(ζ_{L})$',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot9:
    # PLOT 4) FFR (P_d/P_a) vs P_Loss_Coeff, colored by HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0)
    ]
    boundaries_fourth = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='P_Loss_Coeff', y_col='P_d/P_a', color_col='HMR',
        x_label='$log_{10}(ζ_{L})$', y_label='FFR',
        title='FFR vs P_Loss_Coeff, Colored by HMR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_fourth,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups = True,
        show_singletons = False
    )

if plot10:
    # PLOT 5) CFR vs HMR, colored by P_Loss_Coeff
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna()
    ]
    boundaries_cfr_hmr = np.linspace(0.5, 2.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HMR', y_col='CFR', color_col='P_Loss_Coeff',
        x_label='HMR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HMR, Colored by $log_{10}(ζ_{L})$',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='$log_{10}(ζ_{L})$',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
    )

if plot11:
    wss_var = 'WSS_LE_Area_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area LE WSS < 0.5 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot12:
    wss_var = 'WSS_TE_Area_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area TE WSS < 0.5 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot13:
    wss_var = 'WSS_Area_Bifur_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area Bifurcation WSS < 0.5 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot14:
    wss_var = 'WSS_LE_Area'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area LE WSS > 7 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot15:
    wss_var = 'WSS_TE_Area'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area TE WSS > 7 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot16:
    wss_var = 'WSS_Area_Bifur'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna() &
        (df[wss_var] != 0) &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area Bifurcation WSS > 7 Pa vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )
    
if plot17:
    # PLOT 17) FFR (P_d/P_a) vs v_distal, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['v_distal'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='v_distal', y_col='P_d/P_a', color_col='HSR',
        x_label='distal velocity [cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=False,
        show_singletons=True
    )