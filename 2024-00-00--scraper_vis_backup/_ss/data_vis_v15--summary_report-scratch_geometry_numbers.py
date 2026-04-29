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
        'axes.labelsize': 16,       # Axis label font size
        'axes.titlesize': 16,       # Title font size
        'xtick.labelsize': 16,      # X-tick label font size
        'ytick.labelsize': 16,      # Y-tick label font size
        'legend.fontsize': 16,       # Legend font size
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

# -------------------------------------------------------------
# QUICK-AND-DIRTY SCATTER WITH POINT LABELS
# -------------------------------------------------------------
def make_scatter_with_labels(
    data, x_col, y_col, color_col,
    x_label='', y_label='', title='',
    label_col='Geometry Number',
    label_every=1,           # set >1 to label every N-th point
    label_fontsize=9,
    cmap_name='viridis',
    alpha_scatter=0.8,
    s_scatter=50,
    savefig=False, outdir='images', dpi=600
):
    """
    Simple scatter + text labels for each point.

    Parameters
    ----------
    data : pd.DataFrame        – already filtered
    x_col, y_col, color_col :  – column names for x, y, colour
    label_col : str            – column whose value is printed next to each point
    label_every : int          – 1 = label all, 2 = every 2nd, etc.
    """

    # -----------------------------------------------------------------
    df_plot = data[[x_col, y_col, color_col, label_col]].dropna().copy()
    if df_plot.empty:
        print("Nothing to plot – aborting.")
        return

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(df_plot[color_col].min(),
                         df_plot[color_col].max())

    fig, ax = plt.subplots(figsize=(6,4))
    sc = ax.scatter(df_plot[x_col], df_plot[y_col],
                    c=df_plot[color_col],
                    cmap=cmap, norm=norm,
                    s=s_scatter, edgecolors='k', alpha=alpha_scatter)

    # --- add labels --------------------------------------------------
    for i, (x, y, lab) in enumerate(zip(df_plot[x_col],
                                        df_plot[y_col],
                                        df_plot[label_col])):
        if i % label_every == 0:
            ax.text(x, y, str(lab),
                    fontsize=label_fontsize,
                    ha='center', va='center',
                    color='black',
                    path_effects=[plt.matplotlib.patheffects.withStroke(
                                  linewidth=1.5, foreground='white')])

    # --- cosmetics ---------------------------------------------------
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_col)
    ax.grid(False)
    fig.tight_layout()

    if savefig:
        fname = (f"{y_col}_vs_{x_col}_labels.png"
                 if isinstance(savefig, bool) else str(savefig))
        fig.savefig(f"{outdir}/{fname}", dpi=dpi,
                    bbox_inches='tight', transparent=True)
        print(f"Saved → {outdir}/{fname}")

    plt.show()


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
            {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
            {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=True,
        show_singletons=True,
        # savefig=True,

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