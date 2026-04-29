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
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

# ---- Global Font Size Control ----
FONT_SIZE = 12
FONT_TITLE = FONT_SIZE + 1
FONT_TICKS = FONT_SIZE
FONT_ANNOT = FONT_SIZE - 1

pairgrid_switch = True  # Toggle if you want to display the PairGrid

# Load the data
# Reload the raw data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data.csv')

# Keep only the columns of interest
keep_cols = [
    'Condition', 'Geometry Number',  'P_Loss_Coeff', 'HMR', 'BMR/HMR',
    'P_d/P_a', 'CFR', 'CFR/FFR', 'discord'
]
df_full = df_full[keep_cols]

# # Keep 'Condition' in string form (not numeric codes)
# df_full['Condition'] = df_full['Condition'].astype(str)

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic',
                          df_full['P_d/P_a'], np.nan)

df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic',
                          df_full['P_d/P_a'], np.nan)

# Drop unused columns
df_full.drop(columns=['P_d/P_a'], inplace=True)
df_full.drop(columns=['Condition', 'Geometry Number','iFR'], inplace=True, errors='ignore')

# Rename columns
df_full.rename(columns={
    'Stenosis Percentage': 'Stenosis',
}, inplace=True)


# -------------------- Optional PairGrid -------------------- #
if pairgrid_switch:

    # ---------- 1)  Pre‑clean dataframe ----------
    vars_to_plot = ['P_Loss_Coeff', 'HMR', 'BMR/HMR', 'FFR', 'CFR', 'CFR/FFR']
    df_pg = (
        df_full
        .dropna(subset=vars_to_plot + ['discord'])   # need all vars + discord (not used, just filtered)
        .query("FFR.notna()")                        # keep only Hyperemic rows
        .copy()
    )

    for col in df_pg.filter(like="_min"):
        df_pg[col] = np.log1p(df_pg[col])

    # ---------- 2)  Custom scatter with CFR/FFR marker categories ----------
    def scatter_cmap_marker(x, y, *, data=None, c=None, cmap=None, norm=None,
                            edgecolor="k", s=25, **kwargs):
        kwargs.pop("color", None)  # remove seaborn-injected hue color if present

        if isinstance(c, pd.Series):
            c = c.loc[x.index]
        sub = data.loc[x.index]

        categories = [
            ((sub['CFR'] >= 2) & (sub['FFR'] >= 0.8), "o"),
            ((sub['CFR'] >= 2) & (sub['FFR'] <  0.8), "^"),
            ((sub['CFR'] <  2) & (sub['FFR'] >= 0.8), "^"),
            ((sub['CFR'] <  2) & (sub['FFR'] <  0.8), "o"),
        ]

        for mask, marker in categories:
            plt.scatter(
                x[mask], y[mask],
                c=None if c is None else c[mask],
                cmap=cmap, norm=norm,
                marker=marker,
                edgecolors=edgecolor,
                s=s,
                **kwargs
            )

    # ---------- 3)  Build PairGrid ----------
    g = sns.PairGrid(df_pg, vars=vars_to_plot)
    g.map_diag(sns.histplot)

    boundaries = np.linspace(df_pg["P_Loss_Coeff"].min(),
                             df_pg["P_Loss_Coeff"].max(), 6)
    cmap_pg = plt.get_cmap("RdYlGn_r", 5)
    norm_pg = mcolors.BoundaryNorm(boundaries, cmap_pg.N)

    g.map_lower(
        scatter_cmap_marker,
        data=df_pg,
        c=df_pg["P_Loss_Coeff"],
        cmap=cmap_pg,
        norm=norm_pg,
        edgecolor="k",
        s=25
    )

    g.map_upper(lambda *args, **kwargs: None)

    # ---------- 5)  Marker Legend ----------
    marker_legend = [
        mlines.Line2D([], [], marker='o', color='gray', linestyle='None',
                      markeredgecolor='k', label='CFR & FFR Accord'),
        mlines.Line2D([], [], marker='^', color='gray', linestyle='None',
                      markeredgecolor='k', label='CFR & FFR Discord'),
    ]

    g.fig.legend(
        handles=marker_legend,
        loc='upper right', bbox_to_anchor=(0.995, 0.8),
        fontsize=FONT_SIZE, frameon=False
    )
    g.fig.tight_layout()
    g.fig.set_size_inches(12, 12)  # same for every figure
    plt.show()
