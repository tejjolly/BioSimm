#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes: Removing iFR and changing P_d/P_a name in place to FFR
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
FONT_SIZE = 6.5
FONT_TITLE = FONT_SIZE + 1
FONT_TICKS = FONT_SIZE
FONT_ANNOT = FONT_SIZE - 1

correlation_switch = True
pvalue_switch = True
pairgrid_switch = False  # Toggle if you want to display the PairGrid
WSS_switch = True # True to include  WSS
no_LCX_switch = False # True to turn off LCX simulations
no_stenosis_switch = True # True to turn off 0% stenosis (healthy) simulations

# Load the data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data_manuscript.csv')

# Keep 'Condition' in string form (not numeric codes)
df_full['Condition'] = df_full['Condition'].astype(str)
df_full = df_full[df_full['Condition'] == 'Hyperemic']
df_full = df_full[df_full['source'] == 'mine']

if no_stenosis_switch:
    df_full = df_full[df_full['Stenosis Percentage'] > 0.2]

if no_LCX_switch:
    df_full = df_full[df_full['Location'] == 'LAD']

# # Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
# df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic',
#                           df_full['P_d/P_a'], np.nan)
# df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic',
#                           df_full['P_d/P_a'], np.nan)

df_full = df_full.rename(columns={'P_d/P_a': 'FFR'})

# Normalize selected WSS features by WSS_LMB
normalize_cols = ['WSS_TE', 'WSS_LE', 'WSS_Bif', 'WSS_TE_min', 'WSS_LE_min', 'WSS_min']
for col in normalize_cols:
    if col in df_full.columns and 'WSS_LMB' in df_full.columns:
        df_full[col] = df_full[col] / df_full['WSS_LMB']


# Drop unused columns
# df_full.drop(columns=['P_d/P_a'], inplace=True)
df_full.drop(columns=['Condition', 'Geometry Number', 'R_micro','R_scale',
                      'discord','R_total'], inplace=True, errors='ignore')
if WSS_switch:
    # Rename columns
    df_full.rename(columns={
        'Stenosis Percentage': 'Stenosis',
        'Average Flow': 'Average Flow',
        'WSS_Area_Bifur': 'WSS_Bif_Area',
        'WSS_Area_Bifur_min': 'WSS_Bif_Area_min'
    }, inplace=True)
else:
    df_full.drop(columns=df_full.filter(like='WSS').columns, inplace=True, errors='ignore')
    FONT_SIZE = 10
    FONT_TITLE = FONT_SIZE + 1
    FONT_TICKS = FONT_SIZE
    FONT_ANNOT = FONT_SIZE - 1
# -------------------- CORRELATION HEATMAP -------------------- #
if correlation_switch:
    numeric_df = df_full.select_dtypes(include=[np.number])
    # corr_matrix = np.abs(numeric_df.corr())
    corr_matrix = numeric_df.corr()
    mask = np.eye(corr_matrix.shape[0], dtype=bool)
    mask |= corr_matrix.isnull()

    n_bins = 8
    cmap = plt.get_cmap('bwr', n_bins)
    levels = np.linspace(-1, 1, n_bins + 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        corr_matrix, mask=mask, annot=True,
        fmt = '.2f', cmap=cmap, norm=norm,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": FONT_ANNOT}
    )
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_TICKS, rotation=45, ha="left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_TICKS)

    for (i, j), value in np.ndenumerate(mask):
        if value:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

    # plt.title("Correlation Heatmap (iFR and FFR Split)", fontsize=FONT_TITLE)
    plt.show()

    print(f"Total samples (rows) in df_full for correlation heatmap: {df_full.shape[0]}")

# -------------------- P‑VALUE HEATMAP WITH DISCRETE BINS -------------------- #
if pvalue_switch:
    df_numeric = df_full.select_dtypes(include=[np.number])
    cols = df_numeric.columns
    pval_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            sub = df_numeric[[cols[i], cols[j]]].dropna()
            if len(sub) > 1:
                _, pval = pearsonr(sub[cols[i]], sub[cols[j]])
                pval_matrix.iloc[i, j] = pval

    mask_pval = np.eye(len(cols), dtype=bool) | pval_matrix.isnull()

    levels = np.array([0, 0.001, 0.01, 0.05, 0.1, 1.0])
    n_bins = len(levels) - 1

    # any Matplotlib cmap can be quantised this way
    cmap_p = plt.get_cmap("Reds_r", n_bins)
    norm_p = BoundaryNorm(levels, ncolors=cmap_p.N, clip=True)

    plt.figure(figsize=(12, 10))
    ax_p = sns.heatmap(
        pval_matrix,
        mask=mask_pval,
        cmap=cmap_p,
        norm=norm_p,
        fmt=".2g",
        annot=True,
        cbar_kws=dict(
            shrink=0.8,
            ticks=levels,
            format="%.3f"    # tick labels like 0.000 / 0.010 / 0.050 …
        ),
        annot_kws={"size": FONT_ANNOT}
    )

    ax_p.xaxis.set_ticks_position('top')
    ax_p.xaxis.set_label_position('top')
    ax_p.set_xticklabels(ax_p.get_xticklabels(), fontsize=FONT_TICKS, rotation=45, ha="center")
    ax_p.set_yticklabels(ax_p.get_yticklabels(), fontsize=FONT_TICKS)

    # mask diagonal with black squares (optional)
    for (i, j), m in np.ndenumerate(mask_pval):
        if m:
            ax_p.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

    plt.title("P‑value heatmap (discrete bins)", fontsize=FONT_TITLE)
    plt.show()

    # -------------------- Strong Correlation Pairs -------------------- #
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.4
    alpha = 0.05

    strong_corr = upper.stack()
    strong_corr = strong_corr[strong_corr.abs() > threshold]

    # (optional) sort by |r| so strong negatives appear near the top too
    strong_corr = strong_corr.reindex(strong_corr.abs().sort_values(ascending=False).index)

    print(f"\nStrong Correlations (|r| > {threshold} AND p < {alpha}):")
    for (var1, var2), corr_val in strong_corr.items():
        p = pval_matrix.loc[var1, var2]
        if pd.notna(p) and p < alpha:
            print(f"{var1:20} <-> {var2:20} | r = {corr_val:.2f}, p = {p:.3g}")

# -------------------- Optional PairGrid -------------------- #
df_subset = df_full.select_dtypes(include=[np.number]).copy()
df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
min_cols = [col for col in df_subset.columns if '_min' in col]
for col in min_cols:
    df_subset[col] = np.log1p(df_subset[col])

df_plot = df_subset.dropna(subset=["HSR"])
vars_to_plot = df_plot.columns.tolist()

# if pairgrid_switch:
#     def scatter_cmap_marker(x, y, data=None, c=None, cmap=None, norm=None,
#                             edgecolor="k", s=20, **kwargs):
#         if "color" in kwargs:
#             kwargs.pop("color")
#
#         if data is None or ("CFR" not in data.columns) or ("FFR" not in data.columns):
#             plt.scatter(x, y, c=c, cmap=cmap, norm=norm, edgecolors=edgecolor, s=s, **kwargs)
#             return
#
#         cat_definitions = {
#             "cat1": {"mask": (data['CFR'] >= 2) & (data['FFR'] >= 0.8), "marker": "o"},
#             "cat2": {"mask": (data['CFR'] >= 2) & (data['FFR'] < 0.8), "marker": "^"},
#             "cat3": {"mask": (data['CFR'] < 2) & (data['FFR'] >= 0.8), "marker": "^"},
#             "cat4": {"mask": (data['CFR'] < 2) & (data['FFR'] < 0.8), "marker": "o"},
#         }
#
#         for cat_info in cat_definitions.values():
#             mask = cat_info["mask"].values
#             plt.scatter(
#                 x[mask], y[mask],
#                 c=None if c is None else c[mask],
#                 cmap=cmap, norm=norm,
#                 marker=cat_info["marker"],
#                 edgecolors=edgecolor, s=s,
#                 label=None, **kwargs
#             )
#
#     g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)
#     g.map_diag(sns.kdeplot)
#     min_val, max_val = df_plot["HSR"].min(), df_plot["HSR"].max()
#     boundaries = np.linspace(min_val, max_val, 6)
#     cmap_pg = plt.get_cmap("RdYlGn_r", 5)
#     norm_pg = mcolors.BoundaryNorm(boundaries, ncolors=cmap_pg.N, clip=True)
#
#     g.map_lower(scatter_cmap_marker, data=df_plot, c=df_plot["HSR"],
#                 cmap=cmap_pg, norm=norm_pg, edgecolor="k", s=20)
#
#     fig = g.fig
#     cax = fig.add_axes([.975, 0.3, 0.02, 0.4])
#     sm = plt.cm.ScalarMappable(norm=norm_pg, cmap=cmap_pg)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.set_ticks(boundaries)
#     cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])
#     cbar.ax.tick_params(labelsize=12 * 4)
#     cbar.set_label("HSR [mmHg/cm/s]", fontsize=14 * 4)
#
#     plt.show()
#

if pairgrid_switch:
    def scatter_cmap_basic(x, y, data=None, c=None, cmap=None, norm=None,
                           edgecolor="k", s=20, **kwargs):
        if "color" in kwargs:
            kwargs.pop("color", None)
        plt.scatter(x, y, c=c, cmap=cmap, norm=norm,
                    edgecolors=edgecolor, s=s, **kwargs)

    g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)
    g.map_diag(sns.histplot)

    min_val, max_val = df_plot["HSR"].min(), df_plot["HSR"].max()
    boundaries = np.linspace(min_val, max_val, 6)
    cmap_pg = plt.get_cmap("RdYlGn_r", 5)
    norm_pg = mcolors.BoundaryNorm(boundaries, ncolors=cmap_pg.N, clip=True)

    g.map_upper(scatter_cmap_basic, data=df_plot, c=df_plot["HSR"],
                cmap=cmap_pg, norm=norm_pg, edgecolor="k", s=20)
    g.map_lower(scatter_cmap_basic, data=df_plot, c=df_plot["HSR"],
                cmap=cmap_pg, norm=norm_pg, edgecolor="k", s=20)


    fig = g.fig
    cax = fig.add_axes([.975, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(norm=norm_pg, cmap=cmap_pg)
    sm.set_array([])
    # cbar = fig.colorbar(sm, cax=cax)
    # cbar.set_ticks(boundaries)
    # cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])
    # cbar.ax.tick_params(labelsize=12 * 4)
    # cbar.set_label("HSR [mmHg/cm/s]", fontsize=14 * 4)

    plt.show()
