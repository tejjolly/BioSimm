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
FONT_SIZE = 6
FONT_TITLE = FONT_SIZE + 1
FONT_TICKS = FONT_SIZE
FONT_ANNOT = FONT_SIZE - 1

pairgrid_switch = True  # Toggle if you want to display the PairGrid

# Load the data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data.csv')

# Keep 'Condition' in string form (not numeric codes)
df_full['Condition'] = df_full['Condition'].astype(str)

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic',
                          df_full['P_d/P_a'], np.nan)
df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic',
                          df_full['P_d/P_a'], np.nan)

# Normalize selected WSS features by WSS_LMB
normalize_cols = ['WSS_TE', 'WSS_LE', 'WSS_Bif', 'WSS_TE_min', 'WSS_LE_min', 'WSS_min']
for col in normalize_cols:
    if col in df_full.columns and 'WSS_LMB' in df_full.columns:
        df_full[col] = df_full[col] / df_full['WSS_LMB']


# Drop unused columns
df_full.drop(columns=['P_d/P_a'], inplace=True)
df_full.drop(columns=['Condition', 'Geometry Number'], inplace=True, errors='ignore')

# Rename columns
df_full.rename(columns={
    'Stenosis Percentage': 'Stenosis',
    'Average Flow': 'Flow',
    'Rtotal_cor Value': 'R. mult.',
    'WSS_Area_Bifur': 'WSS_Bif_Area',
    'WSS_Area_Bifur_min': 'WSS_Bif_Area_min'
}, inplace=True)

# # -------------------- CORRELATION HEATMAP -------------------- #
# numeric_df = df_full.select_dtypes(include=[np.number])
# corr_matrix = np.abs(numeric_df.corr())
# mask = np.eye(corr_matrix.shape[0], dtype=bool)
# mask |= corr_matrix.isnull()
#
# n_bins = 10
# cmap = plt.get_cmap('Blues_r', n_bins)
# levels = np.linspace(0, 1, n_bins + 1)
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#
# plt.figure(figsize=(12, 10))
# ax = sns.heatmap(
#     corr_matrix, mask=mask, annot=True,
#     fmt = '.2f', cmap=cmap, norm=norm,
#     cbar_kws={"shrink": 0.8}, annot_kws={"size": FONT_ANNOT}
# )
# ax.xaxis.set_ticks_position('top')
# ax.xaxis.set_label_position('top')
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_TICKS, rotation=45, ha="left")
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_TICKS)
#
# for (i, j), value in np.ndenumerate(mask):
#     if value:
#         ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
#
# plt.title("Correlation Heatmap (iFR and FFR Split)", fontsize=FONT_TITLE)
# plt.show()
#
# print(f"Total samples (rows) in df_full for correlation heatmap: {df_full.shape[0]}")
#
# # -------------------- P-VALUE MATRIX -------------------- #
# df_numeric = df_full.select_dtypes(include=[np.number])
# cols = df_numeric.columns
# pval_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
#
# for i in range(len(cols)):
#     for j in range(len(cols)):
#         if i == j:
#             pval_matrix.iloc[i, j] = np.nan
#         else:
#             sub_df = df_numeric[[cols[i], cols[j]]].dropna(how='any')
#             if len(sub_df) < 2:
#                 continue
#             else:
#                 _, pval = pearsonr(sub_df[cols[i]], sub_df[cols[j]])
#                 pval_matrix.iloc[i, j] = pval
#
# mask_pval = np.eye(pval_matrix.shape[0], dtype=bool) | pval_matrix.isnull()
#
# plt.figure(figsize=(12, 10))
# ax_pval = sns.heatmap(
#     pval_matrix,
#     mask=mask_pval,
#     fmt = '.2f',
#     annot=True,
#     cmap="Blues_r",
#     cbar_kws={"shrink": 0.8},
#     annot_kws={"size": FONT_ANNOT}
# )
# ax_pval.xaxis.set_ticks_position('top')
# ax_pval.xaxis.set_label_position('top')
# ax_pval.set_xticklabels(ax_pval.get_xticklabels(), fontsize=FONT_TICKS, rotation=45, ha="center")
# ax_pval.set_yticklabels(ax_pval.get_yticklabels(), fontsize=FONT_TICKS)
#
# for (m, n), val in np.ndenumerate(mask_pval):
#     if val:
#         ax_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))
#
# plt.title("P-Values Heatmap (Pairwise Overlap)", fontsize=FONT_TITLE)
# plt.show()
#
# print(f"Samples (rows) used in df_numeric_pval for p-value heatmap: {pval_matrix.shape[0]}")
#
# # -------------------- Strong Correlation Pairs -------------------- #
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# threshold = 0.8
# strong_corr = upper.stack().sort_values(ascending=False)
# strong_corr = strong_corr[strong_corr > threshold]
#
# print(f"\nStrong Correlations (|r| > {threshold}):")
# for (var1, var2), corr_val in strong_corr.items():
#     print(f"{var1:20} <-> {var2:20} | r = {corr_val:.2f}")

# -------------------- Optional PairGrid -------------------- #
df_subset = df_full.select_dtypes(include=[np.number]).copy()
df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
min_cols = [col for col in df_subset.columns if '_min' in col]
for col in min_cols:
    df_subset[col] = np.log1p(df_subset[col])

df_plot = df_subset.dropna(subset=["HSR"])
vars_to_plot = df_plot.columns.tolist()

if pairgrid_switch:
    def scatter_cmap_marker(x, y, data=None, c=None, cmap=None, norm=None,
                            edgecolor="k", s=20, **kwargs):
        if "color" in kwargs:
            kwargs.pop("color")

        if data is None or ("CFR" not in data.columns) or ("FFR" not in data.columns):
            plt.scatter(x, y, c=c, cmap=cmap, norm=norm, edgecolors=edgecolor, s=s, **kwargs)
            return

        cat_definitions = {
            "cat1": {"mask": (data['CFR'] >= 2) & (data['FFR'] >= 0.8), "marker": "o"},
            "cat2": {"mask": (data['CFR'] >= 2) & (data['FFR'] < 0.8), "marker": "^"},
            "cat3": {"mask": (data['CFR'] < 2) & (data['FFR'] >= 0.8), "marker": "^"},
            "cat4": {"mask": (data['CFR'] < 2) & (data['FFR'] < 0.8), "marker": "o"},
        }

        for cat_info in cat_definitions.values():
            mask = cat_info["mask"].values
            plt.scatter(
                x[mask], y[mask],
                c=None if c is None else c[mask],
                cmap=cmap, norm=norm,
                marker=cat_info["marker"],
                edgecolors=edgecolor, s=s,
                label=None, **kwargs
            )

    g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)
    min_val, max_val = df_plot["HSR"].min(), df_plot["HSR"].max()
    boundaries = np.linspace(min_val, max_val, 6)
    cmap_pg = plt.get_cmap("RdYlGn_r", 5)
    norm_pg = mcolors.BoundaryNorm(boundaries, ncolors=cmap_pg.N, clip=True)

    g.map_lower(scatter_cmap_marker, data=df_plot, c=df_plot["HSR"],
                cmap=cmap_pg, norm=norm_pg, edgecolor="k", s=20)

    fig = g.fig
    cax = fig.add_axes([.975, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(norm=norm_pg, cmap=cmap_pg)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])
    cbar.ax.tick_params(labelsize=12 * 4)
    cbar.set_label("HSR [mmHg/cm/s]", fontsize=14 * 4)

    plt.show()
