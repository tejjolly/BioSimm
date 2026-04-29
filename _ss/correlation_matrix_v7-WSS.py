#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:52:41 2025

@author: tejjolly
"""

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
}, inplace=True)

# columns_of_interest = [
#     # WSS variables in the order you specified:
#     'WSS_LMB',
#     'WSS', 
#     'WSS_TE', 
#     'WSS_LE', 
#     'WSS_Bif', 
#     'WSS_Avg_Area',
#     'WSS_TE_Area', 
#     'WSS_LE_Area', 
#     'WSS_Area_Bifur',
#     # target variables in the order you specified:
#     'HMR', 
#     'BMR/HMR', 
#     'iFR', 
#     'FFR', 
#     'CFR', 
#     'CFR/FFR'
# ]
    
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
    'WSS_Area_Bifur',
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

# # Create correlation matrix on these columns
# corr_subset = np.abs(df_subset.corr())
# mask_subset = np.eye(corr_subset.shape[0], dtype=bool) | corr_subset.isnull()

# plt.figure(figsize=(10, 8))
# ax2 = sns.heatmap(
#     corr_subset, mask=mask_subset, annot=True,
#     cmap=cmap, norm=norm, cbar_kws={"shrink": 0.8}
# )
# ax2.xaxis.set_ticks_position('top')
# ax2.xaxis.set_label_position('top')
# ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=45, ha="center")
# ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=7)
# for (i, j), value in np.ndenumerate(mask_subset):
#     if value:
#         ax2.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
# plt.title("Correlation Heatmap (WSS + Target Variables Only)", fontsize=9)
# plt.show()

# # Build p-value matrix for this subset
# cols_sub = df_subset.columns
# pval_matrix_sub = pd.DataFrame(np.nan, index=cols_sub, columns=cols_sub)

# for i in range(len(cols_sub)):
#     for j in range(len(cols_sub)):
#         if i == j:
#             pval_matrix_sub.iloc[i, j] = np.nan
#         else:
#             sub_df2 = df_subset[[cols_sub[i], cols_sub[j]]].dropna(how='any')
#             if len(sub_df2) < 2:
#                 continue
#             else:
#                 _, pval_sub = pearsonr(sub_df2[cols_sub[i]], sub_df2[cols_sub[j]])
#                 pval_matrix_sub.iloc[i, j] = pval_sub

# mask_pval_sub = np.eye(pval_matrix_sub.shape[0], dtype=bool) | pval_matrix_sub.isnull()

# plt.figure(figsize=(10, 8))
# ax2_pval = sns.heatmap(
#     pval_matrix_sub,
#     mask=mask_pval_sub,
#     annot=True,
#     cmap="Blues_r",
#     cbar_kws={"shrink": 0.8},
#     annot_kws={"size": 9},
# )
# ax2_pval.xaxis.set_ticks_position('top')
# ax2_pval.xaxis.set_label_position('top')
# ax2_pval.set_xticklabels(ax2_pval.get_xticklabels(), fontsize=7, rotation=45, ha="center")
# ax2_pval.set_yticklabels(ax2_pval.get_yticklabels(), fontsize=7)
# plt.title("P-Values (WSS + Target Variables)")
# for (m, n), val in np.ndenumerate(mask_pval_sub):
#     if val:
#         ax2_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))
# plt.show()

# print(f"Rows in df_subset: {df_subset.shape[0]}")
# print(f"Rows in df_subset: {df_subset}")

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# 1) Decide columns to plot
# vars_to_plot = ["WSS_LMB", "WSS", "WSS_TE", "WSS_LE", "FFR", "CFR", "CFR/FFR"]

vars_to_plot = [
    'WSS_LMB',
    'WSS', 
    'WSS_TE', 
    'WSS_LE', 
    'WSS_Bif', 
    'WSS_Avg_Area',
    'WSS_TE_Area', 
    'WSS_LE_Area', 
    'WSS_Area_Bifur',
    ## target variables
    'HMR', 
    'BMR/HMR', 
    'iFR', 
    'FFR', 
    'CFR', 
    'CFR/FFR'
    ]

# 2) Drop rows where HSR is NaN
df_plot = df_subset.dropna(subset=["HSR"])

# 3) Build the PairGrid
# g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)
g = sns.PairGrid(df_plot, diag_sharey=False)


# 4) Helper for scatter with color
def scatter_cmap(x, y, c=None, cmap=None, norm=None, **kwargs):
    # Remove seaborn’s default color
    if "color" in kwargs:
        kwargs.pop("color")
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

# 5) Map your custom scatter
g.map_lower(scatter_cmap, c=df_plot["HSR"], cmap=cmap, norm=norm, edgecolor="k")
g.map_diag(sns.histplot, fill=True)

# 6) Add the discrete colorbar
fig = g.fig
cax = fig.add_axes([.975, 0.3, 0.02, 0.4])  # x, y, width, height (in figure fraction)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # dummy array for older mpl versions
cbar = fig.colorbar(sm, cax=cax)

# Optionally label the colorbar:
cbar.set_label("HSR [[mmHg/cm/s]")

# Optionally set bin-edge ticks explicitly:
cbar.set_ticks(boundaries)
cbar.set_ticklabels([f"{v:.2f}" for v in boundaries])  # or custom labels

plt.show()


