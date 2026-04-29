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

# ---- Global Font Size Control ----
FONT_SIZE = 6
FONT_TITLE = FONT_SIZE + 1
FONT_TICKS = FONT_SIZE
FONT_ANNOT = FONT_SIZE - 1

# Load the data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary2.csv')

# Keep 'Condition' in string form (not numeric codes)
df_full['Condition'] = df_full['Condition'].astype(str)

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic',
                          df_full['P_d/P_a'], np.nan)
df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic',
                          df_full['P_d/P_a'], np.nan)

# Drop unused columns
df_full.drop(columns=['P_d/P_a'], inplace=True)
df_full.drop(columns=['Condition', 'Geometry Number'], inplace=True)

# Rename columns
df_full.rename(columns={
    'Stenosis Percentage': 'Stenosis',
    'Average Flow': 'Flow',
    'Rtotal_cor Value': 'R. mult.',
    'WSS_Area_Bifur': 'WSS_Bif_Area',
}, inplace=True)

# Quick look
print(df_full.head())
print(df_full.info())
print(df_full.isnull().sum())

# -------------------- CORRELATION HEATMAP -------------------- #
numeric_df = df_full.select_dtypes(include=[np.number])
corr_matrix = np.abs(numeric_df.corr())
mask = np.eye(corr_matrix.shape[0], dtype=bool)
mask |= corr_matrix.isnull()

n_bins = 10
cmap = plt.get_cmap('Blues_r', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
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

plt.title("Correlation Heatmap (iFR and FFR Split)", fontsize=FONT_TITLE)
plt.show()

print(f"Total samples (rows) in df_full for correlation heatmap: {df_full.shape[0]}")

# -------------------- P-VALUE MATRIX -------------------- #
df_numeric = df_full.select_dtypes(include=[np.number])
cols = df_numeric.columns
pval_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        if i == j:
            pval_matrix.iloc[i, j] = np.nan
        else:
            sub_df = df_numeric[[cols[i], cols[j]]].dropna(how='any')
            if len(sub_df) < 2:
                continue
            else:
                _, pval = pearsonr(sub_df[cols[i]], sub_df[cols[j]])
                pval_matrix.iloc[i, j] = pval

mask_pval = np.eye(pval_matrix.shape[0], dtype=bool) | pval_matrix.isnull()

plt.figure(figsize=(12, 10))
ax_pval = sns.heatmap(
    pval_matrix,
    mask=mask_pval,
    fmt = '.2f',
    annot=True,
    cmap="Blues_r",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": FONT_ANNOT}
)
ax_pval.xaxis.set_ticks_position('top')
ax_pval.xaxis.set_label_position('top')
ax_pval.set_xticklabels(ax_pval.get_xticklabels(), fontsize=FONT_TICKS, rotation=45, ha="center")
ax_pval.set_yticklabels(ax_pval.get_yticklabels(), fontsize=FONT_TICKS)

for (m, n), val in np.ndenumerate(mask_pval):
    if val:
        ax_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))

plt.title("P-Values Heatmap (Pairwise Overlap)", fontsize=FONT_TITLE)
plt.show()

print(f"Samples (rows) used in df_numeric_pval for p-value heatmap: {pval_matrix.shape[0]}")



# Get upper triangle of the correlation matrix without the diagonal
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Flatten and filter
threshold = 0.8
strong_corr = upper.stack().sort_values(ascending=False)
strong_corr = strong_corr[strong_corr > threshold]

# Print nicely
print(f"\nStrong Correlations (|r| > {threshold}):")
for (var1, var2), corr_val in strong_corr.items():
    print(f"{var1:20} <-> {var2:20} | r = {corr_val:.2f}")

