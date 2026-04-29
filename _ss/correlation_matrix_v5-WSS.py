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

### Subset for WSS + Target Variables ###
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary.csv')

n_bins = 5
cmap = plt.get_cmap('Reds', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

columns_of_interest = [
    # WSS variables in the order you specified:
    'WSS', 
    'WSS_TE', 
    'WSS_LE', 
    'WSS_BIF', 
    'WSS_AVG_AREA',
    'WSS_TE_AREA', 
    'WSS_LE_AREA', 
    'WSS_AREA_BIFUR',
    # target variables in the order you specified:
    'HMR', 
    'BMR/HMR', 
    'iFR', 
    'FFR', 
    'CFR', 
    'CFR/FFR'
]

# Filter out columns that might not exist in df_full
existing_cols = [col for col in columns_of_interest if col in df_full.columns]

# Subset the DataFrame to just these columns
df_subset = df_full[existing_cols]

# Convert to numeric if needed (coerce any weird strings to NaN)
df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

# Create correlation matrix on these columns
corr_subset = np.abs(df_subset.corr())
mask_subset = np.eye(corr_subset.shape[0], dtype=bool) | corr_subset.isnull()

plt.figure(figsize=(10, 8))
ax2 = sns.heatmap(
    corr_subset, mask=mask_subset, annot=True,
    cmap=cmap, norm=norm, cbar_kws={"shrink": 0.8}
)
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=45, ha="center")
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=7)
for (i, j), value in np.ndenumerate(mask_subset):
    if value:
        ax2.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
plt.title("Correlation Heatmap (WSS + Target Variables Only)", fontsize=9)
plt.show()

# Build p-value matrix for this subset
cols_sub = df_subset.columns
pval_matrix_sub = pd.DataFrame(np.nan, index=cols_sub, columns=cols_sub)

for i in range(len(cols_sub)):
    for j in range(len(cols_sub)):
        if i == j:
            pval_matrix_sub.iloc[i, j] = np.nan
        else:
            sub_df2 = df_subset[[cols_sub[i], cols_sub[j]]].dropna(how='any')
            if len(sub_df2) < 2:
                continue
            else:
                _, pval_sub = pearsonr(sub_df2[cols_sub[i]], sub_df2[cols_sub[j]])
                pval_matrix_sub.iloc[i, j] = pval_sub

mask_pval_sub = np.eye(pval_matrix_sub.shape[0], dtype=bool) | pval_matrix_sub.isnull()

plt.figure(figsize=(10, 8))
ax2_pval = sns.heatmap(
    pval_matrix_sub,
    mask=mask_pval_sub,
    annot=True,
    cmap="Blues_r",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9},
)
ax2_pval.xaxis.set_ticks_position('top')
ax2_pval.xaxis.set_label_position('top')
ax2_pval.set_xticklabels(ax2_pval.get_xticklabels(), fontsize=7, rotation=45, ha="center")
ax2_pval.set_yticklabels(ax2_pval.get_yticklabels(), fontsize=7)
plt.title("P-Values (WSS + Target Variables)")
for (m, n), val in np.ndenumerate(mask_pval_sub):
    if val:
        ax2_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))
plt.show()

print(f"Rows in df_subset: {df_subset.shape[0]}")

# Pairwise plot for the subset
sns.pairplot(df_subset)
plt.suptitle("Pairwise Plot (WSS + Target Variables Only)", y=1.02)
plt.show()
