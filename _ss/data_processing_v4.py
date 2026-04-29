#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:43:19 2025

@author: tejjolly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:18:28 2024

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

# Load the data
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/summary.csv')

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

# At this point, df_full has columns like:
#  iFR, FFR, Stenosis, Flow, R. mult., plus whatever else was originally there.

# -------------------- Quick look at the data --------------------
print(df_full.head())
print(df_full.info())
print(df_full.isnull().sum())

# -------------------- CORRELATION HEATMAP -------------------- #
corr_matrix = df_full.corr()  # or .abs() if you want absolute values
mask = np.eye(corr_matrix.shape[0], dtype=bool)
mask |= corr_matrix.isnull()

# For a two-sided correlation range [-1,1]
n_bins = 10
cmap = plt.get_cmap('coolwarm', n_bins)
levels = np.linspace(-1, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    corr_matrix, mask=mask, annot=True,
    cmap=cmap, norm=norm, cbar_kws={"shrink": 0.8}
)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Color the masked cells black
for (i, j), value in np.ndenumerate(mask):
    if value:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

plt.title("Correlation Heatmap (iFR and FFR Split)")
plt.show()

print(f"Total samples (rows) in df_full for correlation heatmap: {df_full.shape[0]}")

df_numeric = df_full.select_dtypes(include=[np.number])  # numeric only
cols = df_numeric.columns

# -- BUILD P-VALUE MATRIX PAIR-BY-PAIR --
pval_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        if i == j:
            # Diagonal is NaN
            pval_matrix.iloc[i, j] = np.nan
        else:
            # Focus on just the two columns
            sub_df = df_numeric[[cols[i], cols[j]]].dropna(how='any')
            if len(sub_df) < 2:
                # Not enough overlap to compute correlation => remains NaN
                continue
            else:
                # Compute Pearson r, p-value with the overlapping data
                _, pval = pearsonr(sub_df[cols[i]], sub_df[cols[j]])
                pval_matrix.iloc[i, j] = pval

# -- MASK & PLOT THE P-VALUE HEATMAP --
mask_pval = np.eye(pval_matrix.shape[0], dtype=bool) | pval_matrix.isnull()

plt.figure(figsize=(12, 10))
ax_pval = sns.heatmap(
    pval_matrix, 
    mask=mask_pval, 
    annot=True,
    cmap="Blues_r", 
    cbar_kws={"shrink": 0.8}
)
ax_pval.xaxis.set_ticks_position('top')
ax_pval.xaxis.set_label_position('top')
plt.title("P-Values Heatmap (Pairwise Overlap)")

for (m, n), val in np.ndenumerate(mask_pval):
    if val:  # if the cell is masked
        ax_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))

plt.show()

print(f"Samples (rows) used in df_numeric_pval for p-value heatmap: {pval_matrix.shape[0]}")


g = sns.pairplot(df_numeric)
for ax in g.axes.flatten():
    ax.tick_params(axis='x', which='both', labeltop=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
plt.suptitle("Pairwise Plot of Numeric Variables", y=1.02)
plt.show()
