
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
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary.csv')

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
# Select only numeric columns to compute the correlation matrix
numeric_df = df_full.select_dtypes(include=[np.number])
# corr_matrix = numeric_df.corr()  # or use .abs() if desired
corr_matrix = np.abs(numeric_df.corr())  # or use .abs() if desired
mask = np.eye(corr_matrix.shape[0], dtype=bool)
mask |= corr_matrix.isnull()

# For a two-sided correlation range [-1,1]
n_bins = 10
# cmap = plt.get_cmap('coolwarm', n_bins)
cmap = plt.get_cmap('Blues_r', n_bins)
# levels = np.linspace(-1, 1, n_bins + 1)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    corr_matrix, mask=mask, annot=True,
    cmap=cmap, norm=norm, cbar_kws={"shrink": 0.8}
)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Adjust the font size of x and y tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=45, ha="center",)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

# Color the masked cells black
for (i, j), value in np.ndenumerate(mask):
    if value:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))

plt.title("Correlation Heatmap (iFR and FFR Split)", fontsize=7)
plt.show()

print(f"Total samples (rows) in df_full for correlation heatmap: {df_full.shape[0]}")

df_numeric = df_full.select_dtypes(include=[np.number])  # numeric only
cols = df_numeric.columns

# -- BUILD P-VALUE MATRIX PAIR-BY-PAIR ---------------------- #
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

ax_pval.set_xticklabels(ax_pval.get_xticklabels(), fontsize=5, rotation=45, ha="center",)
ax_pval.set_yticklabels(ax_pval.get_yticklabels(), fontsize=5)

plt.title("P-Values Heatmap (Pairwise Overlap)")

for (m, n), val in np.ndenumerate(mask_pval):
    if val:  # if the cell is masked
        ax_pval.add_patch(plt.Rectangle((n, m), 1, 1, color='black'))

plt.show()

print(f"Samples (rows) used in df_numeric_pval for p-value heatmap: {pval_matrix.shape[0]}")

# sns.pairplot(df_numeric)
# plt.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# columns_of_interest = [
#     # WSS variables in the order you specified:
#     'WSS', 
#     'WSS_TE', 
#     'WSS_LE', 
#     'WSS_BIF', 
#     'WSS_AVG_AREA',
#     'WSS_TE_AREA', 
#     'WSS_LE_AREA', 
#     'WSS_AREA_BIFUR',
#     # target variables in the order you specified:
#     'HMR', 
#     'BMR/HMR', 
#     'iFR', 
#     'FFR', 
#     'CFR', 
#     'CFR/FFR'
# ]

# # Filter out columns that might not exist in df_full
# existing_cols = [col for col in columns_of_interest if col in df_full.columns]

# # Subset the DataFrame to just these columns
# df_subset = df_full[existing_cols]

# # Convert to numeric if needed (coerce any weird strings to NaN)
# df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

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

# # Pairwise plot for the subset
# sns.pairplot(df_subset)
# plt.show()

